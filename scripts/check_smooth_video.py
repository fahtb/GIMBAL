"""Headless tests of delayed masks, losses, stepping, output and ROI selection."""
import csv
import tempfile
import time
import threading
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import cv2
import numpy as np

from mask_display import MaskDisplay, overlay
from realtime_video import InferenceSlot, VideoDecoder, OutputWorker, run_realtime
from video_playback import select_box


def scene(index):
    frame = np.zeros((180,320,3),np.uint8)
    texture = np.random.default_rng(12).integers(40,240,(60,80,3),dtype=np.uint8)
    x,y = 40+index,60
    frame[y:y+60,x:x+80] = texture
    mask = np.zeros(frame.shape[:2],bool)
    mask[y:y+60,x:x+80] = True
    return frame,mask,[x,y,x+79,y+59]


def sam(index,status='TRACKING'):
    frame,mask,box = scene(index)
    return dict(seq=index,captured=index/60,frame=frame,mask=mask if status=='TRACKING' else None,
                box=box if status=='TRACKING' else None,status=status,valid=status=='TRACKING',
                inference_ms=60.,latency_ms=60.)


class Telemetry:
    def __init__(self,path): pass
    def submit(self,data): pass
    def close(self): pass


class SmoothTests(unittest.TestCase):
    def test_main_pause_steps_reselection_labels_quit(self):
        import camera
        workers=[]
        class FakeWorker:
            def __init__(self,slot,args):
                self.slot=slot;self.lock=threading.Lock();self.result=None;self.error=None
                self.ready=True;self.generation=0;self.active=False;self.commands=[];self.labels=[]
                self.command=None;self.clears=0
                slot.start_read.set();workers.append(self)
                self.thread=threading.Thread(target=self.run,daemon=True);self.thread.start()
            def select(self,packet,box):
                with self.lock:
                    self.generation+=1;self.command=packet;self.result=None;self.active=True
                    self.commands.append(packet[1])
            def clear(self):
                with self.lock:
                    self.generation+=1;self.active=False;self.result=None;self.command=None;self.clears+=1
            def label(self,correct): self.labels.append(correct)
            def run(self):
                last=-1
                while not self.slot.stop.wait(.002):
                    with self.lock:
                        generation=self.generation
                        packet=self.command or self.slot.get();self.command=None
                        active=self.active
                    if not active or packet is None or packet[0]==last: continue
                    time.sleep(.025)
                    i=round(packet[1]*60);result=sam(i)
                    result.update(seq=packet[0],frame=packet[2],captured=packet[1],generation=generation)
                    with self.lock:
                        if generation==self.generation: self.result=result
                    last=packet[0]
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path=root/'clip.avi'
            writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'MJPG'),60.,(320,180))
            for i in range(120): writer.write(scene(i)[0])
            writer.release()
            cfg=root/'config.json';cfg.write_text('{"history_frames":90,"max_mask_age":1,"writer_queue":120,"display_fps":60,"opencv_threads":2}')
            args=SimpleNamespace(source=str(path),start_frame=0,init_frame=0,resize_width=320,
                init_bbox=[40,60,80,60],csv=root/'frames.csv',telemetry=root/'target.json',
                save_video=root/'output.avi',profile=root/'profile.json',audit_masks=None,
                realtime_config=cfg,no_display=False,max_frames=None,loop=False,overlay_quality='full')
            keys=iter([(.35,'p'),(.5,'d'),(.7,'a'),(.9,'s'),(1.1,'w'),(1.15,'c'),(1.2,'p'),(1.5,'r'),(1.9,'q')])
            pending=[next(keys)];start=[None];sent=[]
            def wait(_):
                if start[0] is None: start[0]=time.perf_counter()
                if time.perf_counter()-start[0]>=pending[0][0]:
                    key=pending[0][1];sent.append(key);pending[0]=next(keys,(99,'q'));return ord(key)
                return -1
            with patch.object(camera,'Worker',FakeWorker),patch('cv2.namedWindow'),patch('cv2.imshow'), \
                 patch('cv2.destroyAllWindows'),patch('cv2.getWindowProperty',return_value=1), \
                 patch('cv2.waitKey',side_effect=wait),patch('realtime_video.select_box',return_value=[40,60,120,120]):
                run_realtime(args,Telemetry)
            self.assertEqual(sent,list('pdaswcprq'))
            self.assertEqual(workers[0].clears,1)
            self.assertEqual(workers[0].labels,[False,True])
            self.assertEqual(len(workers[0].commands),3)
            self.assertFalse(workers[0].thread.is_alive())
            with args.csv.open() as handle: rows=list(csv.DictReader(handle))
            self.assertTrue(any(r['pass']=='1' for r in rows))
            self.assertTrue(any(r['selected_bbox'] for r in rows))

    def test_delayed_mask_alignment_and_roi(self):
        display = MaskDisplay()
        ious=[]
        for i in range(40):
            f,truth,b = scene(i)
            result = display.step((i,i/60,f),sam(max(0,(i-3)//5*5)))
            self.assertIsNotNone(result['mask'])
            ious.append((result['mask'] & truth).sum()/(result['mask'] | truth).sum())
            self.assertAlmostEqual(result['dx'],(b[0]+b[2]-320)/2,delta=1.)
            rendered = overlay(f,result)
            self.assertEqual(rendered.shape,f.shape)
            self.assertTrue(np.array_equal(f[:20,:20],rendered[:20,:20]))
        self.assertGreater(np.mean(ious),.97)

    def test_loss_clears_mask_and_new_sam_can_recover(self):
        display=MaskDisplay()
        f,_,_=scene(0)
        self.assertTrue(display.step((0,0.,f),sam(0))['valid'])
        f,_,_=scene(1)
        result=display.step((1,1/60,f),sam(1,'LOST'))
        self.assertFalse(result['valid'])
        self.assertIsNone(result['mask'])
        f,_,_=scene(2)
        self.assertTrue(display.step((2,2/60,f),sam(2))['valid'])

    def test_expired_mask_is_not_exported(self):
        display=MaskDisplay(max_age=.05)
        for i in range(6):
            f,_,_=scene(i)
            result=display.step((i,i/60,f),sam(0))
        self.assertFalse(result['valid'])
        self.assertEqual(result['status'],'FLOW_UNCERTAIN')

    def test_decoder_pause_exact_seek_and_eof(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'clip.avi'
            writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'MJPG'),60.,(320,180))
            for i in range(12): writer.write(scene(i)[0])
            writer.release()
            args=SimpleNamespace(source=str(path),start_frame=0,init_frame=0,resize_width=320)
            slot=InferenceSlot()
            decoder=VideoDecoder(args,slot)
            slot.start_read.set()
            def wait(predicate):
                deadline=time.perf_counter()+3
                while not predicate():
                    self.assertIsNone(decoder.error)
                    self.assertLess(time.perf_counter(),deadline)
                    time.sleep(.002)
            try:
                wait(lambda: decoder.get() is not None)
                self.assertEqual(decoder.get()[1],0.)
                time.sleep(.05)
                self.assertEqual(decoder.get()[1],0.)
                decoder.seek(1);wait(lambda: decoder.get()[1]==1/60)
                decoder.seek(0);wait(lambda: decoder.get()[1]==0.)
                decoder.pause(False);wait(lambda: decoder.done)
                self.assertEqual(decoder.end_index,11)
            finally:
                slot.stop.set();decoder.thread.join(3)
            self.assertFalse(decoder.thread.is_alive())

    def test_output_fills_skips_and_drains(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            args=SimpleNamespace(telemetry=root/'t.json',save_video=root/'out.avi',audit_masks=None)
            output=OutputWorker(args,root/'out.csv',60.,Telemetry,20)
            for i in (0,3,4):
                result=sam(i)
                output.submit(0,i,i/60,result['frame'],result,.001)
            output.queue.put(('tail',6))
            output.close()
            with (root/'out.csv').open() as f: rows=list(csv.DictReader(f))
            self.assertEqual([int(r['frame']) for r in rows],list(range(7)))
            self.assertEqual([r['status'] for r in rows],['TRACKING','SKIPPED','SKIPPED','TRACKING','TRACKING','SKIPPED','SKIPPED'])
            cap=cv2.VideoCapture(str(args.save_video))
            self.assertEqual(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),7)
            cap.release()

    def test_wheel_zoom_drag_enter_maps_to_source(self):
        callback={}
        def wait(_):
            fn=callback['mouse']
            fn(cv2.EVENT_MOUSEWHEEL,160,90,120<<16,None)
            fn(cv2.EVENT_LBUTTONDOWN,100,60,0,None)
            fn(cv2.EVENT_LBUTTONUP,160,90,0,None)
            return 13
        with patch('cv2.namedWindow'),patch('cv2.destroyWindow'),patch('cv2.imshow'), \
             patch('cv2.getWindowProperty',return_value=1),patch('cv2.waitKey',side_effect=wait), \
             patch('cv2.setMouseCallback',side_effect=lambda _,fn:callback.update(mouse=fn)):
            box=select_box(scene(0)[0])
        np.testing.assert_allclose(box,[120,70,160,90],atol=1e-5)


if __name__=='__main__':
    unittest.main()
