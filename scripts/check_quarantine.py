import tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch,Mock
import torch
import numpy as np
from live_tracker import LiveTracker
from recovery.quarantine import MemoryQuarantine
from sam_preprocess import prepare_rgb


class Tests(unittest.TestCase):
    def sam(self):
        s=LiveTracker.__new__(LiveTracker)
        s.model=SimpleNamespace();s.reset();s.memory=24
        return s

    def pending(self,s,index):
        tensor=torch.ones(2)*index
        out={k:tensor for k in ('maskmem_features','obj_ptr','kf_ious')}
        out['maskmem_pos_enc']=[tensor]
        s.pending_memory=(index,out,[1,2,3,4],.95,5.)
        return tensor

    def test_quarantine_requires_time_and_observations_and_discards(self):
        s=self.sam();self.pending(s,0);s.commit_memory()
        q=MemoryQuarantine();cfg=dict(pinned_frames=4,quarantine_frames=3,quarantine_s=.2)
        for i,t in enumerate((.01,.02,.03),1):
            self.pending(s,i);self.assertFalse(q.observe(s,t,True,cfg))
        q.observe(s,.1,False,cfg)
        self.assertEqual(len(q.entries),0);self.assertEqual(len(s.outputs['non_cond_frame_outputs']),0)
        for i,t in enumerate((.2,.3,.41),4):
            self.pending(s,i);q.observe(s,t,True,cfg)
        self.assertEqual(list(s.outputs['non_cond_frame_outputs']),[4])
        self.assertIn(0,s.outputs['cond_frame_outputs'])

    def test_snapshots_do_not_alias_graph_buffers_and_banks_bounded(self):
        s=self.sam();original=self.pending(s,0);s.commit_memory();original.fill_(999)
        self.assertEqual(s.outputs['cond_frame_outputs'][0]['obj_ptr'][0],0)
        for i in range(1,200):
            self.pending(s,i);s.commit_memory(condition=i<5)
        self.assertTrue(set(range(5))<=set(s.outputs['cond_frame_outputs']))
        self.assertLessEqual(len(s.outputs['cond_frame_outputs']),10)
        self.assertLessEqual(len(s.outputs['non_cond_frame_outputs']),24)

    def test_rebase_resets_counters_and_retains_original_anchor(self):
        s=self.sam();self.pending(s,0);s.commit_memory()
        original=s.outputs['cond_frame_outputs'][0]
        self.pending(s,8);s.commit_memory();s.index=44;s.cached_features='same-frame'
        def step(frame,stamp,**kw):
            self.assertEqual(s.index,1);self.assertEqual(s.cached_features,'same-frame')
            self.assertEqual(s.outputs['non_cond_frame_outputs'],{})
            self.assertIs(s.outputs['cond_frame_outputs'][0],original)
            self.assertTrue(kw['reuse_features']);s.index+=1;return {'mask':kw['mask_prompt']}
        s.step=step
        s.rebase_verified(np.zeros((4,4,3)),5.,np.ones((4,4)))
        self.assertEqual(s.index,2);self.assertIsNone(s.pending_memory)

    def test_telemetry_backoff_then_success(self):
        from camera import write_telemetry
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'target.json'
            original=Path.replace;calls=[]
            def replace(source,target):
                calls.append(1)
                if len(calls)<3:raise PermissionError('sharing violation')
                return original(source,target)
            with patch.object(Path,'replace',replace),patch('camera.time.sleep') as sleep:
                write_telemetry(path,{'valid':False})
                self.assertEqual(sleep.call_count,2)
            self.assertTrue(path.exists());self.assertEqual(len(list(Path(directory).glob('*.tmp'))),0)

    def test_rgb_and_jpeg_options(self):
        frame=np.zeros((16,16,3),np.uint8);frame[:,:,2]=255
        for resize in ('linear','bicubic'):
            rgb=prepare_rgb(frame,32,resize,0)
            self.assertEqual(int(rgb[4,4,0]),255);self.assertEqual(int(rgb[4,4,2]),0)
            self.assertEqual(prepare_rgb(frame,32,resize,90).shape,(32,32,3))


if __name__=='__main__':unittest.main()
