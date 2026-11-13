"""Controlled small-target relocation fixture; not a natural-video benchmark."""
import csv
import json
import sys
from pathlib import Path
import cv2
import numpy as np
from live_tracker import LiveTracker, ROOT
from sam_memory import SAMMemoryTracker, overlap
from sam_crop import NativeCropTracker
from source_frame import SourceFrame


def main():
    capture = cv2.VideoCapture(str(ROOT/'videos/test1.mp4'))
    ok,original = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError('Cannot read small-target fixture source')
    h,w = original.shape[:2]
    scale = w/960
    prompt = [288.5185185,293.1111111,295.8930041,305.7530864]
    moved,expected = None,None
    sam = LiveTracker(ROOT/'sam2/checkpoints/sam2.1_hiera_tiny.pt',image_size=512,cuda_graph=True)
    sam.warmup()
    tracker = NativeCropTracker(SAMMemoryTracker(sam))
    rows = []
    for index in range(160):
        raw = original if index < 8 else np.zeros_like(original) if index < 12 else moved
        frame = SourceFrame(cv2.resize(raw,(960,round(h/scale))),raw)
        result = tracker.step(frame,index/15,prompt if index == 0 else None)
        if index == 0:
            # Move the complete SAM-selected target, not a possibly tighter user box.
            a,b,c,d = np.rint(result['native_box']).astype(int)
            c,d = c+1,d+1
            moved = original.copy()
            patch = original[b:d,a:c].copy()
            moved[b:d,a:c] = np.median(original[max(0,b-10):b,a:c],axis=(0,1)).astype(np.uint8)
            x,y = round(w*.75),round(h*.55)
            moved[y:y+d-b,x:x+c-a] = patch
            expected = [x,y,x+c-a-1,y+d-b-1]
            if '--probe' in sys.argv:
                from sam_tiles import tile_windows
                core=tracker.tracker
                center=(np.asarray(expected[:2])+np.asarray(expected[2:]))/2
                win=tile_windows(original.shape,512,.25,center)[0]
                ta,tb,tc,td=win
                core.sam.encode_features(moved[tb:td,ta:tc])
                core.search(moved[tb:td,ta:tc],proposals_only=True)
                target=center-np.array([ta,tb])
                values=sorted(core.last_search_values,key=lambda v:np.linalg.norm(np.asarray(v['point'])-target))
                print('EXPECTED',target.tolist(),'NEAREST',values[:2])
                ta,tb=np.rint(center-128).astype(int)
                crop=moved[tb:tb+256,ta:ta+256]
                core.sam.encode_features(crop)
                masks,ious,logits,stability=core.sam.decode_points([[128,128]],(256,256))
                value=core.assess(crop,masks[0],ious[0],logits[0],stability[0])
                print('EXACT_REFINED',None if value is None else {k:value[k] for k in ('score','similarity','color','ratio','iou','logit','stability')})
                return
        rows.append(dict(frame=index,status=result['status'],reacquired=result.get('reacquired',0),
                         tile_count=result.get('tile_count',0),tile_hits=result.get('tile_hits',0),
                         tile_similarity=result.get('tile_similarity'),
                         tile_debug=json.dumps(result.get('tile_candidate_debug')),
                         bbox_iou=overlap(result.get('native_box'),expected) if index>=12 else None,
                         transition_reason=result.get('transition_reason')))
    path = ROOT/'runtime/sam_tile_fixture.csv'
    with path.open('w',newline='',encoding='utf-8') as handle:
        writer=csv.DictWriter(handle,fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    accepted=[r for r in rows if r['frame']>=12 and r['status']=='SMALL' and (r['bbox_iou'] or 0)>.3]
    print(json.dumps(dict(accepted_frames=len(accepted),first_accepted=accepted[0]['frame'] if accepted else None,
                          reacquired=rows[-1]['reacquired'],last_status=rows[-1]['status'])))
    if not accepted:
        from sam_tiles import tile_windows
        center=(np.asarray(expected[:2])+np.asarray(expected[2:]))/2
        win=tile_windows(original.shape,512,.25,center)[0]
        a,b,c,d=win
        sampler=tracker.tracker.sam
        sampler.encode_features(moved[b:d,a:c])
        masks,ious,logits,stability=sampler.decode_points([center-np.array([a,b])],(d-b,c-a))
        value=tracker.tracker.assess(moved[b:d,a:c],masks[0],ious[0],logits[0],stability[0])
        print('KNOWN_POINT', None if value is None else {k:value[k] for k in ('score','similarity','color','ratio','iou','logit','stability')})
        raise AssertionError('Tiled relocation has not recovered the known pasted target')


if __name__ == '__main__':
    main()
