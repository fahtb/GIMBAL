"""Human-reviewed single-target DAVIS-style annotation; SAM proposes, S approves."""
import argparse,hashlib,json
from pathlib import Path
import cv2
import numpy as np


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--split',choices=('train','val'),required=True)
    p.add_argument('--kind',choices=('object','person'),required=True)
    p.add_argument('--start',type=int,default=0)
    p.add_argument('--checkpoint',type=Path,default=Path(__file__).resolve().parents[1]/'sam2/checkpoints/sam2.1_hiera_tiny.pt')
    args=p.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        p.error('Use a new output directory; existing reviewed annotations will not be overwritten')
    cap=cv2.VideoCapture(str(args.source))
    if not cap.isOpened():p.error('Cannot open source')
    for _ in range(args.start):
        if not cap.grab():p.error('Start is beyond video')
    from live_tracker import LiveTracker
    sam=LiveTracker(args.checkpoint,image_size=512)
    images=args.output/'JPEGImages';masks=args.output/'Annotations'
    images.mkdir(parents=True,exist_ok=True);masks.mkdir(parents=True,exist_ok=True)
    source_id=hashlib.sha256(str(args.source.resolve()).lower().encode()).hexdigest()
    metadata=dict(source_id=source_id,split=args.split,kind=args.kind,frames=[])
    index=args.start;name='Annotate | B SAM box | left paint / right erase | E absent | S approve | N next | Q quit'
    brush=dict(mask=None,empty=False)
    def paint(event,x,y,flags,param):
        if brush['mask'] is not None and flags&(cv2.EVENT_FLAG_LBUTTON|cv2.EVENT_FLAG_RBUTTON):
            cv2.circle(brush['mask'],(x,y),8,0 if flags&cv2.EVENT_FLAG_RBUTTON else 1,-1)
            brush['empty']=False
    cv2.namedWindow(name);cv2.setMouseCallback(name,paint)
    try:
        while True:
            ok,frame=cap.read()
            if not ok:break
            brush.update(mask=np.zeros(frame.shape[:2],np.uint8),empty=False)
            while True:
                view=frame.copy();mask=brush['mask']>0
                view[mask]=(view[mask]*.55+np.array([0,220,0])*.45).astype(np.uint8)
                cv2.putText(view,f'frame {index} | APPROVE only the selected target, exclude distractors',(8,23),0,.5,(0,255,255),1)
                cv2.imshow(name,view);key=cv2.waitKey(20)&255
                if key in (ord('q'),27):return
                if key==ord('e'):brush.update(mask=np.zeros(frame.shape[:2],np.uint8),empty=True)
                if key==ord('b'):
                    x,y,w,h=cv2.selectROI(name,frame,False,False)
                    if w and h:
                        result=sam.step(frame,0.,[x,y,x+w,y+h])
                        if result.get('mask') is not None:brush.update(mask=result['mask'].astype(np.uint8),empty=False)
                if key==ord('s'):
                    if not brush['mask'].any() and not brush['empty']:
                        print('Press E to explicitly confirm target absence before saving an empty mask.');continue
                    stem=f'{index:06d}'
                    if not cv2.imwrite(str(images/(stem+'.jpg')),frame) or not cv2.imwrite(str(masks/(stem+'.png')),brush['mask']):
                        raise RuntimeError('Could not save annotation')
                    metadata['frames']=[r for r in metadata['frames'] if r['frame']!=index]
                    metadata['frames'].append(dict(frame=index,reviewed=True,absent=brush['empty'],image=stem+'.jpg',mask=stem+'.png'))
                    (args.output/'review.json').write_text(json.dumps(metadata,indent=2))
                    break
                if key==ord('n'):break
            index+=1
    finally:
        cap.release();cv2.destroyAllWindows()


if __name__=='__main__':main()
