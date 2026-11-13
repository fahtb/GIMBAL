"""Capture-only FPS probe; no SAM, no display, no image recording."""
import argparse
import json
import time
import cv2


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',type=int,default=0)
    p.add_argument('--backend',choices=('dshow','msmf'),default='dshow')
    p.add_argument('--seconds',type=float,default=5.)
    p.add_argument('--manual-exposure',type=float)
    p.add_argument('--auto-value',type=float,help='driver-specific manual-mode value; required with manual-exposure')
    args=p.parse_args()
    if args.seconds<=0 or (args.manual_exposure is not None and args.auto_value is None):
        p.error('seconds must be positive; manual exposure requires a driver-specific --auto-value')
    cap=cv2.VideoCapture(args.source,cv2.CAP_DSHOW if args.backend=='dshow' else cv2.CAP_MSMF)
    try:
        if not cap.isOpened():
            raise RuntimeError('Cannot open camera; close other camera applications first')
        accepted={}
        for name,prop,value in [('mjpg',cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc(*'MJPG')),
                               ('width',cv2.CAP_PROP_FRAME_WIDTH,640),('height',cv2.CAP_PROP_FRAME_HEIGHT,480),
                               ('fps',cv2.CAP_PROP_FPS,30),('buffer',cv2.CAP_PROP_BUFFERSIZE,1)]:
            accepted[name]=dict(accepted=cap.set(prop,value),readback=cap.get(prop))
        if args.manual_exposure is not None:
            accepted['auto_exposure']=cap.set(cv2.CAP_PROP_AUTO_EXPOSURE,args.auto_value)
            accepted['exposure']=cap.set(cv2.CAP_PROP_EXPOSURE,args.manual_exposure)
        for _ in range(5):
            cap.read()
        count=failed=0;start=time.perf_counter()
        while time.perf_counter()-start<args.seconds:
            ok,_=cap.read();count+=int(ok);failed+=int(not ok)
        elapsed=time.perf_counter()-start
        print(json.dumps(dict(backend=args.backend,frames=count,failed=failed,elapsed_s=elapsed,
                              measured_fps=count/elapsed,settings=accepted),indent=2))
    finally:
        cap.release()


if __name__=='__main__':
    main()
