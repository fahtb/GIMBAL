"""Actual webcam/CUDA lifecycle with simulated UI events; no accuracy annotation."""
import sys,time,json,csv,argparse
from unittest.mock import patch
import cv2
from camera import main,ROOT

parser=argparse.ArgumentParser();parser.add_argument('--output',default='test_results/hybrid/live')
options=parser.parse_args()
out=ROOT/options.output;out.mkdir(parents=True,exist_ok=True)
callback=None;shown=None;done=set()
def show(name,img):
    global shown
    if shown is None:shown=time.perf_counter()
    assert img.ndim==3 and img.shape[1]==1280
def set_callback(name,fn):
    global callback
    callback=fn
def wait(delay):
    time.sleep(max(.001,delay/1000))
    if shown is None:return -1
    age=time.perf_counter()-shown
    for threshold,event in ((1,'box'),(1.8,'k_off'),(2.1,'k_on'),(2.2,'i_off'),(2.4,'i_on'),(2.5,'h'),(3,'c'),(4,'s'),(4.5,'point'),(5,'o_off'),(5.4,'o_on'),(6,'w'),(7,'r'),(7.5,'box2'),(9,'q')):
        if age>=threshold and event not in done:
            done.add(event)
            if event in ('box','box2','point'):
                callback(cv2.EVENT_LBUTTONDOWN,240,140,0,None)
                callback(cv2.EVENT_LBUTTONUP,240 if event=='point' else 400,140 if event=='point' else 370,0,None)
                return -1
            return ord(event[0])
    return -1

sys.argv=['camera.py','--source','0','--engine','enhanced','--csv',str(out/'frames.csv'),
          '--profile',str(out/'profile.json'),'--telemetry',str(out/'target.json')]
with patch.object(cv2,'namedWindow'),patch.object(cv2,'getWindowProperty',return_value=1),patch.object(cv2,'destroyAllWindows'),patch.object(cv2,'imshow',show),patch.object(cv2,'waitKey',wait),patch.object(cv2,'setMouseCallback',set_callback):
    main()
assert 'q' in done
assert len(list(csv.DictReader((out/'frames.selections.csv').open())))>=3
assert json.loads((out/'target.json').read_text())['gimbal_command']['mode']=='HOLD'
(out/'lifecycle.json').write_text(json.dumps(dict(events=sorted(done),camera_cuda=True,ui='simulated callbacks; images rendered offscreen',accuracy='not measured'),indent=2))
assert all(key in done for key in ('k_off','k_on','i_off','i_on','o_off','o_on'))
print('CAMERA_CUDA_BOX_POINT_S_R_H_W_C_K_I_O_Q_HOLD_PASS')
