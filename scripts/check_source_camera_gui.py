"""Exercise main camera UI callbacks with real camera and simulated window/input."""
import sys,time,csv
from unittest.mock import patch
import cv2
from camera import main, ROOT
out=ROOT/'test_results/source_camera';out.mkdir(parents=True,exist_ok=True)
callback=None;shown=None;done=set()
original_show=cv2.imshow;original_wait=cv2.waitKey;original_callback=cv2.setMouseCallback

def show(name,img):
    global shown
    if shown is None: shown=time.perf_counter()
    assert img.ndim==3 and img.shape[1]==1280

def set_callback(name,fn):
    global callback
    callback=fn

def wait(delay):
    time.sleep(max(.001,delay/1000))
    if shown is None: return -1
    age=time.perf_counter()-shown
    for threshold,event in ((1,'select'),(3,'c'),(4,'s'),(4.5,'select2'),(6,'w'),(7,'r'),(7.5,'select3'),(9,'q')):
        if age>=threshold and event not in done:
            done.add(event)
            if event.startswith('select'):
                callback(cv2.EVENT_LBUTTONDOWN,240,140,0,None)
                callback(cv2.EVENT_LBUTTONUP,400,370,0,None)
                return -1
            return ord(event)
    return -1

sys.argv=['camera.py','--source','0','--camera-config','configs/camera_msmf_measured.json',
          '--csv','test_results/source_camera/gui_simulated.csv','--profile','test_results/source_camera/gui_simulated.profile.json',
          '--telemetry','test_results/source_camera/gui_target.json']
with patch.object(cv2,'namedWindow'),patch.object(cv2,'getWindowProperty',return_value=1),patch.object(cv2,'destroyAllWindows'),patch.object(cv2,'imshow',show),patch.object(cv2,'waitKey',wait),patch.object(cv2,'setMouseCallback',set_callback):
    main()
assert 'q' in done and len(list(csv.DictReader((out/'gui_simulated.selections.csv').open())))>=3
print('GUI_SELECT_S_R_W_C_Q_OK')
