"""Full async video path on a generated fixture, never a user's recording."""
import json,sys
import cv2
from live_tracker import ROOT
from check_object_continuity_gpu import scene


def main():
    out=ROOT/'test_results/identity_confidence'
    out.mkdir(parents=True,exist_ok=True)
    clip=out/'generated_D.avi'
    writer=cv2.VideoWriter(str(clip),cv2.VideoWriter_fourcc(*'MJPG'),30,(640,480))
    assert writer.isOpened()
    try:
        for i in range(120):
            frame,_=scene('D',min(i,21))
            writer.write(cv2.resize(frame,(640,480)))
    finally:
        writer.release()
    sys.argv=['camera.py','--source',str(clip),'--mode','realtime','--sam-only',
              '--target-mode','object','--recovery','friend','--no-display',
              '--init-bbox','156,136,36,208','--max-frames','120',
              '--csv',str(out/'realtime.csv'),'--profile',str(out/'realtime.profile.json'),
              '--telemetry',str(out/'target.json'),'--save-video',str(out/'overlay.mp4')]
    from camera import main as run
    run()
    profile=json.loads((out/'realtime.profile.json').read_text())
    print(json.dumps(profile),flush=True)


if __name__=='__main__':
    main()
