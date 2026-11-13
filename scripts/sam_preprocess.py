"""Controlled preprocessing experiments; BGR->RGB is always explicit."""
import cv2


def prepare_rgb(frame,size,resize='linear',jpeg_quality=0):
    if jpeg_quality:
        ok,encoded=cv2.imencode('.jpg',frame,[cv2.IMWRITE_JPEG_QUALITY,int(jpeg_quality)])
        if not ok:
            raise RuntimeError('JPEG preprocessing failed')
        frame=cv2.imdecode(encoded,cv2.IMREAD_COLOR)
    modes={'linear':cv2.INTER_LINEAR,'bicubic':cv2.INTER_CUBIC}
    return cv2.cvtColor(cv2.resize(frame,(size,size),interpolation=modes[resize]),cv2.COLOR_BGR2RGB)
