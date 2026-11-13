"""Masked Lucas-Kanade with forward/backward validation and RANSAC."""
import cv2
import numpy as np

def partial_support(reference,reference_mask,current,mask,predicted_box,min_pixels=32):
    """Verify surviving grayscale texture in aligned coordinates, never by color."""
    y,x=np.nonzero(reference_mask)
    if not len(x) or predicted_box is None:return None
    old=np.array([x.min(),y.min(),x.max()+1,y.max()+1],float)
    predicted=np.asarray(predicted_box,float)
    scale=(predicted[2:]-predicted[:2])/(old[2:]-old[:2])
    transform=np.array([[scale[0],0,predicted[0]-old[0]*scale[0]],
                        [0,scale[1],predicted[1]-old[1]*scale[1]]],np.float32)
    size=current.shape[1],current.shape[0]
    support=cv2.warpAffine(reference_mask.astype(np.uint8),transform,size,flags=cv2.INTER_NEAREST)>0
    visible=mask&support;count=int(visible.sum())
    if count<min_pixels or mask.sum()==0:return None
    warped=cv2.warpAffine(cv2.cvtColor(reference,cv2.COLOR_BGR2GRAY),transform,size)
    gray=cv2.cvtColor(current,cv2.COLOR_BGR2GRAY)
    a=warped[visible].astype(float);b=gray[visible].astype(float)
    texture=min(float(a.std()),float(b.std()))
    a-=a.mean();b-=b.mean()
    similarity=float(a@b/max(1e-6,np.linalg.norm(a)*np.linalg.norm(b)))
    return dict(similarity=similarity,texture=texture,containment=count/int(mask.sum()),
                visible_ratio=count/max(1,int(support.sum())),pixels=count)

def estimate_motion(previous,current,mask,foreground=True):
    h,w=previous.shape[:2];scale=min(1.,480/w)
    size=(round(w*scale),round(h*scale))
    a=cv2.resize(cv2.cvtColor(previous,cv2.COLOR_BGR2GRAY),size)
    b=cv2.resize(cv2.cvtColor(current,cv2.COLOR_BGR2GRAY),size)
    eligible=cv2.resize(mask.astype('uint8')*255,size,interpolation=cv2.INTER_NEAREST)
    if not foreground:eligible=255-cv2.dilate(eligible,np.ones((15,15),np.uint8))
    points=cv2.goodFeaturesToTrack(a,100,.02,5,mask=eligible)
    if points is None or len(points)<6:return None
    forward,ok,_=cv2.calcOpticalFlowPyrLK(a,b,points,None)
    if forward is None:return None
    back,ok2,_=cv2.calcOpticalFlowPyrLK(b,a,forward,None)
    if back is None:return None
    keep=(ok.ravel()>0)&(ok2.ravel()>0)&(np.linalg.norm(back-points,axis=2).ravel()<1.)
    if keep.sum()<6:return None
    transform,inliers=cv2.estimateAffinePartial2D(points[keep],forward[keep],method=cv2.RANSAC,ransacReprojThreshold=2.)
    if transform is None or inliers.mean()<.6:return None
    transform[:,2]/=scale
    return dict(matrix=transform,inlier_ratio=float(inliers.mean()),points=int(keep.sum()))
