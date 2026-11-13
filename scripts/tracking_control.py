"""Timestamped motion estimation and hardware-independent control simulation."""
import numpy as np

class BoxKalman:
    def __init__(self, acceleration: float=80., measurement: float=4., relative_measurement: float=0.):
        self.acceleration,self.measurement=acceleration,measurement
        self.relative_measurement=relative_measurement
        self.x=None;self.P=None;self.stamp=None
    @staticmethod
    def observation(box):
        b=np.asarray(box,float);return np.r_[(b[:2]+b[2:])/2,np.maximum(1,b[2:]-b[:2])]
    def reset(self,box,stamp):
        self.x=np.r_[self.observation(box),np.zeros(4)]
        self.P=np.diag([16.]*4+[400.]*4);self.stamp=float(stamp)
    def predict(self,stamp):
        if self.x is None:return None
        dt=float(stamp)-self.stamp
        if dt<0:raise ValueError('Kalman timestamps must be chronological')
        F=np.eye(8);F[:4,4:]=np.eye(4)*dt
        G=np.vstack([np.eye(4)*dt*dt/2,np.eye(4)*dt])
        self.x=F@self.x;self.x[2:4]=np.maximum(1,self.x[2:4])
        self.P=F@self.P@F.T+G@G.T*self.acceleration**2
        self.stamp=float(stamp)
        return self.box()
    def innovation(self,box):
        d=self.observation(box)-self.x[:4];S=self.P[:4,:4]+self.measurement_covariance()
        return float(d@np.linalg.solve(S,d))
    def measurement_covariance(self):
        # Box edges of a deforming large target are less precise than its center.
        wh=np.maximum(1,self.x[2:4])*self.relative_measurement
        return np.diag(self.measurement**2+np.r_[wh*.5,wh]**2)
    def update(self,box):
        R=self.measurement_covariance();S=self.P[:4,:4]+R
        K=np.linalg.solve(S,self.P[:4,:]).T
        self.x+=K@(self.observation(box)-self.x[:4])
        H=np.zeros((4,8));H[:,:4]=np.eye(4);I=np.eye(8)-K@H
        self.P=I@self.P@I.T+K@R@K.T
        self.x[2:4]=np.maximum(1,self.x[2:4])
    def box(self):
        return np.r_[self.x[:2]-self.x[2:4]/2,self.x[:2]+self.x[2:4]/2].tolist()

class SimulatedGimbal:
    """Normalized rate command only. No serial port or motor connection."""
    def __init__(self,kp=1.,ki=.05,kd=.02,dead_zone=.02,max_rate=.6,max_accel=2.,tau=.10):
        self.kp,self.ki,self.kd=kp,ki,kd
        self.dead_zone,self.max_rate,self.max_accel,self.tau=dead_zone,max_rate,max_accel,tau
        self.reset()
    def reset(self):
        self.integral=np.zeros(2);self.filtered=np.zeros(2);self.rate=np.zeros(2);self.stamp=None
    def step(self,error,stamp,confirmed):
        if not confirmed or error is None:
            self.reset();return dict(mode='HOLD',rate_x=0.,rate_y=0.,simulated=True)
        dt=0. if self.stamp is None else max(0.,min(.25,stamp-self.stamp));self.stamp=stamp
        e=np.asarray(error,float);e=np.where(np.abs(e)<self.dead_zone,0.,e)
        old=self.filtered.copy();alpha=dt/(self.tau+dt) if dt else 0.
        self.filtered+=alpha*(e-self.filtered)
        candidate=np.clip(self.integral+self.filtered*dt,-1,1)
        raw=self.kp*self.filtered+self.ki*candidate+self.kd*(self.filtered-old)/max(dt,.001)
        self.integral=np.where(np.abs(raw)<self.max_rate,candidate,self.integral)
        desired=np.clip(raw,-self.max_rate,self.max_rate)
        self.rate+=np.clip(desired-self.rate,-self.max_accel*dt,self.max_accel*dt)
        return dict(mode='TRACK',rate_x=float(self.rate[0]),rate_y=float(self.rate[1]),simulated=True)

def safe_output(data: dict, max_age_ms: float=250.) -> dict:
    """Never expose a stale simulated command as a confirmed control target."""
    result=dict(data)
    if 'gimbal_command' not in result:return result
    age=max(float(result.get('source_age_ms') or 0),float(result.get('age_ms') or 0))
    if not result.get('valid') or not result.get('is_confirmed') or age>max_age_ms:
        result.update(is_confirmed=False,recommended_control_mode='HOLD',normalized_error_x=None,normalized_error_y=None,
                      gimbal_command=dict(mode='HOLD',rate_x=0.,rate_y=0.,simulated=True))
    return result
