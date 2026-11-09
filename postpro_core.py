"""Float sRGB creative grading. No quantization, models or external dependencies."""
import numpy as np

DEFAULTS = dict(exposure=0.,brightness=0.,gamma=1.,contrast=1.,pivot=.5,
                shadows=0.,highlights=0.,black_point=0.,white_point=1.,
                temperature=0.,tint=0.,red_gain=1.,green_gain=1.,blue_gain=1.,
                saturation=1.,vibrance=0.,hue_shift=0.,sharpness=0.,sharpen_radius=1.)
RANGES = dict(exposure=(-5,5),brightness=(-1,1),gamma=(.1,5),contrast=(0,3),pivot=(0,1),
              shadows=(-1,1),highlights=(-1,1),black_point=(0,.99),white_point=(.01,1),
              temperature=(-1,1),tint=(-1,1),red_gain=(0,3),green_gain=(0,3),blue_gain=(0,3),
              saturation=(0,3),vibrance=(-1,1),hue_shift=(-180,180),sharpness=(0,3),sharpen_radius=(.5,20))


def validate(settings):
    for k,v in settings.items():
        if k not in RANGES or not np.isfinite(v) or not RANGES[k][0]<=v<=RANGES[k][1]:
            raise ValueError('Invalid postprocessing setting: '+k)
    if settings['white_point']-settings['black_point']<.001:
        raise ValueError('white_point must exceed black_point by at least 0.001.')


def blur(x,radius,check):
    # Three separable box passes, edge extension; approximate Gaussian sigma.
    half=max(1,int(round((np.sqrt(1+4*radius*radius)-1)/2)))
    out=x.astype(np.float64)
    for _ in range(3):
        for axis in (0,1):
            check()
            pads=[(0,0)]*3; pads[axis]=(half,half)
            a=np.pad(out,pads,mode='edge')
            pads[axis]=(1,0)
            a=np.pad(np.cumsum(a,axis=axis),pads)
            lo,hi=[slice(None)]*3,[slice(None)]*3
            lo[axis]=slice(None,-(2*half+1)); hi[axis]=slice(2*half+1,None)
            out=(a[tuple(hi)]-a[tuple(lo)])/(2*half+1)
    return out


def hue_rotate(x,degrees):
    # Rotate HSV hue while preserving HSV saturation and value.
    v=x.max(-1); low=x.min(-1); d=v-low
    safe=np.maximum(d,1e-12)
    idx=np.argmax(x,axis=-1)
    h=np.select([idx==0,idx==1],[(x[...,1]-x[...,2])/safe,2+(x[...,2]-x[...,0])/safe],default=4+(x[...,0]-x[...,1])/safe)
    h=np.where(d>1e-12,(h/6+degrees/360)%1,0)
    sector=h*6; i=np.floor(sector).astype(int)%6; f=sector-np.floor(sector)
    p=low; q=v-f*d; t=low+f*d
    choices=[np.stack(c,-1) for c in [(v,t,p),(q,v,p),(p,v,t),(p,q,v),(t,p,v),(v,p,q)]]
    return np.take_along_axis(np.stack(choices,axis=-2),i[...,None,None],axis=-2)[...,0,:]


def process(image,check=lambda:None,**kwargs):
    s={**DEFAULTS,**kwargs}; validate(s)
    a=np.asarray(image)
    if a.ndim!=3 or a.shape[-1]!=3 or min(a.shape[:2])<1 or not np.isfinite(a).all():
        raise ValueError('Expected a finite, nonempty HWC RGB image.')
    # Preserve exact pixels for neutral settings, including the user's levels pivot/radius.
    neutral=all(s[k]==v for k,v in DEFAULTS.items() if k not in ('pivot','sharpen_radius'))
    if neutral:
        return a.copy()
    x=a.astype(np.float64).clip(0,1)
    check()
    if s['exposure']:
        linear=np.where(x<=.04045,x/12.92,((x+.055)/1.055)**2.4)
        linear*=2**s['exposure']
        x=np.where(linear<=.0031308,12.92*linear,1.055*np.maximum(linear,0)**(1/2.4)-.055)
    x=(x-s['black_point'])/(s['white_point']-s['black_point'])
    x+=s['brightness']
    x=(x-s['pivot'])*s['contrast']+s['pivot']
    y=np.clip(x@np.array([.2126,.7152,.0722]),0,1)[...,None]
    # Signed tonal lifts: positive lifts, negative darkens. Midtones overlap gently.
    x+=s['shadows']*(1-y)**2*.5+s['highlights']*y**2*.5
    x=np.maximum(x,0)**(1/s['gamma'])
    x*=np.array([s['red_gain'],s['green_gain'],s['blue_gain']])*np.exp2(
        [s['temperature']*.5+s['tint']*.25,-s['tint']*.5,-s['temperature']*.5+s['tint']*.25])
    x=x.clip(0,1)
    check()
    if s['hue_shift']:
        x=hue_rotate(x,s['hue_shift'])
    y=(x@np.array([.2126,.7152,.0722]))[...,None]
    spread=(x.max(-1)-x.min(-1))[...,None]
    factor=s['saturation']*(1+s['vibrance']*(1-spread))
    x=y+(x-y)*factor
    if s['sharpness']:
        check()
        # Luma-only unsharp mask, avoiding independently sharpened color edges.
        luma=(x@np.array([.2126,.7152,.0722]))[...,None]
        detail=luma-blur(luma,s['sharpen_radius'],check)
        x+=s['sharpness']*detail
    return x.clip(0,1).astype(np.float32)
