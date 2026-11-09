import numpy as np
import torch
import torch.nn.functional as F
import comfy.model_management as mm
from comfy.utils import ProgressBar
from .postpro_core import DEFAULTS,RANGES,validate,process

TIPS={
'exposure':'Exposure in stops, applied in linear RGB: +1 doubles linear light.',
'brightness':'Adds a display RGB offset. Positive brightens; negative darkens.',
'gamma':'Midtone adjustment: 1 neutral, above 1 brightens, below 1 darkens. Uses x^(1/gamma).',
'contrast':'Contrast around pivot: 1 neutral, 0 flattens, above 1 increases contrast.',
'pivot':'Display RGB contrast pivot. Default 0.5; used only by contrast.',
'shadows':'Signed shadow lift: positive brightens dark areas, negative darkens them.',
'highlights':'Signed highlight lift: positive brightens light areas, negative darkens them. Cannot recover already clipped details.',
'black_point':'Input levels black point. Values at/below this become black before later adjustments.',
'white_point':'Input levels white point; must exceed black_point. Values at/above this become white before later adjustments.',
'temperature':'Creative warm/cool balance: positive warmer, negative cooler. An artistic RGB adjustment, not calibrated Kelvin white balance.',
'tint':'Creative green/magenta balance: positive magenta, negative green.',
'red_gain':'Red channel multiplier: 1 neutral.',
'green_gain':'Green channel multiplier: 1 neutral.',
'blue_gain':'Blue channel multiplier: 1 neutral.',
'saturation':'Luma-relative color intensity: 0 grayscale, 1 neutral, above 1 stronger colors.',
'vibrance':'Selective saturation: positive boosts less colorful pixels more. No semantic skin protection.',
'hue_shift':'HSV hue rotation in degrees; 0 neutral.',
'sharpness':'Luma unsharp-mask amount: 0 off. Strong values may produce halos.',
'sharpen_radius':'Approximate Gaussian blur sigma in output pixels for sharpening. Larger emphasizes broader edges.'}

class MagikColorPostpro:
    CATEGORY='Magik/Image'
    FUNCTION='apply'
    RETURN_TYPES=('IMAGE',)
    RETURN_NAMES=('image',)
    DESCRIPTION='Float sRGB postprocessing. Neutral by default. Processes batches sequentially on CPU, preserves alpha, supports a soft mask and global strength. Hover settings for details.'

    @classmethod
    def INPUT_TYPES(cls):
        required={'image':('IMAGE',{'tooltip':'RGB/RGBA image or batch. Output preserves size, alpha, dtype and device.'}),
                  'strength':('FLOAT',{'default':1.,'min':0.,'max':1.,'step':.01,'tooltip':'Blend all adjustments with the original. 0 bypasses; 1 full effect.'})}
        # Order related settings together while retaining keyword execution.
        for k in ['exposure','brightness','gamma','contrast','pivot','shadows','highlights','black_point','white_point','temperature','tint','red_gain','green_gain','blue_gain','saturation','vibrance','hue_shift','sharpness','sharpen_radius']:
            lo,hi=RANGES[k]
            required[k]=('FLOAT',{'default':DEFAULTS[k],'min':lo,'max':hi,'step':1. if k=='hue_shift' else .01,'tooltip':TIPS[k]})
        return {'required':required,'optional':{'mask':('MASK',{'tooltip':'White applies adjustments; black keeps original; gray blends. One shared mask or one per image. Resized bilinearly.'})}}

    def apply(self,image,strength=1.,mask=None,**kwargs):
        if image.ndim!=4 or image.shape[-1] not in (3,4) or min(image.shape[:3])<1:
            raise ValueError('image must be a nonempty BHWC RGB/RGBA IMAGE.')
        if not torch.isfinite(image).all():
            raise ValueError('image contains NaN or infinity.')
        settings={**DEFAULTS,**kwargs}; validate(settings)
        if not np.isfinite(strength) or not 0<=strength<=1:
            raise ValueError('strength must be between 0 and 1.')
        if mask is not None:
            if mask.ndim==2: mask=mask.unsqueeze(0)
            if mask.ndim!=3 or mask.shape[0] not in (1,len(image)) or min(mask.shape[1:])<1:
                raise ValueError('mask must contain one shared mask or one per image.')
            if not torch.isfinite(mask).all():
                raise ValueError('mask contains NaN or infinity.')
        neutral=all(settings[k]==v for k,v in DEFAULTS.items() if k not in ('pivot','sharpen_radius'))
        if strength==0 or neutral:
            return (image,)
        out=torch.empty_like(image); progress=ProgressBar(len(image))
        for i in range(len(image)):
            mm.throw_exception_if_processing_interrupted()
            src=image[i,...,:3].detach().float().cpu().numpy()
            corrected=process(src,check=mm.throw_exception_if_processing_interrupted,**settings)
            blend=strength
            if mask is not None:
                m=mask[0 if len(mask)==1 else i].detach().float().cpu()
                m=F.interpolate(m[None,None],size=src.shape[:2],mode='bilinear',align_corners=False)[0,0].clamp(0,1).numpy()
                blend=strength*m[...,None]
            mixed=src+blend*(corrected-src)
            if not np.isfinite(mixed).all():
                raise ValueError(f'Nonfinite output in image {i+1}.')
            out[i,...,:3]=torch.from_numpy(mixed.astype(np.float32)).to(device=image.device,dtype=image.dtype)
            if image.shape[-1]==4: out[i,...,3]=image[i,...,3]
            progress.update(1)
        return (out,)
