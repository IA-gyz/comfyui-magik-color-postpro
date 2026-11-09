# Magik Color Postpro — v1.0.0

Standalone ComfyUI node for manual image postprocessing. All adjustments default to neutral. No reference image, models, extra dependencies, or custom JavaScript needed.

## Install

Extract `Magik_Color_Postpro` into `ComfyUI/custom_nodes/`, then restart ComfyUI. Add **Magik → Image → Magik Color Postpro**.

Connect `image` to your source and the output to Preview Image / Save Image. You can place it after Magik Color Correct for final creative grading.

## Controls

| Control | Neutral | Effect |
| --- | --- | --- |
| strength | 1 | Overall blend with original, 0–1. Set 0 to bypass. |
| exposure | 0 | Exposure in stops, in linear RGB. +1 doubles linear light. |
| brightness | 0 | Additive display RGB offset. |
| gamma | 1 | Midtones via x^(1/gamma). Above 1 brightens; below 1 darkens. |
| contrast | 1 | Contrast around pivot. 0 flattens; above 1 increases contrast. |
| pivot | 0.5 | Anchor for contrast. Does nothing when contrast is 1. |
| shadows | 0 | Signed broad lift/darken weighted toward dark pixels. |
| highlights | 0 | Signed broad lift/darken weighted toward bright pixels. |
| black_point | 0 | Input levels black point. |
| white_point | 1 | Input levels white point; must exceed black_point by at least 0.001. |
| temperature | 0 | Creative warm/cool balance. Positive warmer, negative cooler. |
| tint | 0 | Positive magenta, negative green. |
| red_gain / green_gain / blue_gain | 1 | Independent channel multipliers. |
| saturation | 1 | 0 grayscale; above 1 stronger colors. |
| vibrance | 0 | Selectively boosts/mutes low-chroma pixels; no skin detection. |
| hue_shift | 0 | HSV hue rotation in degrees, −180 to +180. |
| sharpness | 0 | Luma-only unsharp mask amount. |
| sharpen_radius | 1 | Approximate blur sigma in output pixels; relevant only when sharpness >0. |

Each setting includes a tooltip. Values are saved with the workflow. This uses native ComfyUI widgets; preview updates when you execute the workflow, not continuously while dragging sliders.

## Optional mask

White applies the adjustments, black preserves the original, and gray blends smoothly. The mask is resized bilinearly. Use one shared mask or one mask per image in the batch. Invert alpha-derived Load Image masks if needed to get the intended white-selected convention.

## Adjustment order

1. Linear-light exposure.
2. Levels, brightness, contrast.
3. Shadow/highlight lifts using pre-lift display luma weights.
4. Gamma.
5. Temperature, tint, and RGB gains.
6. Clamp RGB, then HSV hue rotation.
7. Luma-relative saturation and selective vibrance.
8. Luma-only sharpening.
9. Clamp processed RGB to [0,1], then blend with original using strength and mask.

Controls interact in this fixed order. Shadow/highlight controls are creative tonal lifts, not HDR recovery. Temperature/tint are artistic RGB balancing, not calibrated Kelvin white balance. Strong adjustments can clip colors; strong sharpening can create halos. Saturation/sharpening luma weights operate on display RGB, not physically linear luminance.

## Batches and precision

Processes frames sequentially on CPU using NumPy with interruption checks and ComfyUI batch progress. Supports RGB and RGBA, preserving image dimensions, batch count, alpha, output dtype and output device. Alpha does not implicitly act as a mask. All-neutral settings and strength 0 return the original tensor exactly. The neutral test excludes contrast pivot and sharpening radius because those do nothing with neutral contrast/sharpness.

Input is expected to be display-referred sRGB in [0,1], not HDR. Computation uses floating-point arrays without 8-bit quantization. Processing clamps its working RGB; exact bypass and black areas of the application mask retain the original values. NaN/infinity inputs and invalid settings raise clear errors. Large images require several full-frame CPU arrays; batch processing is sequential to limit peak memory.

NumPy and PyTorch are already included in ComfyUI; no requirements installation is needed.

## Validation

Seven numerical tests pass: exact neutral passthrough; brightness/gamma/contrast; linear exposure; hue rotation/grayscale; sharpening and tiny-image blur; finite in-range outputs across controls; invalid input rejection.

Run `python Magik_Color_Postpro/test_postpro_core.py` to repeat numerical tests without ComfyUI. Python compilation passed. PyTorch and a running ComfyUI are unavailable in the build environment, so native widget rendering, mask integration, batch/device/alpha behavior need verification on the target installation.
