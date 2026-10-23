"""Read un-clipped half-size Bayer measurements and interrogate LibRaw's matrix.

The decoder matrix is a finite difference of four known digital input vectors,
not a fit to a scene, film or scan. No files are modified. Linear XYZ output
from LibRaw can itself clip at 16 bits despite unsaturated CFA channels, so it
must not be used as the sole test of available RAW highlight information.
"""
from pathlib import Path
import sys

import numpy as np
import rawpy

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from filmlab.rawio import sensor_saturation


def orient(a, flip):
    if flip == 3: return a[::-1, ::-1]
    if flip == 5: return np.rot90(a, 1)
    if flip == 6: return np.rot90(a, -1)
    return a


def bayer_measurements(path):
    with rawpy.imread(str(path)) as raw:
        sat = sensor_saturation(raw, per_channel=True)
        data = raw.raw_image_visible.astype(float)
        colors = raw.raw_colors_visible.copy()
        black = np.array(raw.black_level_per_channel, float)
        white = float(raw.white_level)
        values = (data-black[colors])/(white-black[colors])
        h,w = (np.array(values.shape)//2)*2
        values=values[:h,:w].reshape(h//2,2,w//2,2)
        colors=colors[:h,:w].reshape(h//2,2,w//2,2)
        desc=raw.color_desc.decode(); channels=[]
        for name in 'RGB':
            use=np.isin(colors,[i for i,ch in enumerate(desc) if ch==name])
            channels.append(np.sum(values*use,axis=(1,3))/use.sum((1,3)))
        cam=orient(np.stack(channels,-1),raw.sizes.flip)
        if sat is None: sat=np.zeros(cam.shape,bool)
        return cam,sat,{'black':black.tolist(),'white':white,'daylight_wb':list(raw.daylight_whitebalance)}


def decoder_matrix(path):
    results=[]; inputs=[]
    for channel in (None,0,1,2):
        with rawpy.imread(str(path)) as raw:
            data=raw.raw_image_visible;colors=raw.raw_colors_visible
            black=np.array(raw.black_level_per_channel,float);white=raw.white_level
            desc=raw.color_desc.decode()
            actual=[]
            for name in 'RGB':
                value=.25 if channel=='RGB'.index(name) else .125
                ids=[i for i,ch in enumerate(desc) if ch==name]
                signal=[]
                for i in ids:
                    level=round(black[i]+value*(white-black[i]))
                    data[colors==i]=level
                    signal.append((level-black[i])/(white-black[i]))
                actual.append(np.mean(signal))
            wb=list(raw.daylight_whitebalance);wb=wb[:3]+[wb[1] if wb[3]<=0 else wb[3]]
            xyz=raw.postprocess(half_size=True,no_auto_bright=True,gamma=(1,1),output_bps=16,
                                output_color=rawpy.ColorSpace.XYZ,user_wb=wb,use_camera_wb=False,use_auto_wb=False)
            interior=xyz[20:-20,20:-20]
            if np.any(interior==0) or np.any(interior==65535):raise ValueError('Decoder probe clipped')
            results.append(np.median(interior.reshape(-1,3),0)/65535)
            inputs.append(actual)
    matrix=np.stack([(results[c+1]-results[0])/(inputs[c+1][c]-inputs[0][c]) for c in range(3)])
    offset=np.array(results[0])-np.array(inputs[0])@matrix
    if np.max(np.abs(offset))>3/65535:raise ValueError('Decoder is not linear within probe quantization')
    return matrix,{'probe_inputs':inputs,'probe_outputs':np.array(results).tolist(),
                   'intercept_quantization_error':offset.tolist(),'method':'LibRaw digital finite differences, no scene fit'}
