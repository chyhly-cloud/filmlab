"""Register short RAWs and test exposure-normalized unsaturated measurements.

No HDR synthesis, highlight prior or model changes. Geometry only is estimated;
the photometric scale is fixed by shutter/aperture metadata, never fitted.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

REPO=Path(__file__).resolve().parents[2];sys.path.insert(0,str(REPO))
sys.path.insert(0,str(REPO/'tools/eval'))


def camera(path):
    import rawpy
    from filmlab.rawio import sensor_saturation
    with rawpy.imread(str(path)) as r:
        sat=sensor_saturation(r,per_channel=True)
        rgb=r.postprocess(half_size=True,no_auto_bright=True,use_camera_wb=False,use_auto_wb=False,
                          user_wb=[1,1,1,1],gamma=(1,1),output_bps=16,output_color=rawpy.ColorSpace.raw).astype(float)/65535
    return rgb,sat


def main():
    import colour
    from PIL import Image,ImageDraw
    from scipy.ndimage import binary_dilation
    from skimage.feature import ORB,match_descriptors
    from skimage.measure import ransac
    from skimage.exposure import equalize_adapthist
    from skimage.transform import ProjectiveTransform,warp
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--data',type=Path,required=True)
    ap.add_argument('--audit',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    records=json.loads(args.audit.read_text())['scenes'];results=[]
    for d in records:
        options=[r for r in d['alternates'] if r['same_iso'] and r['optical_exposure_difference_stops']<0]
        if not options:continue
        short=min(options,key=lambda r:r['optical_exposure_difference_stops'])
        a,sa=camera(args.data/'epfl/data/digital'/d['primary']['file'])
        b,sb=camera(args.data/'epfl/data/digital'/short['file'])
        scale=2**-short['optical_exposure_difference_stops']
        features=[]
        for rgb in [a,b*scale]:
            gray=np.log1p(np.maximum(rgb[::2,::2].mean(-1),0)*100)
            gray=np.clip(gray/max(np.percentile(gray,99),1e-9),0,1)
            gray=equalize_adapthist(gray)
            orb=ORB(n_keypoints=5000,fast_threshold=.04);orb.detect_and_extract(gray)
            features.append((orb.keypoints[:,::-1],orb.descriptors))
        matches=match_descriptors(features[1][1],features[0][1],cross_check=True,max_ratio=.8)
        source=features[1][0][matches[:,0]];target=features[0][0][matches[:,1]]
        model,inliers=ransac((source,target),ProjectiveTransform,min_samples=4,residual_threshold=2,max_trials=3000,rng=0)
        if model is None or inliers.sum()<40:raise ValueError('Unreliable bracket geometry: '+d['scene'])
        # [::2] feature samples are at native indices 0,2,4 (no half-pixel offset).
        S=np.diag([2.,2.,1.]);native=ProjectiveTransform(S@model.params@np.linalg.inv(S))
        bw=warp(b,native.inverse,output_shape=a.shape[:2],cval=np.nan)
        # Require a full unsaturated neighbourhood before bilinear sampling.
        safe=~binary_dilation(sb.any(-1),iterations=1)
        safew=warp(safe.astype(float),native.inverse,output_shape=a.shape[:2],cval=0)>.999
        valid=np.isfinite(bw).all(-1)
        measured=valid&safew&~binary_dilation(sa.any(-1),iterations=1)
        # Avoid read-noise dominated and near-saturated values; fixed thresholds.
        measured&=(a.min(-1)>.02)&(a.max(-1)<.65)&(bw.min(-1)>.003)
        residual=np.log2(np.maximum(bw*scale,1e-9)/np.maximum(a,1e-9))
        recovered=sa.any(-1)&safew&valid
        record={'scene':d['scene'],'primary':d['primary']['file'],'short':short['file'],'exposure_scale':scale,
                'homography_short_to_primary':native.params.tolist(),'matches':len(matches),'inliers':int(inliers.sum()),
                'median_geometry_error_native_px':float(np.median(model.residuals(source[inliers],target[inliers]))*2),
                'measured_pixels':int(measured.sum()),
                'photometric_residual_stops_median_RGB':np.median(residual[measured],0).tolist(),
                'photometric_residual_stops_p10_p90_RGB':np.percentile(residual[measured],[10,90],axis=0).tolist(),
                'primary_any_saturated':int(sa.any(-1).sum()),'recoverable_with_all_short_channels_unsaturated':int(recovered.sum()),
                'primary_all_saturated':int(sa.all(-1).sum()),'all_saturated_recoverable':int((sa.all(-1)&safew&valid).sum()),
                'limits':'Whole-frame counts; registration uncertainty and moving/flickering sources remain. No HDR merge.'}
        np.savez_compressed(args.out/(d['scene']+'_bracket.npz'),primary=a.astype('float32'),short_normalized=(bw*scale).astype('float32'),
                            recovered=recovered,primary_saturation=sa,short_safe=safew)
        # Sensor-space RGB preview, equal exposure scale and display transform.
        # These are diagnostic false-colour sensor channels, not scene or film colours.
        panels=[]
        for label,rgb in [('Primary camera RGB',a),('Short camera RGB x metadata exposure ratio',bw*scale)]:
            disp=colour.models.eotf_inverse_sRGB(np.clip(np.nan_to_num(rgb),0,1))
            im=Image.fromarray(np.uint8(np.clip(disp*255+.5,0,255)));im.thumbnail((1152,866))
            panel=Image.new('RGB',(1152,910));panel.paste(im,(0,44));draw=ImageDraw.Draw(panel)
            draw.text((10,10),d['scene']+' | '+label,fill='white');panels.append(panel)
        sheet=Image.new('RGB',(2304,910));sheet.paste(panels[0],(0,0));sheet.paste(panels[1],(1152,0))
        sheet.save(args.out/(d['scene']+'_bracket_camera.png'))
        results.append(record);print(json.dumps(record),flush=True)
    (args.out/'bracket-registration.json').write_text(json.dumps({'scope':__doc__,'scenes':results},indent=2,allow_nan=False))


if __name__=='__main__':main()
