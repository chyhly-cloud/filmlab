"""Per-source RAW bracket validation, fixed metadata exposure ratios.

Only geometric translations are optimized; no radiometric gain, spectral or
film/scanner parameter is fitted. Unknown spectrum remains unknown. Bright
surfaces have explicit control annotations, not automatic emitter labels.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

REPO=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(REPO),str(REPO/'tools/eval')]
from linear_raw_measurements import bayer_measurements


def global_geometry(a,b,scale):
    from skimage.feature import ORB,match_descriptors
    from skimage.measure import ransac
    from skimage.transform import ProjectiveTransform
    from skimage.exposure import equalize_adapthist
    features=[]
    for rgb in (a,b*scale):
        g=np.log1p(np.maximum(rgb[::2,::2].mean(-1),0)*100)
        g=equalize_adapthist(np.clip(g/max(np.percentile(g,99),1e-9),0,1))
        o=ORB(n_keypoints=5000,fast_threshold=.04);o.detect_and_extract(g)
        features.append((o.keypoints[:,::-1],o.descriptors))
    matches=match_descriptors(features[1][1],features[0][1],cross_check=True,max_ratio=.8)
    src=features[1][0][matches[:,0]];dst=features[0][0][matches[:,1]]
    m,ok=ransac((src,dst),ProjectiveTransform,min_samples=4,residual_threshold=2,max_trials=3000,rng=0)
    if m is None or ok.sum()<40:raise ValueError('Global geometry failed')
    S=np.diag([2.,2.,1.]);matrix=S@m.params@np.linalg.inv(S)
    return matrix,{'matches':len(matches),'inliers':int(ok.sum()),'median_error_px':float(np.median(m.residuals(src[ok],dst[ok]))*2)}


def align_lamp(a,sa,b,sb,matrix,scale,region):
    from scipy.ndimage import map_coordinates,binary_dilation,gaussian_filter
    from scipy.optimize import least_squares
    from skimage.transform import ProjectiveTransform
    x,y=region['center'];half=region.get('half',90)
    yy,xx=np.mgrid[y-half:y+half,x-half:x+half];r=np.hypot(xx-x,yy-y)
    local=np.array([[1,0,-(x-half)],[0,1,-(y-half)],[0,0,1]])@matrix
    inverse=ProjectiveTransform(local).inverse
    coords=inverse(np.column_stack([(xx-(x-half)).ravel(),(yy-(y-half)).ravel()]))
    bx=coords[:,0].reshape(xx.shape);by=coords[:,1].reshape(yy.shape)
    primary=a[yy,xx];primary_sat=sa[yy,xx]
    def sample(values,dx=0,dy=0,order=1):
        return np.stack([map_coordinates(values[...,c],[by+dy,bx+dx],order=order,mode='constant',cval=np.nan if order else 1)
                         for c in range(3)],-1)
    valid_a=(primary.min(-1)>.007)&(primary.max(-1)<.7)&~primary_sat.any(-1)
    background=(r>region['radius']+8)&(r<80)&valid_a
    # Hold out alternate pixels; use only surrounding surfaces for alignment.
    train=background&((xx+yy)%2==0);test=background&~train
    ref=np.log2(np.maximum(gaussian_filter(primary.mean(-1),.7),1e-5))
    initial=sample(b)*scale
    valid=np.isfinite(initial).all(-1)&(initial.min(-1)>.005)&(initial.max(-1)<.8*scale)
    train&=valid;test&=valid
    points=np.flatnonzero(train)[::3]
    def error(v):
        sample_b=sample(b,v[0],v[1])*scale
        log=np.log2(np.maximum(gaussian_filter(np.nan_to_num(sample_b.mean(-1)),.7),1e-5))
        return (log-ref).ravel()[points]
    if len(points)>=60:
        result=least_squares(error,[0.,0.],bounds=([-5,-5],[5,5]),loss='soft_l1',f_scale=.15,max_nfev=35,diff_step=.02)
        delta=result.x
    else:delta=np.zeros(2)
    measured=sample(b,*delta)*scale
    # Strict all-channel support: an unsaturated 3x3 neighbourhood is required.
    unsafe=binary_dilation(sb.any(-1),iterations=1)
    safe=map_coordinates((~unsafe).astype(float),[by+delta[1],bx+delta[0]],order=1,mode='constant',cval=0)>.999
    # Separate holdout geometry error from actual light stability.
    log=np.log2(np.maximum(gaussian_filter(np.nan_to_num(measured.mean(-1)),.7),1e-5))
    residual=(log-ref)[test]
    median=float(np.median(np.abs(residual))) if residual.size else None
    p90=float(np.percentile(np.abs(residual),90)) if residual.size else None
    geometry_ok=len(points)>=60 and np.max(np.abs(delta))<4.95 and median is not None and median<.15 and p90<.5
    masks=(r<=region['radius'])
    return {'camera':measured,'sat':sample(sb.astype(float),*delta,order=0)>.5,'safe':safe,
            'mask':masks,'primary':primary,'primary_sat':primary_sat,'origin':[x-half,y-half],
            'local_inverse_coordinates':np.stack([by+delta[1],bx+delta[0]]).astype('float32'),
            'geometry':{'translation_xy':delta.tolist(),'train_points':len(points),'holdout_median_abs_stops':median,
                        'holdout_p90_abs_stops':p90,'pass':bool(geometry_ok)}}


def pair_stability(a,b,mask):
    """Per-channel agreement at measured signal levels; no fitted brightness gain."""
    reports=[]
    for c in range(3):
        use=mask&~a['sat'][...,c]&~b['sat'][...,c]&np.isfinite(a['camera'][...,c])&np.isfinite(b['camera'][...,c])
        use&=(a['camera'][...,c]>.03)&(b['camera'][...,c]>.03)
        # Exclude one-pixel source edges: residual registration should not masquerade as flicker.
        from scipy.ndimage import gaussian_gradient_magnitude
        for image in (a,b):
            g=gaussian_gradient_magnitude(np.log2(np.maximum(np.nan_to_num(image['camera'][...,c]),1e-5)),1)
            use&=g<.3
        if use.sum()>=8:
            delta=np.log2(b['camera'][...,c][use]/a['camera'][...,c][use])
            reports.append({'channel':'RGB'[c],'pixels':int(use.sum()),'median_stops':float(np.median(delta)),
                            'p10_p90_stops':np.percentile(delta,[10,90]).tolist()})
        else:reports.append({'channel':'RGB'[c],'pixels':int(use.sum()),'median_stops':None})
    return reports


def previews(region,views,out):
    from PIL import Image,ImageDraw
    panels=[]
    # Diagnostic camera channels, same scale for every exposure; not film colours.
    for name,v in views.items():
        a=np.maximum(np.nan_to_num(v['camera']),0)
        display=np.power(np.clip(a/np.maximum(1,a.max(-1,keepdims=True)),0,1),1/2.2)
        im=Image.fromarray(np.uint8(display*255+.5)).resize((540,540),Image.Resampling.NEAREST)
        panel=Image.new('RGB',(540,595),(18,18,18));panel.paste(im,(0,55))
        d=ImageDraw.Draw(panel);d.text((8,6),region['scene']+' '+region['name']+' | '+name,fill='white')
        d.text((8,28),'camera RGB; fixed metadata scale; chromaticity preview',fill='white')
        panels.append(panel)
    sheet=Image.new('RGB',(540*len(panels),595))
    for i,p in enumerate(panels):sheet.paste(p,(540*i,0))
    sheet.save(out/(region['scene']+'_'+region['name']+'_raw.png'))


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--data',type=Path,required=True)
    ap.add_argument('--brackets',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    regions=json.loads((REPO/'tools/eval/measured_lamp_regions.json').read_text())['regions']
    records=json.loads(args.brackets.read_text())['scenes'];results=[]
    def checked_read(row):
        path=args.data/'epfl/data/digital'/row['file']
        with path.open('rb') as stream:
            if hashlib.file_digest(stream,'sha256').hexdigest()!=row['sha256']:
                raise ValueError('RAW acquisition checksum mismatch: '+row['file'])
        return bayer_measurements(path)
    for record in records:
        sid=record['scene'];a,sa,metadata=checked_read(record['primary'])
        alternatives=[]
        for row in record['alternates']:
            if not row['same_iso']:continue
            b,sb,meta=checked_read(row)
            if metadata!=meta:raise ValueError('Sensor normalization/WB changed')
            scale=2**-row['optical_exposure_difference_stops']
            matrix,info=global_geometry(a,b,scale)
            alternatives.append((row,b,sb,scale,matrix,info))
            print(sid,row['file'],'global done',flush=True)
        for region in [r for r in regions if r['scene']==sid]:
            region=dict(region)
            x,y=region['center'];half=min(90,x,y,a.shape[1]-x,a.shape[0]-y);region['half']=half
            yy,xx=np.mgrid[y-half:y+half,x-half:x+half];mask=np.hypot(xx-x,yy-y)<=region['radius']
            views={'primary':{'camera':a[yy,xx],'sat':sa[yy,xx],'safe':~sa[yy,xx].any(-1),'mask':mask,'geometry':{'pass':True},'origin':[x-half,y-half]}}
            details=[]
            for row,b,sb,scale,matrix,info in alternatives:
                v=align_lamp(a,sa,b,sb,matrix,scale,region);views[row['file']]=v
                details.append({'file':row['file'],'scale':scale,'global_geometry':info,'local_geometry':v['geometry'],
                                'recoverable_any':int((mask&sa[yy,xx].any(-1)&v['safe']).sum()),
                                'recoverable_all':int((mask&sa[yy,xx].all(-1)&v['safe']).sum()),
                                'remaining_any':int((mask&v['sat'].any(-1)).sum()),'remaining_all':int((mask&v['sat'].all(-1)).sum())})
            comparisons=[]
            names=list(views)
            for i,first in enumerate(names):
                for second in names[i+1:]:
                    comparisons.append({'first':first,'second':second,'channels':pair_stability(views[first],views[second],mask)})
            previews(region,views,args.out)
            arrays={}
            for i,(name,v) in enumerate(views.items()):
                for field in ('camera','sat','safe'):
                    arrays[f'{i}_{field}']=v[field].astype('float32') if field=='camera' else v[field]
                if 'local_inverse_coordinates' in v:arrays[f'{i}_coordinates']=v['local_inverse_coordinates']
            np.savez_compressed(args.out/(sid+'_'+region['name']+'_measurements.npz'),**arrays,mask=mask)
            results.append({'region':region,'primary_file':record['primary']['file'],'views':names,'alternatives':details,
                            'primary_any_saturated':int((sa[yy,xx].any(-1)&mask).sum()),
                            'primary_all_saturated':int((sa[yy,xx].all(-1)&mask).sum()),'comparisons':comparisons})
            print(sid,region['name'],'done',flush=True)
        (args.out/'lamp-measurements.partial.json').write_text(json.dumps(results,indent=2,allow_nan=False))
    report={'scope':__doc__,'thresholds':{'geometry_holdout_median_abs_EV':.15,'geometry_p90_abs_EV':.5,'max_translation_px':4.95},'lamps':results}
    (args.out/'lamp-measurements.json').write_text(json.dumps(report,indent=2,allow_nan=False))


if __name__=='__main__':main()
