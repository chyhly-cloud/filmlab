"""Characterize output-scan texture, NOT isolated film grain or negative RMS density.

No fit to the film model. Plane/quadratic detrending, PSD and covariance are only
observations of the existing 8-bit sRGB scans. Repeated raw scans are not available.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def detrend(a):
    h,w=a.shape[:2];y,x=np.mgrid[-1:1:complex(h),-1:1:complex(w)]
    X=np.stack([np.ones_like(x),x,y,x*x,x*y,y*y],-1).reshape(-1,6)
    flat=a.reshape(h*w,-1)
    return (flat-X@np.linalg.lstsq(X,flat,rcond=None)[0]).reshape(a.shape)


def analyze(a,x0):
    encoded=detrend(a.astype(float))
    x=a.astype(float)/255
    linear=np.where(x<=.04045,x/12.92,((x+.055)/1.055)**2.4)
    Y=linear@np.array([.2126,.7152,.0722]);res=detrend(Y)
    var=float(np.mean(res*res))
    corr={}
    for n in (1,2,4,8):
        corr[str(n)]=[float(np.mean(res[:,:-n]*res[:,n:])/var),float(np.mean(res[:-n]*res[n:])/var)]
    phases=[];diff=np.diff(res,axis=1)
    for phase in range(8):phases.append(float(np.mean(diff[:,(x0+np.arange(diff.shape[1])+1)%8==phase]**2)))
    window=np.outer(np.hanning(a.shape[0]),np.hanning(a.shape[1]))
    power=np.abs(np.fft.fft2(res*window))**2/np.sum(window**2)
    f=np.fft.fftfreq(a.shape[0]);fy,fx=np.meshgrid(f,f,indexing='ij');radius=np.hypot(fx,fy)
    edges=np.linspace(0,.5,65);psd=[]
    for lo,hi in zip(edges[:-1],edges[1:]):
        mask=(radius>=lo)&(radius<hi);psd.append(float(power[mask].mean()) if mask.any() else 0.)
    return {"mean_codes":a.mean((0,1)).tolist(),"residual_std_codes":encoded.std((0,1)).tolist(),
            "distinct_codes":[len(np.unique(a[...,c])) for c in range(3)],
            "at_0_or_255_fraction":((a==0)|(a==255)).mean((0,1)).tolist(),
            "linear_Y_residual_std":float(np.std(res)),"RGB_residual_correlation":np.corrcoef(encoded.reshape(-1,3).T).tolist(),
            "autocorrelation_xy":corr,"block8_phase_energy_ratio":max(phases)/max(min(phases),1e-20),
            "row_column_mean_energy_relative_to_uncorrelated":
            [float(np.var(res.mean(1))/var*a.shape[1]),float(np.var(res.mean(0))/var*a.shape[0])],
            "frequency_cycles_per_pixel":((edges[:-1]+edges[1:])/2).tolist(),"radial_PSD_linear_Y":psd}


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--data',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    manifest=Path(__file__).with_name('epfl_800t_manifest.csv')
    rows={r['scene_id']:r for r in csv.DictReader(manifest.open())}
    samples=[];metadata=[]
    for sid in ('R1-19-20','R1-18-19'):
        path=args.data/'epfl/data'/rows[sid]['film_file']
        with path.open('rb') as stream:sha=hashlib.file_digest(stream,'sha256').hexdigest()
        if sha!=rows[sid]['film_sha256']:raise ValueError('source hash mismatch')
        with Image.open(path) as im:
            metadata.append({'scene':sid,'sha256':sha,'dimensions':list(im.size),'bits':list(im.tag_v2[258]),
                             'TIFF_compression':im.tag_v2.get(259),'software':im.tag_v2.get(305),
                             'sRGB_marker':b'sRGB' in im.info.get('icc_profile',b'')})
            image=np.asarray(im.convert('RGB'))
        h,w=image.shape[:2]
        if sid=='R1-19-20':
            locations=[(f'white_{i}_{j}',x,y) for i,x in enumerate((.3,.5,.7)) for j,y in enumerate((.3,.5,.7))]
        else:locations=[('chart_dark',308/900,280/602),('chart_middle',390/900,280/602),('chart_light',479/900,280/602)]
        overview=Image.fromarray(image);draw=ImageDraw.Draw(overview)
        for name,x,y in locations:
            cx,cy=int(round(x*w)),int(round(y*h));x0,y0=cx-128,cy-128
            patch=image[y0:y0+256,x0:x0+256]
            if patch.shape!=(256,256,3):raise ValueError('invalid ROI')
            draw.rectangle((x0,y0,x0+255,y0+255),outline='red',width=5)
            draw.text((x0,y0-15),name,fill='red')
            Image.fromarray(patch).resize((768,768),Image.Resampling.NEAREST).save(args.out/(name+'.png'))
            samples.append({'name':name,'scene':sid,'roi_xywh':[x0,y0,256,256],**analyze(patch,x0)})
        overview.thumbnail((1600,1600));overview.save(args.out/(sid+'_regions.jpg'))
    report={'scope':'texture of final scans; grain, scanner noise, sharpening and upstream processing not separated',
            'nominal_um_per_pixel':36000/5456,'detrend':'per-patch quadratic; may remove low-frequency signal',
            'metadata':metadata,'samples':samples,
            'not_identified':['negative-density RMS','scanner noise floor','sharpening kernel','upstream compression','film-only PSD']}
    (args.out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for s in samples:
        if s['name'] not in ('white_1_1','chart_dark','chart_middle','chart_light'):continue
        axes[0].semilogy(s['frequency_cycles_per_pixel'],s['radial_PSD_linear_Y'],label=s['name'])
        axes[1].plot([1,2,4,8],[s['autocorrelation_xy'][str(i)][0] for i in (1,2,4,8)],'o-',label=s['name'])
    axes[0].set(xlabel='Spatial frequency (cycles / scan pixel)',ylabel='Output linear-Y PSD',title='Observed scan texture, not film-only grain')
    axes[1].set(xlabel='Horizontal lag (scan pixels)',ylabel='Residual autocorrelation',title='Correlations do not identify their origin')
    for ax in axes:ax.grid(alpha=.3);ax.legend(fontsize=8)
    fig.savefig(args.out/'texture_diagnostics.png',dpi=160);plt.close(fig)
    print(json.dumps({'metadata':metadata,'patches':len(samples),'scope':report['scope']},ensure_ascii=False))


if __name__=='__main__':main()
