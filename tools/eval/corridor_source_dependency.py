"""Partition modeled corridor return exposure by RAW sensor information coverage.

No spectral/radiance recovery, chemistry, scan response or fitting. Saturation
classes describe source pixels, not the receiving pixels. 'unflagged' does not
mean a measured spectrum; RGB illumination reconstruction remains a proxy.
"""
from pathlib import Path
from unittest.mock import patch
import json,sys,zlib
import numpy as np
REPO=Path(__file__).resolve().parents[2];sys.path[:0]=[str(REPO),str(REPO/'tools/eval')]


def plot_report(report,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    labels=['Near R','Near G','Near B','Middle R','Middle G','Middle B','Glass R','Glass G','Glass B']
    for ax,distance in zip(axes,[[0,300],[300,800]]):
        rows=[next(s for s in r['statistics'] if s['distance']==distance and s['real_bin']=='all') for r in report['regions']]
        bottom=np.zeros(9);x=np.arange(9)
        for key,color in [('unflagged','#4a9c8b'),('partly_saturated','#e6ae46'),('fully_saturated','#c85859')]:
            values=np.array([v for row in rows for v in row['source_fraction_RGB'][key]])*100
            ax.bar(x,values,bottom=bottom,color=color,label=key);bottom+=values
        ax.set_xticks(x,labels,rotation=45,ha='right');ax.set_ylim(0,100)
        ax.set_ylabel('Share of modeled returned layer exposure (%)')
        ax.set_title(f'{distance[0]}–{distance[1]} µm receiving surroundings')
    axes[0].legend(fontsize=8);fig.suptitle('Where model halo exposure comes from; source classes ≠ measured source spectra',fontsize=11)
    fig.savefig(out/'source_dependency.png',dpi=170);plt.close(fig)


def main():
    import rawpy
    from PIL import Image
    from skimage.transform import ProjectiveTransform
    from filmlab import generation,rawio,film
    from filmlab.engine import Engine,Params
    from filmlab.recipes import recipe
    from light_acceptance import crop_warp,pixel_scaling
    from brightness_response_audit import LABELS
    from epfl_highlights import digest
    import oracle_timing as ot
    out=Path('/workspace/filmlab-audit/corridor-source-dependency');out.mkdir(exist_ok=True)
    sid='R1-02-3';generation.use('next_sensor');seed=zlib.crc32(sid.encode())%1000
    rawpath,scanpath,bias=ot.sources(sid)
    for p,key in [(rawpath,'raw_sha256'),(scanpath,'film_sha256')]:
        if digest(p)!=ot.MAN[sid][key]:raise ValueError('Source checksum mismatch')
    rgb=rawio.load_raw(rawpath,max_size=3000)
    with rawpy.imread(rawpath) as raw:sat=rawio.sensor_saturation(raw,per_channel=True)
    if sat.shape!=rgb.shape:raise ValueError('Saturation geometry mismatch')
    masks={'unflagged':~sat.any(-1),'partly_saturated':sat.any(-1)&~sat.all(-1),'fully_saturated':sat.all(-1)}
    assert np.all(sum(m.astype(int) for m in masks.values())==1)
    metered,note=rawio.film_metering(rgb,rng=np.random.default_rng(seed),bias_ev=bias)
    e=Engine(Params(**{**recipe('cinestill_800t'),'timing':'frame','roll':1,'frame':seed%36,'grain':0,'output_conditioning':None}))
    captured={}
    class Captured(Exception):pass
    def capture(H,source,px,v,rho):
        captured.update(direct=H,source=source,px=px,v=v,rho=rho)
        raise Captured()
    with patch.object(film,'halation',capture):
        try:e.develop_negative(metered)
        except Captured:pass
    if not captured:raise ValueError('Halation path not reached')
    del rgb,metered
    print('source captured; no development/render performed',flush=True)
    prior=json.loads(Path('/workspace/filmlab-audit/epfl-highlights/R1-02-3.json').read_text())
    oldreal=np.load(Path(ot.GEO)/('geo_'+sid+'.npz'))['real']
    with Image.open(scanpath) as im:w,h=im.size
    step=max(1,round(max(sat.shape[:2])/1400))
    H=pixel_scaling(w/oldreal.shape[1],h/oldreal.shape[0])@np.array(prior['geometry'])@pixel_scaling(1/step,1/step)
    items=[]
    for name,center in [('ceiling_lamp',(676,92)),('middle_ceiling',(706,235)),('glass_reflection',(1437,405))]:
        a=np.load(Path('/workspace/filmlab-audit/brightness-local')/(sid+'_'+name+'.npz'))
        x0,y0=np.round(np.array(center)*[w/oldreal.shape[1],h/oldreal.shape[0]]).astype(int)-256
        wp=crop_warp(ProjectiveTransform(np.array([[1,0,-x0],[0,1,-y0],[0,0,1]])@H))
        items.append(dict(name=name,warp=wp,measured=a['measured']>.5,dist=a['distance_um'],real_bin=a['real_bin'],arrays={}))
    def return_exposure(source):return film.halation(np.zeros_like(captured['direct']),source,captured['px'],captured['v'],captured['rho'])
    total=return_exposure(captured['source']);summed=np.zeros_like(total)
    for item in items:
        item['arrays']['total']=item['warp'](total)
        item['arrays']['direct']=item['warp'](captured['direct'])
    for label,mask in masks.items():
        contribution=return_exposure(captured['source']*mask[...,None]);summed+=contribution
        for item in items:item['arrays'][label]=item['warp'](contribution)
        print(label,'complete',flush=True)
    closure={'max_absolute_error_RGB':np.max(abs(summed-total),axis=(0,1)).tolist(),
             'relative_L1_error_RGB':(abs(summed-total).sum((0,1))/np.maximum(abs(total).sum((0,1)),1e-30)).tolist()}
    if max(closure['relative_L1_error_RGB'])>1e-12:raise ValueError('Linear partition fails closure')
    report={'scene':sid,'seed':seed,'metering':note,'highlight_reconstruction':rawio.HIGHLIGHT_RECON,'source_classes_counts':{k:int(v.sum()) for k,v in masks.items()},
            'source_hashes':{k:ot.MAN[sid][k] for k in ['raw_sha256','film_sha256']},'closure':closure,'regions':[],
            'geometry_policy':'same prior transform, 512 crops, receiving masks/distances and real brightness bins from brightness-local',
            'units':'model sensitivity-calibrated linear layer exposure; R/G/B sums are not optical energy',
            'limits':['unflagged sensor pixels still use RGB spectral inference and processing; no spectrum measured',
                      'saturation masks are not the complete spatial coverage of highlight reconstruction; dilated processing may affect neighboring unflagged pixels',
                      'partly saturated class contains both measured and missing channels; its whole modeled RGB signal is classified, not claimed wholly unmeasured',
                      'fully saturated class relies on existing reconstructed prior; no new prior or strength selected',
                      'return contributions can originate outside named lamp and ROI','current/next unchanged; this is source-dependence, not true halation measurement',
                      '36mm nominal scan width and fixed registration limit micron precision']}
    for item in items:
        rows=[]
        for lo,hi in [(0,300),(300,800)]:
            spatial=item['measured']&(item['dist']>lo)&(item['dist']<=hi)&np.isfinite(item['arrays']['total']).all(-1)
            for bi in [None,0,1,2,3,4]:
                mask=spatial if bi is None else spatial&(item['real_bin']==bi)
                if mask.sum()<30:continue
                denominator=item['arrays']['total'][mask].sum(0)
                row={'distance':[lo,hi],'real_bin':'all' if bi is None else LABELS[bi],'receiving_pixels':int(mask.sum()),
                     'return_exposure_sum_RGB':denominator.tolist(),
                     'halo_fraction_of_total_exposure_RGB':(denominator/np.maximum((item['arrays']['direct'][mask]+item['arrays']['total'][mask]).sum(0),1e-30)).tolist(),
                     'median_added_log2_exposure_RGB':np.median(np.log2(1+np.maximum(item['arrays']['total'][mask],0)/np.maximum(item['arrays']['direct'][mask],1e-30)),axis=0).tolist(),
                     'source_fraction_RGB':{}}
                for label in masks:
                    row['source_fraction_RGB'][label]=(item['arrays'][label][mask].sum(0)/np.maximum(denominator,1e-30)).tolist()
                rows.append(row)
        report['regions'].append({'name':item['name'],'statistics':rows})
        np.savez_compressed(out/(item['name']+'.npz'),**item['arrays'],measured=item['measured'],distance_um=item['dist'],real_bin=item['real_bin'])
    (out/'results.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    plot_report(report,out)
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
