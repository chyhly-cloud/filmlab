"""Three targeted causal controls, not fitted film/scanner candidates.

R-only return removes G/B secondary exposure (unphysical as a proposed film).
First return omits later trips without resizing or renormalizing its kernel.
Separable paper keeps Dp and the exact original neutral response, removing
nonneutral spectral dye/LUT coupling; not measured Frontier output.
"""
from pathlib import Path
from dataclasses import replace
from unittest.mock import patch
import argparse,json,sys,zlib
import numpy as np
REPO=Path(__file__).resolve().parents[2];sys.path[:0]=[str(REPO),str(REPO/'tools/eval')]


def independent_paper_rgb(dp,k,black):
    """Ideal independent channels with the same finite paper black and white."""
    return black+(1-black)*(10.**(-np.asarray(dp))-k)/(1-k)


def separable_paper_rgb(paper,dp,black):
    """Remove nonneutral cross-channel coupling, matching each neutral exactly.

    For channel c, evaluate the original paper on (Dp_c,Dp_c,Dp_c) and take c.
    This preserves the full neutral response without a fit, new LUT or tone shift.
    It is an isolation control, not a realizable measured paper/scanner model.
    """
    dp=np.asarray(dp);out=np.empty_like(dp,dtype=float)
    for c in range(3):
        neutral=np.repeat(dp[...,c,None],3,axis=-1)
        out[...,c]=paper.reference_print(neutral,black).reshape(dp.shape)[...,c]
    return out


def main():
    import colour
    from PIL import Image,ImageDraw,ImageFont
    from skimage.transform import ProjectiveTransform
    from filmlab import generation,rawio,film
    from filmlab.engine import Engine,Params
    from filmlab.recipes import recipe
    from filmlab.scanning.timing import PrintTiming
    from filmlab.scanning.paper import OUTPUT_BLACK
    from filmlab.spectral import srgb_encode
    from light_acceptance import pixel_scaling,crop_warp
    from epfl_highlights import digest
    from brightness_response_audit import lab
    import oracle_timing as ot
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--scene',default='R1-02-3');args=ap.parse_args()
    sid=args.scene;out=Path('/workspace/filmlab-audit/halo-mechanisms')/sid;out.mkdir(exist_ok=True,parents=True)
    generation.use('next_sensor');seed=zlib.crc32(sid.encode())%1000
    e=Engine(Params(**{**recipe('cinestill_800t'),'timing':'frame','roll':1,'frame':seed%36,'grain':0,'output_conditioning':None}))
    rawpath,scanpath,bias=ot.sources(sid)
    for p,k in [(rawpath,'raw_sha256'),(scanpath,'film_sha256')]:
        if digest(p)!=ot.MAN[sid][k]:raise ValueError('source checksum mismatch')
    rgb=rawio.load_raw(rawpath,max_size=3000);metered,note=rawio.film_metering(rgb,rng=np.random.default_rng(seed),bias_ev=bias)
    captured={};original=film.halation
    def capture(H,source,px,v,rho):
        captured.update(direct=H.copy(),source=source.copy(),rho=rho)
        return original(H,source,px,v,rho)
    with patch.object(film,'halation',capture):negative=e.develop_negative(metered)
    del rgb,metered
    q=negative.log_exposure
    print(sid,'baseline complete',flush=True)
    prior=json.loads((Path('/workspace/filmlab-audit/epfl-highlights')/(sid+'.json')).read_text());t=prior['timing']
    e.scanner.timing=PrintTiming(K=np.array(t['K']),s=np.array(t['s']),M=tuple(np.array(v) for v in t['M']),x_knee=t['x_knee'],x_white=t['x_white'])
    oldreal=np.load(Path(ot.GEO)/('geo_'+sid+'.npz'))['real'];oh,ow=oldreal.shape[:2]
    with Image.open(scanpath) as im:h,w=np.asarray(im).shape[:2]
    step=max(1,round(max(q.shape[:2])/1400));H=pixel_scaling(w/ow,h/oh)@np.array(prior['geometry'])@pixel_scaling(1/step,1/step)
    all_regions=json.loads((REPO/'tools/eval/light_acceptance_regions.json').read_text())['regions']
    regions=[(r['name'],r['center']) for r in all_regions if r['scene']==sid and r['name']!='red_beacon']
    if sid=='R1-02-3':regions+=[('middle_ceiling',(706,235)),('glass_reflection',(1437,405))]
    items=[]
    for name,center in regions:
        cache=np.load(Path('/workspace/filmlab-audit/brightness-local')/(sid+'_'+name+'.npz'))
        images={'real':cache['real'],'baseline':cache['base_exp']}
        x0,y0=np.round(np.array(center)*[w/ow,h/oh]).astype(int)-256
        corners=ProjectiveTransform(H).inverse(np.array([[x0,y0],[x0+512,y0],[x0,y0+512],[x0+512,y0+512]]))
        xl,yl=np.maximum(np.floor(corners.min(0)).astype(int)-3,0);xh,yh=np.minimum(np.ceil(corners.max(0)).astype(int)+4,[q.shape[1],q.shape[0]])
        matrix=np.array([[1,0,-x0],[0,1,-y0],[0,0,1]])@H@np.array([[1,0,xl],[0,1,yl],[0,0,1]])
        items.append(dict(name=name,images=images,warp=crop_warp(ProjectiveTransform(matrix)),bounds=(xl,yl,xh,yh),
                          dist=cache['distance_um'],measured=cache['measured']>.5,real_bin=cache['real_bin'],stages={}))
    def collect(st,label,logH):
        for item in items:
            xl,yl,xh,yh=item['bounds'];wp=item['warp']
            sub=replace(st,**{k:v[yl:yh,xl:xh] for k,v in vars(st).items() if isinstance(v,np.ndarray) and v.shape[:2]==q.shape[:2]})
            trans=e._transmittance(sub);dn=e.scanner.normalized_density(trans)
            xe=e.scanner.timing.paper_exposure(dn);dp=e.scanner.digital_paper.density(xe)
            output=e.scanner.to_rgb(dn)
            item['images'][label]=wp(output)
            item['stages'][label]={'dye':wp(sub.dye),'Dn':wp(dn),'Dp':wp(dp),'logH':wp(logH[yl:yh,xl:xh])}
            if label=='reproduced':
                item['images']['independent_paper']=wp(np.maximum(separable_paper_rgb(e.scanner.digital_paper,dp,OUTPUT_BLACK),0))
                error=float(np.nanmax(abs(item['images'][label]-item['images']['baseline'])))
                item['reproduction_error']=error
                if error>2e-6:raise ValueError('baseline not reproduced')
    collect(negative.state,'reproduced',q)
    # These are defined mechanism removals, never new strengths selected from a score.
    total=original(captured['direct'],captured['source'],e.px_um,e.v,captured['rho'])
    red_only=captured['direct'].copy();red_only[...,0]=total[...,0];del total
    for label,exposure in [('red_only',red_only),('first_return',None)]:
        if exposure is None:exposure=original(captured['direct'],captured['source'],e.px_um,e.v,(0,0,0))
        logH=e._log_exposure(np.maximum(exposure,1e-9));st=e._develop(logH,blur_px=12/e.px_um)
        e._agitation(st);e._grain(st);collect(st,label,logH)
        print(sid,label,'complete',flush=True)
    report={'scope':__doc__,'scene':sid,'timing':t,'seed':seed,'generation':'next_sensor','grain':0,'conditioning':None,
            'source_hashes':{k:ot.MAN[sid][k] for k in ['raw_sha256','film_sha256']},'metering':note,'regions':[]}
    keys=['baseline','red_only','first_return','independent_paper']
    for item in items:
        images=item['images'];labs={k:lab(v) for k,v in images.items()};rows=[];dist=item['dist'];valid=item['measured']
        for lo,hi in [(0,300),(300,800)]:
            for bi in [1,2,3,4]:
                mask=valid&(dist>lo)&(dist<=hi)&(item['real_bin']==bi)
                for key in keys:mask&=np.isfinite(images[key]).all(-1)
                if mask.sum()<30:continue
                row={'distance':[lo,hi],'real_bin':bi,'pixels':int(mask.sum()),'outputs':{}}
                for key in keys:
                    gap=labs[key]-labs['real'];rgb=images[key][mask]
                    row['outputs'][key]={'median_dLab':np.median(gap[mask],axis=0).tolist(),
                        'median_deltaE00':float(np.median(colour.delta_E(labs[key][mask],labs['real'][mask],method='CIE 2000'))),
                        'display_clipped_fraction':float(((rgb<0)|(rgb>1)).any(-1).mean())}
                rows.append(row)
        profiles=[]
        for lo in range(0,800,50):
            mask=valid&(dist>lo)&(dist<=lo+50)
            for key in keys:mask&=np.isfinite(images[key]).all(-1)
            if mask.sum()<30:continue
            profiles.append({'radius':[lo,lo+50],'pixels':int(mask.sum()),'median_Lab':{k:np.median(labs[k][mask],axis=0).tolist() for k in ['real']+keys}})
        sheet=Image.new('RGB',(2560,562),(20,20,20));draw=ImageDraw.Draw(sheet)
        labels=[('real','Real scan'),('baseline','Existing exp_plain'),('red_only','Diagnostic: red return only'),('first_return','Diagnostic: first return only'),('independent_paper','Diagnostic: separable paper colour')]
        font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16)
        for i,(key,label) in enumerate(labels):
            a=np.uint8(np.clip(srgb_encode(np.nan_to_num(images[key]))*255+.5,0,255));sheet.paste(Image.fromarray(a),(512*i,50));draw.text((512*i+8,8),label,font=font,fill='white')
            draw.text((512*i+8,29),sid+'_'+item['name'],font=font,fill='white')
        sheet.save(out/(item['name']+'_comparison.png'))
        arrays={**images,**{key+'_'+s:v for key,values in item['stages'].items() for s,v in values.items()},'distance_um':dist,'measured':valid,'real_bin':item['real_bin']}
        np.savez_compressed(out/(item['name']+'.npz'),**arrays)
        report['regions'].append({'name':item['name'],'reproduction_error':item['reproduction_error'],'statistics':rows,'radial_profiles':profiles})
    report['limits']=['diagnostic removals are not physically calibrated 800T candidates','source cores radiance/SPD remain unknown',
                      'nominal 36mm scale and registration limit radial accuracy','real scan has grain/sharpening; model grain off',
                      'neutral-matched separable RGB is not measured device colour','no fitting, no kernel resizing or energy renormalization']
    (out/'results.json').write_text(json.dumps(report,indent=2));print('saved',out,flush=True)


if __name__=='__main__':main()
