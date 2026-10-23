"""One physical closure hypothesis; fixed input/materials/scanner, no fitting."""
from pathlib import Path
from dataclasses import replace
from unittest.mock import patch
import json,sys,zlib
import numpy as np
REPO=Path(__file__).resolve().parents[2];sys.path[:0]=[str(REPO),str(REPO/'tools/eval')]


def main():
    import colour
    from PIL import Image,ImageDraw
    from skimage.transform import ProjectiveTransform
    from filmlab import generation,rawio,chemistry
    from filmlab.engine import Engine,Params
    from filmlab.recipes import recipe
    from filmlab.scanning.timing import PrintTiming
    from filmlab.spectral import srgb_encode
    from self_consistent_dir import solve
    from light_acceptance import pixel_scaling,crop_warp
    from epfl_highlights import digest
    import oracle_timing as ot
    out=Path('/workspace/filmlab-audit/dir-consistency');out.mkdir(exist_ok=True)
    sid='R1-02-3';generation.use('next_sensor');seed=zlib.crc32(sid.encode())%1000
    e=Engine(Params(**{**recipe('cinestill_800t'),'timing':'frame','roll':1,'frame':seed%36,'grain':0,'output_conditioning':None}))
    # Existing grid, reused solely to test this specified closure (no parameter fit).
    dev=chemistry.DEVELOPERS['C-41 CD'];native=chemistry.DEVELOPERS[chemistry.NATIVE_IMAGE_DEVELOPER[e.e.native_process]]
    tau=dev.activity/native.activity*e.v.tau_factor
    axis=np.linspace(e.e.log_exposure.min(),e.e.log_exposure.max(),25)
    grid=np.stack(np.meshgrid(axis,axis,axis,indexing='ij'),-1).reshape(-1,3)
    def f(x,**kw):return solve(e.e,x,tau,e.e.dir_amount,0,**kw)
    info={};d=f(grid,diagnostics=info);slower=f(grid,relaxation=.25)
    eps=1e-4;j=np.stack([(f(grid+np.eye(3)[c]*eps)-f(grid-np.eye(3)[c]*eps))/(2*eps) for c in range(3)],-1)
    neutral=np.repeat(np.linspace(-2,3,51)[:,None],3,axis=1)
    report={'candidate':'steady effective DIR closure, not measured reaction kinetics','tau':tau,'grid_points':len(grid),
            'grid_closure':info,'relaxation_solution_max_diff':float(np.max(abs(d-slower))),
            'negative_self_derivatives':(np.diagonal(j,axis1=1,axis2=2)<-1e-6).sum(0).tolist(),
            'self_derivative_min':np.diagonal(j,axis1=1,axis2=2).min(0).tolist(),
            'neutral_difference_max':float(np.max(abs(f(neutral)-chemistry.develop_image(e.e,neutral,tau,e.e.dir_amount,0)))),
            'regions':[]}
    rawpath,scanpath,bias=ot.sources(sid)
    for p,k in [(rawpath,'raw_sha256'),(scanpath,'film_sha256')]:
        if digest(p)!=ot.MAN[sid][k]:raise ValueError('source checksum mismatch')
    rgb=rawio.load_raw(rawpath,max_size=3000);metered,note=rawio.film_metering(rgb,rng=np.random.default_rng(seed),bias_ev=bias)
    negative=e.develop_negative(metered);q=negative.log_exposure;del rgb,metered
    print('baseline negative complete',flush=True)
    image_info={}
    def candidate(*args,**kw):return solve(*args,**kw,diagnostics=image_info)
    with patch.object(chemistry,'develop_image',candidate):state=e._develop(q,blur_px=12/e.px_um)
    e._agitation(state);e._grain(state)
    report['image_closure']=image_info;report['metering']=note
    print('candidate negative complete',image_info,flush=True)
    np.savez_compressed(out/'full_dye.npz',baseline=negative.state.dye.astype('float32'),candidate=state.dye.astype('float32'))
    prior=json.loads(Path('/workspace/filmlab-audit/epfl-highlights/R1-02-3.json').read_text());t=prior['timing']
    e.scanner.timing=PrintTiming(K=np.array(t['K']),s=np.array(t['s']),M=tuple(np.array(v) for v in t['M']),x_knee=t['x_knee'],x_white=t['x_white'])
    oldreal=np.load(Path(ot.GEO)/('geo_'+sid+'.npz'))['real'];oldh,oldw=oldreal.shape[:2]
    # Reuse the exact native crop transforms from the brightness audit.
    from PIL import Image
    with Image.open(scanpath) as im:h,w=np.asarray(im).shape[:2]
    step=max(1,round(max(q.shape[:2])/1400));H=pixel_scaling(w/oldw,h/oldh)@np.array(prior['geometry'])@pixel_scaling(1/step,1/step)
    for name,center in [('ceiling_lamp',(676,92)),('middle_ceiling',(706,235)),('glass_reflection',(1437,405))]:
        cache=np.load(Path('/workspace/filmlab-audit/brightness-local')/(sid+'_'+name+'.npz'))
        images={'real':cache['real'],'baseline':cache['base_exp']}
        x0,y0=np.round(np.array(center)*[w/oldw,h/oldh]).astype(int)-256
        corners=ProjectiveTransform(H).inverse(np.array([[x0,y0],[x0+512,y0],[x0,y0+512],[x0+512,y0+512]]))
        xl,yl=np.maximum(np.floor(corners.min(0)).astype(int)-3,0);xh,yh=np.minimum(np.ceil(corners.max(0)).astype(int)+4,[q.shape[1],q.shape[0]])
        matrix=np.array([[1,0,-x0],[0,1,-y0],[0,0,1]])@H@np.array([[1,0,xl],[0,1,yl],[0,0,1]])
        wp=crop_warp(ProjectiveTransform(matrix))
        def sub(st):return replace(st,**{k:v[yl:yh,xl:xh] for k,v in vars(st).items() if isinstance(v,np.ndarray) and v.shape[:2]==q.shape[:2]})
        for key,st in [('candidate',state),('reproduced',negative.state)]:
            n=replace(negative,state=sub(st));images[key]=wp(e.scanner.to_rgb(n.measure(e.scanner.normalized_density)))
        error=float(np.nanmax(abs(images['reproduced']-images['baseline'])))
        if error>2e-6:raise ValueError('baseline reproduction failed')
        from brightness_response_audit import lab
        labs={k:lab(v) for k,v in images.items()};dist=cache['distance_um'];valid=(cache['measured']>.5)&np.isfinite(images['candidate']).all(-1)
        rows=[]
        for lo,hi in [(0,300),(300,800)]:
            for bi in [1,2,3,4]:
                mask=valid&(dist>lo)&(dist<=hi)&(cache['real_bin']==bi)
                if mask.sum()<30:continue
                r={'distance':[lo,hi],'real_bin':bi,'pixels':int(mask.sum()),'outputs':{}}
                for key in ['baseline','candidate']:
                    gap=labs[key]-labs['real'];r['outputs'][key]={'median_dLab':np.median(gap[mask],axis=0).tolist(),
                        'median_deltaE00':float(np.median(colour.delta_E(labs[key][mask],labs['real'][mask],method='CIE 2000')))}
                rows.append(r)
        report['regions'].append({'name':name,'reproduction_error':error,'statistics':rows})
        sheet=Image.new('RGB',(1536,550),(20,20,20));draw=ImageDraw.Draw(sheet)
        for i,(key,label) in enumerate([('real','Real CineStill scan'),('baseline','Existing exp_plain'),('candidate','Experimental self-consistent DIR')]):
            a=np.uint8(np.clip(srgb_encode(np.nan_to_num(images[key]))*255+.5,0,255));sheet.paste(Image.fromarray(a),(512*i,38));draw.text((512*i+8,10),label,fill='white')
        sheet.save(out/(name+'_comparison.png'));np.savez_compressed(out/(name+'.npz'),**images)
    report['limits']=['source radiance/SPD not recovered','steady closure is a hypothesis, not calibrated 800T chemistry','single-frame validation cannot promote a default','scanner and all material constants unchanged']
    (out/'results.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
