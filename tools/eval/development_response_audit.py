"""Development-only structural checks and real-light curve coverage; no fitting.

Single-layer perturbations are diagnostic inputs, not realizable measured spectra.
Zero-DIR is an ablation, not a candidate chemistry. Native-process consistency is
an internal construction check, not independent validation of the source curves.
"""
from pathlib import Path
from dataclasses import replace
import json,sys,zlib
import numpy as np
REPO=Path(__file__).resolve().parents[2];sys.path[:0]=[str(REPO),str(REPO/'tools/eval')]

def main():
    from filmlab import generation,chemistry,rawio
    from filmlab.engine import Engine,Params
    from filmlab.recipes import recipe
    from skimage.transform import ProjectiveTransform
    from light_acceptance import pixel_scaling
    import oracle_timing as ot
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=Path('/workspace/filmlab-audit/development-response');out.mkdir(exist_ok=True)
    generation.use('next_sensor');sid='R1-02-3';seed=zlib.crc32(sid.encode())%1000
    eng=Engine(Params(**{**recipe('cinestill_800t'),'timing':'frame','roll':1,'frame':seed%36,'grain':0,'output_conditioning':None}))
    e=eng.e;lo,hi=float(e.log_exposure.min()),float(e.log_exposure.max())
    axes=np.linspace(lo,hi,25);grid=np.stack(np.meshgrid(axes,axes,axes,indexing='ij'),-1).reshape(-1,3)
    def develop(x,film=None,proc=None):return chemistry.run(film or e,proc or eng.proc,x,tau_factor=eng.v.tau_factor).dye
    base=develop(grid);eps=1e-4
    jac=np.stack([(develop(grid+np.eye(3)[c]*eps)-develop(grid-np.eye(3)[c]*eps))/(2*eps) for c in range(3)],-1)
    selfj=np.diagonal(jac,axis1=1,axis2=2)
    report={'limits':__doc__,'logH_domain':[lo,hi],'grid_points':len(grid),'jacobian_units':'model dye units per log10 exposure','self_derivative_min':selfj.min(0).tolist(),'negative_self_derivative_count':(selfj < -1e-6).sum(0).tolist(),'zero_self_derivative_fraction':(np.abs(selfj)<1e-6).mean(0).tolist(),'max_absolute_cross_derivative':[[float(np.max(np.abs(jac[:,i,j]))) for j in range(3)] for i in range(3)]}
    # Neutral native-process reconstruction: compare the same nominal data source.
    neutral=np.repeat(axes[:,None],3,axis=1)
    native=chemistry.run(e,chemistry.PROCESSES[e.native_process],neutral,tau_factor=1).dye
    nominal=np.stack([np.interp(axes,e.log_exposure,e.S_nom[:,c]) for c in range(3)],-1)
    report['native_neutral_construction_max_abs_error']=np.max(np.abs(native-nominal),axis=0).tolist()
    dense=np.linspace(lo-1,hi+1,600);fig,ax=plt.subplots(1,3,figsize=(15,4),constrained_layout=True)
    selected=[]
    for j,bg in enumerate([lo+(hi-lo)*.25,lo+(hi-lo)*.5,lo+(hi-lo)*.75]):
        x=np.full((len(dense),3),bg);x[:,0]=dense
        full=develop(x);nodir=develop(x,replace(e,dir_amount=0))
        for c,color in enumerate(['red','green','blue']):
            ax[j].plot(dense,full[:,c],color=color,label='CMY'[c]+' baseline')
            ax[j].plot(dense,nodir[:,c],color=color,ls='--',alpha=.6,label='CMY'[c]+' no DIR')
        ax[j].axvspan(lo-1,lo,color='gray',alpha=.15);ax[j].axvspan(hi,hi+1,color='gray',alpha=.15)
        ax[j].set(title=f'Other layers logH={bg:.2f}',xlabel='Red-sensitive layer log10 H',ylabel='Model dye amount')
    ax[0].legend(fontsize=7);fig.savefig(out/'red_layer_response.png',dpi=170);plt.close(fig)
    np.savez_compressed(out/'synthetic.npz',grid=grid,dye=base,jacobian=jac,neutral_input=neutral,native_dye=native,nominal_dye=nominal)
    print('synthetic checks complete',flush=True)
    rawpath,scanpath,bias=ot.sources(sid);rgb=rawio.load_raw(rawpath,max_size=3000)
    metered,note=rawio.film_metering(rgb,rng=np.random.default_rng(seed),bias_ev=bias)
    trace={};neg=eng.develop_negative(metered,trace=trace)
    log0=eng._log_exposure(np.clip(trace['scattered_exposure'],1e-9,None));log1=neg.log_exposure
    old=json.loads(Path('/workspace/filmlab-audit/epfl-highlights/R1-02-3.json').read_text())
    step=max(1,round(max(rgb.shape[:2])/1400));H=np.array(old['geometry'])@pixel_scaling(1/step,1/step)
    yy,xx=np.indices(rgb.shape[:2]);coords=ProjectiveTransform(H)(np.stack([xx.ravel(),yy.ravel()],-1)).reshape(*xx.shape,2)
    report['real_regions']=[]
    for name,center in [('near_ceiling',(676,92)),('middle_ceiling',(706,235)),('glass_reflection',(1437,405))]:
        distance=np.linalg.norm(coords-np.array(center),axis=-1)*36000/1600
        mask=distance<300;x=log1[mask];pre=log0[mask]
        # Local homogeneous perturbation diagnostic: excludes spatial DIR diffusion.
        dye=develop(x);dplus=develop(x+np.array([eps,0,0]));dminus=develop(x-np.array([eps,0,0]))
        gain=(dplus-dminus)/(2*eps)
        noDIR=develop(x,replace(e,dir_amount=0))
        report['real_regions'].append({'name':name,'pixels':int(mask.sum()),'logH_p05_p50_p95_RGB':np.percentile(x,[5,50,95],axis=0).tolist(),'fraction_below_curve_RGB':(x<lo).mean(0).tolist(),'fraction_above_curve_RGB':(x>hi).mean(0).tolist(),'median_halo_added_logH_RGB':np.median(x-pre,axis=0).tolist(),'median_local_red_perturbation_dye_derivative_CMY':np.median(gain,axis=0).tolist(),'fraction_local_red_gain_zero':float((np.abs(gain[:,0])<1e-6).mean()),'median_noDIR_minus_full_dye_CMY':np.median(noDIR-dye,axis=0).tolist()})
        np.savez_compressed(out/(name+'.npz'),logH=x,pre_halo_logH=pre,local_dye=dye,local_noDIR_dye=noDIR,local_red_gain=gain)
    render_ablations(eng,neg,log1,old,scanpath,out)
    (out/'results.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
def render_ablations(eng,negative,logH,prior,scanpath,out):
    """Fixed scanner and timing; diagnostic removals, not replacement physics."""
    import colour
    from PIL import Image,ImageDraw
    from skimage.transform import ProjectiveTransform
    from filmlab import chemistry
    from filmlab.scanning.timing import PrintTiming
    from filmlab.spectral import srgb_encode
    from light_acceptance import crop_warp,pixel_scaling
    t=prior['timing']
    eng.scanner.timing=PrintTiming(K=np.array(t['K']),s=np.array(t['s']),M=tuple(np.array(v) for v in t['M']),x_knee=t['x_knee'],x_white=t['x_white'])
    noDIR=chemistry.run(replace(eng.e,dir_amount=0),eng.proc,logH,blur_px=12/eng.px_um,tau_factor=eng.v.tau_factor)
    eng._agitation(noDIR);eng._grain(noDIR)
    with Image.open(scanpath) as im:real=colour.models.eotf_sRGB(np.asarray(im.convert('RGB'),float)/255)
    h,w=real.shape[:2];step=max(1,round(max(logH.shape[:2])/1400))
    H=pixel_scaling(w/1600,h/1070)@np.array(prior['geometry'])@pixel_scaling(1/step,1/step)
    for name,center in [('near_ceiling',(676,92)),('middle_ceiling',(706,235)),('glass_reflection',(1437,405))]:
        cx,cy=np.array(center)*[w/1600,h/1070];x0,y0=int(round(cx))-256,int(round(cy))-256
        corners=np.array([[x0,y0],[x0+512,y0],[x0,y0+512],[x0+512,y0+512]])
        rawcorners=ProjectiveTransform(H).inverse(corners)
        xlo,ylo=np.maximum(np.floor(rawcorners.min(0)).astype(int)-3,0)
        xhi,yhi=np.minimum(np.ceil(rawcorners.max(0)).astype(int)+4,[logH.shape[1],logH.shape[0]])
        shift=np.array([[1,0,-x0],[0,1,-y0],[0,0,1]])
        local=np.array([[1,0,xlo],[0,1,ylo],[0,0,1]])
        warp=crop_warp(ProjectiveTransform(shift@H@local))
        def substate(state):
            return replace(state,**{k:v[ylo:yhi,xlo:xhi] for k,v in vars(state).items() if isinstance(v,np.ndarray) and v.shape[:2]==logH.shape[:2]})
        images={'real':real[y0:y0+512,x0:x0+512]}
        for label,state,shift_off in [('baseline',negative.state,False),('no_DIR',noDIR,False),('no_dye_shift',negative.state,True)]:
            saved=eng.v
            if shift_off:eng.v=replace(saved,dye_shift_nm=0,dye_shift_other_nm=0);eng._dye_cache={}
            try:
                subneg=replace(negative,state=substate(state))
                dn=subneg.measure(eng.scanner.normalized_density)
                images[label]=warp(eng.scanner.to_rgb(dn))
            finally:eng.v=saved;eng._dye_cache={}
        sheet=Image.new('RGB',(2048,550),(20,20,20));d=ImageDraw.Draw(sheet)
        for i,(key,label) in enumerate([('real','Real scan'),('baseline','Baseline exp_plain'),('no_DIR','Diagnostic: no layer inhibition'),('no_dye_shift','Diagnostic: no cross-process dye shift')]):
            arr=np.floor(srgb_encode(np.nan_to_num(images[key]))*255+.5).astype('uint8')
            sheet.paste(Image.fromarray(arr),(512*i,38));d.text((512*i+8,10),label,fill='white')
        sheet.save(out/(name+'_development_ablations.png'))
        np.savez_compressed(out/(name+'_development_ablations.npz'),**images)

if __name__=='__main__':main()
