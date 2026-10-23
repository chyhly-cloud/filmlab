"""Fixed matrix versus exact virtual-paper spectral integration; no fitting.

This tests the internal printing-density approximation on the SAME negative.
Neither route is a measured Frontier file response. Fixed gain, operator timing,
shoulder, digital-paper characteristic curve and spectral colour remain unchanged.
"""
from pathlib import Path
from dataclasses import replace
import json,sys,zlib
import numpy as np
REPO=Path(__file__).resolve().parents[2];sys.path[:0]=[str(REPO),str(REPO/'tools/eval')]


def exact_printing_density(transmittance,base_transmittance,paper_response):
    """Broadband density -log10(integral T*E*S / integral Tbase*E*S)."""
    t=np.asarray(transmittance,float);t0=np.asarray(base_transmittance,float);s=np.asarray(paper_response,float)
    if t.shape[-1]!=s.shape[0] or t0.shape!=(s.shape[0],) or s.ndim!=2:
        raise ValueError('wavelength dimensions must match')
    if not all(np.isfinite(v).all() for v in (t,t0,s)) or np.any(t<0) or np.any(t0<0) or np.any(s<0):
        raise ValueError('finite nonnegative spectral inputs required')
    denominator=t0@s;numerator=t@s
    if np.any(denominator<=0) or np.any(numerator<=0):raise ValueError('strictly positive integrated response required')
    return -np.log10(numerator/denominator)


def neutral_preserving_matrix(design,target):
    """Least squares subject to each column summing to one; no photo fitting."""
    x=np.asarray(design,float);y=np.asarray(target,float)
    basis=np.array([[1.,0.],[0.,1.],[-1.,-1.]])
    origin=np.ones((3,y.shape[1]))/3
    coefficient=np.linalg.lstsq(x@basis,y-x@origin,rcond=None)[0]
    return origin+basis@coefficient


def main():
    from PIL import Image,ImageDraw,ImageFont
    from skimage.transform import ProjectiveTransform
    from filmlab import generation,spectral
    from filmlab.engine import Engine,Params
    from filmlab.recipes import recipe
    from filmlab.scanning.timing import PrintTiming
    from filmlab.scanning.optics import measure
    from filmlab.spectral import srgb_encode
    from brightness_response_audit import lab,LUMA
    from light_acceptance import pixel_scaling,crop_warp
    import oracle_timing as ot
    import colour
    root=Path('/workspace/filmlab-audit');out=root/'printing-density';out.mkdir(exist_ok=True)
    sid='R1-02-3';seed=zlib.crc32(sid.encode())%1000;generation.use('next_sensor')
    e=Engine(Params(**{**recipe('cinestill_800t'),'timing':'frame','roll':1,'frame':seed%36,'grain':0,'output_conditioning':None}))
    sc=e.scanner;T0=e._transmittance_base();paper=sc.digital_paper.paper.stock
    S=spectral.illuminant('TH-KG3')[:,None]*10**paper.log_sensitivity
    prior=json.loads((root/'epfl-highlights'/f'{sid}.json').read_text());t=prior['timing']
    fixed=PrintTiming(K=np.array(t['K']),s=np.array(t['s']),M=tuple(np.array(v) for v in t['M']),x_knee=t['x_knee'],x_white=t['x_white'])
    report={'scope':__doc__,'scene':sid,'seed':seed,'generation':'next_sensor','timing':t,
            'source_hashes':{k:prior[k] for k in ['raw_sha256','film_sha256']},
            'matrix':sc.P.tolist(),'gain':sc.gain.tolist(),'controlled':[],'regions':[]}
    # Same seed and 2000 physical proxy samples used by printing_matrix, with one
    # exact neutral-absorber constraint; no real photographs enter this fit.
    from filmlab.materials import emulsion
    profile=emulsion(sc.m.default_profile).stock
    rng=np.random.default_rng(0);amounts=rng.uniform(0,2.5,(2000,3))
    sample_T=10**(-(amounts@profile.dye_density.T+profile.base_density))
    sample_T0=10**(-profile.base_density)
    design=measure(sample_T,sc.m)-measure(sample_T0[None,:],sc.m)
    target=exact_printing_density(sample_T,sample_T0,S)
    constrained=neutral_preserving_matrix(design,target)
    report['neutral_constrained_P']=constrained.tolist()
    report['training_density_RMSE']={
        'existing':float(np.sqrt(np.mean((design@sc.P-target)**2))),
        'neutral_constrained':float(np.sqrt(np.mean((design@constrained-target)**2)))}
    # Small known-negative controls, not a fit to an observed scene or scanner LUT.
    dye=e._dye_spectra('CD-4')
    cases=[('base',T0)]
    for d in [.1,.5,1.,2.]:cases.append((f'neutral_absorber_{d}',T0*10**(-d)))
    for amounts in [[.2,0,0],[1,0,0],[2,0,0],[0,1,0],[0,0,1],[1,1,1],[.2,1.5,.4]]:
        cases.append(('dye_'+str(amounts),T0*10**(-dye@np.array(amounts))))
    for name,T in cases:
        direct=exact_printing_density(T,T0,S);matrix=(measure(T,sc.m)-sc.base)@sc.P
        report['controlled'].append({'name':name,'matrix_printing_D':matrix.tolist(),'exact_printing_D':direct.tolist(),'error_D':(matrix-direct).tolist(),
            'neutral_constrained_printing_D':((measure(T,sc.m)-sc.base)@constrained).tolist()})
    oldreal=np.load(Path(ot.GEO)/('geo_'+sid+'.npz'))['real'];oh,ow=oldreal.shape[:2]
    _,scanpath,_=ot.sources(sid)
    with Image.open(scanpath) as im:w,h=im.size
    # Existing native half-size is 2304x1732; same step=2 geometry as stored arrays.
    raw_shape=(1732,2304);step=round(max(raw_shape)/1400)
    H=pixel_scaling(w/ow,h/oh)@np.array(prior['geometry'])@pixel_scaling(1/step,1/step)
    for name,center in [('ceiling_lamp',(676,92)),('middle_ceiling',(706,235)),('glass_reflection',(1437,405))]:
        cache=np.load(root/'brightness-local'/(sid+'_'+name+'.npz'))
        D=cache['native_base_spectral_density'];T=10**(-D.astype(float));dn_matrix=sc.normalized_density(T)
        dn_constrained=((measure(T,sc.m)-sc.base)@constrained)*sc.gain
        dn_exact=exact_printing_density(T,T0,S)*sc.gain
        Tminus=10**(-cache['native_minus_red_spectral_density'].astype(float))
        minus_matrix=sc.normalized_density(Tminus);minus_exact=exact_printing_density(Tminus,T0,S)*sc.gain
        minus_constrained=((measure(Tminus,sc.m)-sc.base)@constrained)*sc.gain
        x0,y0=np.round(np.array(center)*[w/ow,h/oh]).astype(int)-256
        corners=ProjectiveTransform(H).inverse(np.array([[x0,y0],[x0+512,y0],[x0,y0+512],[x0+512,y0+512]]))
        xl,yl=np.maximum(np.floor(corners.min(0)).astype(int)-3,0);xh,yh=np.minimum(np.ceil(corners.max(0)).astype(int)+4,[raw_shape[1],raw_shape[0]])
        if D.shape[:2]!=(yh-yl,xh-xl):raise ValueError('saved native crop geometry mismatch')
        matrix=np.array([[1,0,-x0],[0,1,-y0],[0,0,1]])@H@np.array([[1,0,xl],[0,1,yl],[0,0,1]])
        wp=crop_warp(ProjectiveTransform(matrix));images={'real':cache['real']}
        for shoulder,timing in [('exp',fixed),('paper',replace(fixed,x_knee=None,x_white=None))]:
            sc.timing=timing
            for key,dn in [('matrix',dn_matrix),('constrained',dn_constrained),('exact',dn_exact),('minus_matrix',minus_matrix),('minus_constrained',minus_constrained),('minus_exact',minus_exact)]:images[key+'_'+shoulder]=wp(sc.to_rgb(dn))
        error=float(np.nanmax(abs(images['matrix_exp']-cache['base_exp'])))
        if error>2e-6:raise ValueError('baseline not reproduced: '+str(error))
        labs={k:lab(v) for k,v in images.items()};dist=cache['distance_um'];valid=cache['measured']>.5;rows=[]
        for lo,hi in [(0,300),(300,800)]:
            for bi in [1,2,3,4]:
                mask=valid&(dist>lo)&(dist<=hi)&(cache['real_bin']==bi)
                for key in images:mask&=np.isfinite(images[key]).all(-1)
                if mask.sum()<30:continue
                row={'distance':[lo,hi],'real_bin':bi,'pixels':int(mask.sum()),'outputs':{}}
                for key in ['matrix_exp','constrained_exp','exact_exp','matrix_paper','constrained_paper','exact_paper']:
                    gap=labs[key]-labs['real']
                    row['outputs'][key]={'median_dLab':np.median(gap[mask],axis=0).tolist(),
                        'median_deltaE00':float(np.median(colour.delta_E(labs[key][mask],labs['real'][mask],method='CIE 2000'))),
                        'median_Y':float(np.median((images[key]@LUMA)[mask])),
                        'model_red_chroma_step':float(np.median(np.linalg.norm((labs[key]-labs['minus_'+key])[mask,1:],axis=-1))),
                        'model_red_delta_ab':np.median((labs[key]-labs['minus_'+key])[mask,1:],axis=0).tolist()}
                rows.append(row)
        # Matrix approximation errors before any subsequent tone/colour conversion.
        delta=dn_exact-dn_matrix
        report['regions'].append({'name':name,'reproduction_error':error,'native_Dn_exact_minus_matrix_median':np.median(delta.reshape(-1,3),axis=0).tolist(),
                                  'native_Dn_exact_minus_matrix_p05_p95':np.percentile(delta.reshape(-1,3),[5,95],axis=0).tolist(),'statistics':rows})
        sheet=Image.new('RGB',(3584,562),(20,20,20));draw=ImageDraw.Draw(sheet);font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16)
        for i,(key,label) in enumerate([('real','Real scan'),('matrix_exp','Existing matrix + exp'),('constrained_exp','Neutral constraint + exp'),('exact_exp','Exact printing integral + exp'),('matrix_paper','Existing matrix, paper_only'),('constrained_paper','Neutral constraint, paper_only'),('exact_paper','Exact integral, paper_only')]):
            arr=np.uint8(np.clip(srgb_encode(np.nan_to_num(images[key]))*255+.5,0,255));sheet.paste(Image.fromarray(arr),(512*i,50));draw.text((512*i+8,8),label,font=font,fill='white')
            draw.text((512*i+8,29),sid+'_'+name,font=font,fill='white')
        sheet.save(out/(name+'_comparison.png'))
        np.savez_compressed(out/(name+'.npz'),**images,native_Dn_matrix=dn_matrix,native_Dn_constrained=dn_constrained,native_Dn_exact=dn_exact,measured=valid,distance_um=dist,real_bin=cache['real_bin'])
    report['limits']=['exact only within the existing virtual optical-print proxy, not measured Frontier device response',
                      'source core radiance/SPD unknown; actual film optics/development unchanged',
                      'gain from old profile and operator timing fixed; no readjustment for exact integral',
                      'different colour/brightness does not establish a valid device correction']
    (out/'results.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':main()
