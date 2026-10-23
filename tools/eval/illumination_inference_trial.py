"""Bounded illumination-inference checks; no target-photo colour fitting.

Compare existing random/nearest member choices and removal of global class
pruning. The latter is a numerical locality candidate, NOT a recovered light
measurement. Controlled CIE-camera spectra remain conditional on stock S.
"""
from pathlib import Path
from dataclasses import replace
import json,sys
import numpy as np
REPO=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(REPO),str(REPO/'tools/eval')]


def weighted_responses(engine,rgb,weights,members,minimum_global_share=0.):
    """Same response equations as Engine, without inferring new illumination."""
    from filmlab import film,lights
    H=np.zeros_like(rgb,dtype=float);src=np.zeros_like(H) if engine.halo is not None else None
    for k,c in enumerate(lights.classes()):
        wk=weights[...,k]
        if float(wk.mean())<minimum_global_share:continue
        spec=members[c.key][1]
        method='spectral' if c.key=='daylight' else ('linear_wp' if 'linear_white' in engine.p.corrections else 'linear')
        H+=wk[...,None]*film.expose(rgb,film.exposure_response(engine.e.stock,spec,engine.p.lens_filter,method))
        if src is not None:
            src+=wk[...,None]*film.halation_response(engine.e.stock,spec,engine.p.lens_filter,engine.v,engine._back_reflect,method)(rgb)
    return np.clip(H,1e-9,None),None if src is None else np.clip(src,0,None)


def unpruned_exposure(engine,rgb):
    from scipy.ndimage import zoom
    from filmlab import lights
    if engine.p.light_member!='random':raise ValueError('Trial freezes the original random member policy')
    oldH,oldS=engine._expose_mixed(rgb)
    w=engine.light_W;h,width=rgb.shape[:2]
    W=zoom(w,(h/w.shape[0],width/w.shape[1],1),order=1)[:h,:width]
    members=lights.pick_members(engine.v.frame_seed)
    replayH,replayS=weighted_responses(engine,rgb,W,members,.003)
    error=max(float(abs(replayH-oldH).max()),0 if oldS is None else float(abs(replayS-oldS).max()))
    if error>2e-9:raise ValueError('Weighted-equation reproduction failure '+str(error))
    H,S=weighted_responses(engine,rgb,W,members,0.)
    skipped=[{'class':c.key,'mean_weight':float(W[...,k].mean()),'max_local_weight':float(W[...,k].max())}
             for k,c in enumerate(lights.classes()) if float(W[...,k].mean())<.003]
    return H,S,{'baseline_equation_reproduction_max':error,'previously_pruned_classes':skipped}


def main():
    import colour,hashlib
    from filmlab import generation,lights,spectral,film
    from filmlab.engine import Engine,Params
    from filmlab.recipes import recipe
    from colour_foundation_audit import layer_error
    generation.use('next_sensor');out=Path('/workspace/filmlab-audit/illumination-inference-trial');out.mkdir(exist_ok=True)
    e=Engine(Params(**{**recipe('cinestill_800t'),'exposure_ev':0.,'timing':'standard','scan_neutral':'film',
                      'scan_color_strength':0.,'roll':None,'grain':0,'output_conditioning':None}))
    chart=colour.SDS_COLOURCHECKERS['ColorChecker N Ohta'];names=list(chart)
    R=np.array([v.copy().align(colour.SpectralShape(380,780,5)).values for v in chart.values()])
    S=10**e.e.stock.log_sensitivity;luma=np.array([.2126,.7152,.0722])
    report={'scope':__doc__,'predeclared_variants':['original','nearest_member','no_global_pruning'],'charts':[],'contexts':[],
            'limits':['Nearest is an existing estimator, not a measured scene SPD.',
                'Global pruning counterexample is a constructed algorithm test, not an estimate of real photograph error.',
                'No-pruning does not remove uncertainty in the inferred local light weights.',
                'RGB error versus conditional spectral integration does not validate 800T C-41 material physics.']}
    def variants(im):
        e.p=replace(e.p,light_member='random');old,_=e._expose_mixed(im)
        new,_,details=unpruned_exposure(e,im)
        e.p=replace(e.p,light_member='nearest');nearest,_=e._expose_mixed(im)
        e.p=replace(e.p,light_member='random')
        return {'original':old,'nearest_member':nearest,'no_global_pruning':new},details
    for light in ['D65','BB3200','LED-B3','FL11']:
        I=spectral.illuminant(light) if light in ['D65','BB3200'] else lights._sd(light)
        I=I/float(lights.camera_rgb(I)@luma)
        rgb=lights.camera_rgb(R*I);truth=(R*I)@S
        im=np.repeat(np.repeat(rgb.reshape(4,6,3),24,0),24,1)
        states,details=variants(im)
        values={key:np.array([h[24*y+12,24*x+12] for y in range(4) for x in range(6)]) for key,h in states.items()}
        report['charts'].append({'light':light,'variants':{key:layer_error(v,truth) for key,v in values.items()},'pruning':details})
    for light in ['D65','BB3200']:
        I=spectral.illuminant(light);I=I/float(lights.camera_rgb(I)@luma);center=lights.camera_rgb(.18*I)
        records={key:[] for key in report['predeclared_variants']}
        for bg in ['neutral_18','blue','green','orange']:
            refl=np.full(81,.18) if bg=='neutral_18' else R[names.index(bg)]
            im=np.broadcast_to(lights.camera_rgb(refl*I),(96,96,3)).copy();im[40:56,40:56]=center
            if not np.array_equal(im[48,48],center):raise ValueError('Fixed patch changed')
            states,_=variants(im)
            for key,h in states.items():records[key].append(h[46:50,46:50].mean((0,1)))
        report['contexts'].append({'light':light,'backgrounds':['neutral_18','blue','green','orange'],
            'variants':{key:{'center_H':np.asarray(v).tolist(),'center_layer_EV_span':np.ptp(np.log2(v),axis=0).tolist()} for key,v in records.items()}})
    # Frozen local assignment: same centre ray, only amount of unrelated image changes.
    members=lights.pick_members(e.v.frame_seed);rare=[]
    for size in [32,128]:
        im=np.full((size,size,3),.18);W=np.zeros((size,size,len(lights.classes())))
        W[...,0]=1.;W[:4,:4,0]=0.;W[:4,:4,4]=1.
        a,_=weighted_responses(e,im,W,members,.003);b,_=weighted_responses(e,im,W,members,0.)
        rare.append({'frame_size':size,'local_class_weight':1.,'global_class_weight':float(W[...,4].mean()),
                     'pruned_center_H':a[1,1].tolist(),'unpruned_center_H':b[1,1].tolist()})
    report['frozen_assignment_locality_counterexample']=rare
    report['no_pruning_locality_relative_error']=float(np.max(abs(np.array(rare[0]['unpruned_center_H'])/rare[1]['unpruned_center_H']-1)))
    if report['no_pruning_locality_relative_error']>1e-12:raise ValueError('Unpruned locality failed')
    snapshot=json.loads(Path('/workspace/filmlab-audit/lamp-edge-spectral-candidate/production_snapshot.json').read_text())['production_files_sha256']
    changed=[p for p,h in snapshot.items() if hashlib.sha256((REPO/p).read_bytes()).hexdigest()!=h]
    if changed:raise ValueError(changed)
    report['production']={'verified_files':len(snapshot),'changed':changed}
    (out/'controlled_results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
