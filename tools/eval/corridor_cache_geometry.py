"""Recover native local cache fields through the exact fixed bilinear operator.

No RAW reads, new geometry fit, physical fit or inverse-interpolation shortcut.
Unobserved native pixels stay NaN. A least-squares inverse is accepted only with
explicit reprojection, independent native-dye validation and coverage records.
"""
from dataclasses import dataclass
from pathlib import Path
import json,sys,zlib
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import lsmr
REPO=Path(__file__).resolve().parents[2];sys.path[:0]=[str(REPO),str(REPO/'tools/eval')]


@dataclass(frozen=True)
class CacheGeometry:
    source_shape: tuple
    matrix: np.ndarray
    source_bounds: tuple = ()
    output_shape: tuple = (512,512)

    def warp(self,values):
        from skimage.transform import warp,ProjectiveTransform
        return warp(values,ProjectiveTransform(self.matrix).inverse,output_shape=self.output_shape,order=1,cval=np.nan)

    @classmethod
    def from_cache(cls,cache):
        return cls(tuple(cache['source_shape'].astype(int)),np.array(cache['matrix'],float),
                   tuple(cache['source_bounds'].astype(int)),tuple(cache['output_shape'].astype(int)))


def bilinear_operator(source_shape,matrix,output_shape=(512,512)):
    """CSR operator on interior rows, and their flat output indices.

    Coordinates are integer pixel centres, (x,y), inverse homogeneous transform.
    Outside native-array rows are omitted because historical warp cval was NaN.
    """
    h,w=map(int,source_shape);oh,ow=map(int,output_shape)
    yy,xx=np.indices((oh,ow));points=np.stack([xx.ravel(),yy.ravel(),np.ones(oh*ow)])
    native=np.linalg.inv(np.asarray(matrix,float))@points;x=native[0]/native[2];y=native[1]/native[2]
    valid=np.isfinite(x)&np.isfinite(y)&(x>=0)&(y>=0)&(x<=w-1)&(y<=h-1)
    indices=np.flatnonzero(valid);x=x[valid];y=y[valid]
    x0=np.floor(x).astype(int);y0=np.floor(y).astype(int);fx=x-x0;fy=y-y0
    x1=np.minimum(x0+1,w-1);y1=np.minimum(y0+1,h-1)
    cols=np.column_stack([y0*w+x0,y0*w+x1,y1*w+x0,y1*w+x1])
    weights=np.column_stack([(1-fx)*(1-fy),fx*(1-fy),(1-fx)*fy,fx*fy])
    rows=np.repeat(np.arange(len(indices)),4)
    a=sparse.coo_matrix((weights.ravel(),(rows,cols.ravel())),shape=(len(indices),h*w)).tocsr();a.eliminate_zeros()
    return a,indices


def recover_native(a,observations,source_shape,*,atol=1e-12,btol=1e-12,maxiter=1200):
    """Sparse coordinate least squares; no regularizer, no physical parameters."""
    obs=np.asarray(observations,float)
    if obs.ndim==1:obs=obs[:,None]
    if len(obs)!=a.shape[0] or not np.isfinite(obs).all():raise ValueError('finite observations must match operator rows')
    active=np.asarray(a.getnnz(axis=0)).ravel()>0;reduced=a[:,active]
    output=np.full((a.shape[1],obs.shape[1]),np.nan);details=[]
    for c in range(obs.shape[1]):
        result=lsmr(reduced,obs[:,c],atol=atol,btol=btol,conlim=1e12,maxiter=maxiter)
        output[active,c]=result[0];residual=reduced@result[0]-obs[:,c]
        details.append({'stop_code':int(result[1]),'iterations':int(result[2]),'normr':float(result[3]),
                        'condition_estimate':float(result[6]),'max_reprojection_error':float(np.max(abs(residual))),
                        'rms_reprojection_error':float(np.sqrt(np.mean(residual**2)))})
    return output.reshape(*source_shape,obs.shape[1]),active.reshape(source_shape),details


def main():
    from filmlab import generation
    from filmlab.engine import Engine,Params
    from filmlab.recipes import recipe
    from filmlab.scanning.timing import PrintTiming
    from light_acceptance import pixel_scaling
    from skimage.transform import ProjectiveTransform
    from scipy.ndimage import binary_erosion
    import oracle_timing as ot
    root=Path('/workspace/filmlab-audit');out=root/'corridor-dye-response/cache';out.mkdir(parents=True,exist_ok=True)
    sid='R1-02-3';seed=zlib.crc32(sid.encode())%1000;generation.use('next_sensor')
    e=Engine(Params(**{**recipe('cinestill_800t'),'timing':'frame','roll':1,'frame':seed%36,'grain':0,'output_conditioning':None}))
    if not np.all(np.isneginf(e.e.fog_exposure)):raise ValueError('simple q_alloff subtraction invalid with additive fog')
    ds=e._dye_spectra('CD-4');pinv=np.linalg.pinv(ds.T)
    prior=json.loads((root/'epfl-highlights'/f'{sid}.json').read_text());oldreal=np.load(Path(ot.GEO)/f'geo_{sid}.npz')['real']
    # Dimensions come from the committed manifest, never a new RAW/TIFF decode.
    w,h=map(int,ot.MAN[sid]['film_dimensions'].split('x'));raw_shape=(1732,2304);step=round(max(raw_shape)/1400)
    H=pixel_scaling(w/oldreal.shape[1],h/oldreal.shape[0])@np.array(prior['geometry'])@pixel_scaling(1/step,1/step)
    t=prior['timing'];e.scanner.timing=PrintTiming(K=np.array(t['K']),s=np.array(t['s']),M=tuple(np.array(v) for v in t['M']),x_knee=t['x_knee'],x_white=t['x_white'])
    report={'scope':__doc__,'scene':sid,'fog_exposure':str(e.e.fog_exposure),
            'inverse_algorithm':{'solver':'LSMR','atol':1e-12,'btol':1e-12,'maxiter':1200,'regularization':'none','acceptance':'validated interior only, not global inverse'},'regions':[]}
    for name,oldname,center in [('ceiling_lamp','near_ceiling',(676,92)),('middle_ceiling','middle_ceiling',(706,235)),('glass_reflection','glass_reflection',(1437,405))]:
        b=np.load(root/'brightness-local'/f'{sid}_{name}.npz');m=np.load(root/'halo-mechanisms'/sid/f'{name}.npz');c=np.load(root/'corridor-lights'/f'{oldname}.npz')
        x0,y0=np.round(np.array(center)*[w/oldreal.shape[1],h/oldreal.shape[0]]).astype(int)-256
        corners=ProjectiveTransform(H).inverse(np.array([[x0,y0],[x0+512,y0],[x0,y0+512],[x0+512,y0+512]]))
        xl,yl=np.maximum(np.floor(corners.min(0)).astype(int)-3,0);xh,yh=np.minimum(np.ceil(corners.max(0)).astype(int)+4,[raw_shape[1],raw_shape[0]])
        matrix=np.array([[1,0,-x0],[0,1,-y0],[0,0,1]])@H@np.array([[1,0,xl],[0,1,yl],[0,0,1]])
        shape=b['native_base_spectral_density'].shape[:2]
        if shape!=(yh-yl,xh-xl):raise ValueError('native crop shape mismatch')
        geo=CacheGeometry(shape,matrix,(int(xl),int(yl),int(xh),int(yh)))
        a,rows=bilinear_operator(shape,matrix)
        # All operators are linear. Subtract on the saved output grid before inversion.
        fields={'q_base':m['reproduced_logH'],'q_minus_red':m['reproduced_logH']-b['red_EV'][...,None]*np.array([np.log10(2),0,0]),
                'q_alloff':m['reproduced_logH']-c['added_exposure_EV']*np.log10(2),
                'dye_base_inverse':m['reproduced_dye'],'dye_minus_red_inverse':b['minus_red_dye'],
                'dye_alloff':m['reproduced_dye']-c['dye_change']}
        finite=np.ones(len(rows),bool)
        for v in fields.values():finite&=np.isfinite(v.reshape(-1,3)[rows]).all(-1)
        rows=rows[finite];a=a[finite]
        observed=np.column_stack([v.reshape(-1,3)[rows] for v in fields.values()])
        print(name,'solving',a.shape,flush=True)
        recovered,active,solver=recover_native(a,observed,shape)
        arrays={key:recovered[...,3*i:3*i+3] for i,key in enumerate(fields)}
        dbase=(b['native_base_spectral_density'].astype(float)-e.e.stock.base_density)@pinv
        dminus=(b['native_minus_red_spectral_density'].astype(float)-e.e.stock.base_density)@pinv
        # Independent D81->dye reconstructs native quantities without inverse warp.
        errors={'base':np.max(abs(arrays['dye_base_inverse']-dbase),axis=-1),
                'minus_red':np.max(abs(arrays['dye_minus_red_inverse']-dminus),axis=-1)}
        trusted=binary_erosion(active,iterations=2)&(errors['base']<1e-5)&(errors['minus_red']<1e-5)
        native_shape=shape
        arrays.update(native_base_dye=dbase,native_minus_red_dye=dminus,native_alloff_dye=arrays['dye_alloff'],local_transform=matrix,
                      dye_base=dbase,dye_minus_red=dminus,native_valid=active,native_trusted=trusted,
                      source_bounds=np.array(geo.source_bounds),source_shape=np.array(shape),matrix=matrix,output_shape=np.array(geo.output_shape),
                      measured=b['measured'],distance_um=b['distance_um'],real_bin=b['real_bin'],real=b['real'],
                      base_dye_native_validation_error=errors['base'],minus_red_dye_native_validation_error=errors['minus_red'])
        # Check sparse forward operator against the real historical warp, using D81-derived dye.
        forward=geo.warp(dbase);forward_sparse=a@dbase.reshape(-1,3)
        forward_error=float(np.max(abs(forward.reshape(-1,3)[rows]-forward_sparse)))
        independent={}
        for key,native,observed_field in [('base',dbase,m['reproduced_dye']),('minus_red',dminus,b['minus_red_dye'])]:
            residual=a@native.reshape(-1,3)-observed_field.reshape(-1,3)[rows]
            err=errors[key][active]
            independent[key]={'native_abs_error_p50_p95_max':np.percentile(err,[50,95,100]).tolist(),
                              'known_native_forward_max_error':float(np.max(abs(residual))),
                              'known_native_forward_rms_error':float(np.sqrt(np.mean(residual**2)))}
        target=(b['measured']>.5)&(b['distance_um']>0)&(b['distance_um']<=300)&np.isin(b['real_bin'],[2,3])
        target_valid=target.ravel()[rows]
        required=np.asarray(a[target_valid].getnnz(axis=0)).ravel()>0
        trust_rows=np.asarray(a@trusted.ravel().astype(float)).ravel()>=1-1e-10
        output_trusted=np.zeros(512*512,bool);output_trusted[rows]=trust_rows
        arrays['output_trusted']=output_trusted.reshape(512,512)
        target_reprojection={}
        for i,key in enumerate(fields):
            residual=a@np.nan_to_num(arrays[key]).reshape(-1,3)-observed[:,3*i:3*i+3]
            target_reprojection[key]={'max_error':float(np.max(abs(residual[target_valid]))),'rms_error':float(np.sqrt(np.mean(residual[target_valid]**2)))}
        coverage={'active_native_pixels':int(active.sum()),'total_native_pixels':int(active.size),'trusted_native_pixels':int(trusted.sum()),
                  'target_receiving_pixels':int(target.sum()),'target_rows_in_operator':int(target_valid.sum()),
                  'target_required_native_pixels':int(required.sum()),'target_required_native_trusted':int((required&trusted.ravel()).sum()),
                  'target_rows_fully_trusted':int((target_valid&trust_rows).sum())}
        item={'name':name,'source_bounds':list(geo.source_bounds),'source_shape':list(shape),'matrix':matrix.tolist(),
              'source_bounds_order':'x_min,y_min,x_max_exclusive,y_max_exclusive',
              'warp_policy':'native-local to output-local integer pixel centres; skimage warp order=1,cval=NaN; no inverse-interpolation approximation',
              'coverage':coverage,'target_reprojection':target_reprojection,'solver_channels':{key:solver[3*i:3*i+3] for i,key in enumerate(fields)},
              'sparse_vs_skimage_forward_max_error':forward_error,'independent_native_dye_validation':independent,
              'literal_float32_base_minus_alloff_difference_max':float(np.nanmax(abs(m['reproduced_dye']-b['base_dye']))),
              'limits':['unobserved native pixels remain NaN','trusted mask erodes two native pixels and requires independent dye error below 1e-5',
                        'native dye from float32 D81 has finite spectral-quantization error','q_alloff subtraction valid here because additive fog is zero',
                        'spatial chemistry outside recovered crop not known; use trusted support and do not invent boundary context']}
        if forward_error>1e-9:raise ValueError('sparse operator disagrees with historical warp')
        # Only the verified interior is publishable as recovered native data.
        for key in fields:arrays[key][~trusted]=np.nan
        arrays['native_alloff_dye']=arrays['dye_alloff']
        trans=10**(-(arrays['native_alloff_dye']@ds.T+e.e.stock.base_density))
        reproduced=geo.warp(e.scanner.to_rgb(e.scanner.normalized_density(trans)))
        mask=target&arrays['output_trusted'];difference=(reproduced-c['none'])[mask]
        item['independent_alloff_RGB_reproduction']={'pixels':int(mask.sum()),'max_linear_RGB_error':float(np.max(abs(difference))),'rms_linear_RGB_error':float(np.sqrt(np.mean(difference**2)))}
        item['accepted_scope']='verified native interior/target only; full-frame global inverse not accepted'
        item['recovered_fields_outside_trusted']='NaN'
        temporary=out/f'{name}.temporary.npz'
        np.savez_compressed(temporary,**arrays);temporary.replace(out/f'{name}.npz')
        (out/f'{name}.json').write_text(json.dumps(item,indent=2));report['regions'].append(item)
        print(name,coverage,independent,flush=True)
    (out/'results.json').write_text(json.dumps(report,indent=2));print('all saved',flush=True)


if __name__=='__main__':main()
