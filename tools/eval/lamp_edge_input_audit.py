"""Audit measured and censored inputs on explicitly reviewed lamp geometry.

No material, development, scanner, transport or fit parameters are changed.
The geometric strip is a conservative visible-luminaire hypothesis, not proof
that every mixed boundary pixel is a pure emitter or a measured source SPD.
"""
from pathlib import Path
import argparse,json,sys,zlib
from unittest.mock import patch
import numpy as np
from scipy import ndimage

REPO=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(REPO),str(REPO/'tools/eval')]
ROOT=Path('/workspace/filmlab-audit/inverse-source-radiance')
DEFAULT_OUT=Path('/workspace/filmlab-audit/lamp-edge-input-audit')
# Local native half-size Bayer XY coordinates, chosen by visual inspection of
# the actual measured RAW and the registered scan's common lamp structure.
# The left bright arm / right dark triangle are not automatically emitters.
GEOMETRY={
 'ceiling_lamp':{'strip_xy':[(94,94),(100,94),(112,111),(111,115),(106,114),(91,99)],
                 'housing_xy':[(84,91),(122,93),(112,116),(90,101)]},
 'middle_ceiling':{'strip_xy':[(99,99),(103,99),(115,112),(113,116),(109,113),(97,104)],
                   'housing_xy':[(94,96),(122,100),(114,117),(96,106)]},
}


def polygon_mask(shape,vertices):
    from skimage.draw import polygon
    p=np.asarray(vertices);rr,cc=polygon(p[:,1],p[:,0],shape);mask=np.zeros(shape,bool);mask[rr,cc]=True
    return mask


def collect(out):
    """One physical RAW read; borrow it for existing measurement/decoder APIs."""
    import rawpy
    from filmlab import rawio
    from inverse_source_radiance import engine_for,strict_full_cfa
    from linear_raw_measurements import bayer_measurements
    import oracle_timing as ot
    out.mkdir(parents=True,exist_ok=True);prep=json.loads((ROOT/'prepared.json').read_text())
    e,meta=engine_for();src,scan,bias=ot.sources('R1-02-3');captured={};raw=rawpy.imread(str(src))
    class Borrow:
        def __enter__(self):return raw
        def __exit__(self,*args):return False
    try:
        with patch.object(rawpy,'imread',lambda *a,**kw:Borrow()):
            camera,sat,cammeta=bayer_measurements(src);strict,threshold=strict_full_cfa(raw)
            old=rawio.reconstruct_highlights
            def recon(a,*args,**kw):
                captured['pre_prior']=a.copy();return old(a,*args,**kw)
            with patch.object(rawio,'reconstruct_highlights',recon):rgb=rawio.load_raw(src,max_size=3000)
    finally:raw.close()
    metered,note=rawio.film_metering(rgb,rng=np.random.default_rng(zlib.crc32(b'R1-02-3')%1000),bias_ev=bias)
    scale=float(np.sum(metered*rgb)/np.sum(rgb*rgb))
    direct,source=e._expose_mixed(metered*2**e.p.exposure_ev)
    labels,_=ndimage.label(sat.any(-1));probe=prep['probe'];inp=np.array(probe['probe_inputs']);outputs=np.array(probe['probe_outputs'])
    matrix=np.stack([(outputs[c+1]-outputs[0])/(inp[c+1,c]-inp[0,c]) for c in range(3)])
    cam_to_rgb=matrix@rawio.XYZ_TO_SRGB.T
    for r in prep['regions']:
        name=r['name'];xl,yl,xh,yh=r['source_bounds'];sl=(slice(yl,yh),slice(xl,xh));a=np.load(ROOT/(name+'_source.npz'));support=a['support']
        component=int(np.bincount(labels[sl][support]).argmax())
        arr={key:values[sl] for key,values in [('camera',camera),('sat',sat),('strict',strict),('rgb',rgb),('pre_prior',captured['pre_prior']),('direct',direct),('return_source',source)]}
        arr.update(component=labels[sl]==component,support=support,matrix=matrix,cam_to_rgb=cam_to_rgb,source_bounds=np.array(r['source_bounds']),scale=scale)
        np.savez_compressed(out/(name+'_input.npz'),**arr)
    (out/'capture.json').write_text(json.dumps({'raw':str(src),'scan':str(scan),'scale':scale,'metering':note,'thresholds':threshold.tolist(),'camera':cammeta,'policy':'Single physical RAW decode, existing legacy metadata and reconstruction; no development or scan'},indent=2))


def analyze(out):
    from PIL import Image,ImageDraw,ImageFont
    from filmlab.spectral import srgb_encode
    from filmlab import film
    from inverse_source_radiance import engine_for
    engine,_=engine_for();rho=film.bounce_retention(engine.e.stock,engine.v)
    results={};luma=np.array([.2126,.7152,.0722])
    for name,geometry in GEOMETRY.items():
        path=out/(name+'_input.npz');a={k:v for k,v in np.load(path).items()};shape=a['camera'].shape[:2]
        strip=polygon_mask(shape,geometry['strip_xy']);housing=polygon_mask(shape,geometry['housing_xy'])|strip
        sat=a['sat'];component=a['component'];support=a['support'];edge=strip&~support
        source_cache=np.load(ROOT/(name+'_source.npz'));prior_fit=np.load(ROOT/(name+'_fit.npz'))
        removed=film.emulsion_scatter(a['direct']*support[...,None],engine.px_um)
        removed+=film.halation(np.zeros_like(a['direct']),a['return_source']*support[...,None],engine.px_um,engine.v,rho)
        replay_error=float(abs(removed-source_cache['removed']).max())
        if replay_error>1e-10:raise ValueError('Cached strict source contribution differs from preceding inverse input')
        masks={'strict_core':support,'RGB_all_nonstrict':component&sat.all(-1)&~support,
               'partial_component':component&~sat.all(-1),'strip_partial_edge':edge&sat.any(-1),
               'strip_unflagged_edge':edge&~sat.any(-1),'component_outside_strip':component&~strip,
               'ambiguous_housing':housing&~strip}
        # Main-line consumers must use the explicit candidate mask, not the
        # entire connected saturation component or the display-bright region.
        a.update(geometry_mask=strip,body_mask=strip,housing_mask=housing,edge_mask=edge,
                 candidate_edge_mask=edge,preprior_rgb=a['pre_prior'],postprior_rgb=a['rgb'],halo_src=a['return_source'])
        a.update(saturation=sat,strict_support=support,component_mask=component,
                 old_direct=a['direct'],old_halo_source=a['return_source'])
        np.savez_compressed(path,**a)
        # Stable small interface for follow-on constrained-source diagnostics.
        np.savez_compressed(out/(name+'.npz'),**{key:a[key] for key in ['camera','saturation','strict_support','component_mask','body_mask','housing_mask','edge_mask','old_direct','old_halo_source','preprior_rgb','postprior_rgb','source_bounds','cam_to_rgb','scale']})
        inverse=np.linalg.inv(a['cam_to_rgb']);before=a['pre_prior']@inverse;after=a['rgb']@inverse;delta=after-before
        input_y=a['rgb']@luma;strip_budget=float(input_y[strip].sum());rows={}
        for label,mask in masks.items():
            measured=mask[...,None]&~sat;abs_change=abs(delta[measured]);relative=abs_change/np.maximum(abs(a['camera'][measured]),.01)
            n=int(mask.sum());row={'cells':n,'measured_channels_RGB':[(int((mask&~sat[...,c]).sum())) for c in range(3)],
              'all_measured_cells':int((mask&~sat.any(-1)).sum()),'old_input_Y_sum':float(input_y[mask].sum()),
              'old_input_Y_fraction_of_strip':float(input_y[mask].sum()/strip_budget),
              'old_direct_sum_layers':a['direct'][mask].sum(0).tolist(),'old_return_source_sum_layers':a['return_source'][mask].sum(0).tolist(),
              'prior_measured_channel_abs_change_max':float(abs_change.max()) if len(abs_change) else None,
              'prior_measured_channel_relative_change_p50_p95':np.quantile(relative,[.5,.95]).tolist() if len(relative) else None,
              'prior_measured_channel_fraction_changed_gt1e-5':float(np.mean(abs_change>1e-5)) if len(abs_change) else None}
            row['prior_measured_channel_relative_change_by_RGB_p50_p95']=[]
            for c in range(3):
                use=mask&~sat[...,c]
                changes=abs(delta[...,c][use])/np.maximum(abs(a['camera'][...,c][use]),.01)
                row['prior_measured_channel_relative_change_by_RGB_p50_p95'].append(np.quantile(changes,[.5,.95]).tolist() if len(changes) else None)
            for field in ['direct','return_source']:
                values=a[field][mask];norm=values/np.maximum(values.sum(-1,keepdims=True),1e-30)
                row[field+'_layer_fraction_q05_median_q95']=np.quantile(norm,[.05,.5,.95],axis=0).tolist() if n else None
            rows[label]=row
        strict_out=int((support&~strip).sum())
        component_y=float(input_y[component].sum());component_edge=component&~support
        component_totals={key:a[key][component].sum(0) for key in ['direct','return_source']}
        component_edge_summary={'cells':int(component_edge.sum()),'old_input_Y_fraction_of_component':float(input_y[component_edge].sum()/component_y),
          'old_direct_layer_fraction_of_component':(a['direct'][component_edge].sum(0)/component_totals['direct']).tolist(),
          'old_return_source_layer_fraction_of_component':(a['return_source'][component_edge].sum(0)/component_totals['return_source']).tolist()}
        fixed_core={};profile_sum=source_cache['profile'][support].sum()
        for field,response in [('direct','direct_response'),('return_source','return_response')]:
            core=profile_sum*(prior_fit['coefficients']@source_cache[response]);edge_sum=a[field][component_edge].sum(0)
            fixed_core[field+'_edge_fraction_after_prior_strictfit']=(edge_sum/(edge_sum+core)).tolist()
        results[name]={'native_shape':list(shape),'source_bounds':a['source_bounds'].tolist(),'geometry':geometry,
                      'strict_support_outside_reviewed_strip':strict_out,'component_cells':int(component.sum()),
                      'strip_cells':int(strip.sum()),'candidate_edge_cells':int(edge.sum()),'rows':rows,
                      'component_edge_summary':component_edge_summary,
                      'strict_source_replay_max_error':replay_error,'remaining_edge_budget_with_prior_strict_core_fit':fixed_core,
                      'strip_legacy_input_Y_sum':strip_budget,'mask_policy':'Manual visible luminous stripe only. The housing and left horizontal bright arm remain ambiguous; no brightness threshold identifies emitters. Boundary cells include lens/demosaic mixing, so this is an explicitly conditional source mask.'}
        # Native coordinate overlay and registered-scan overlay, no image edit
        # or generated replacement. Display transforms never alter cache data.
        camrgb=np.maximum(a['camera']@a['cam_to_rgb'],0);display=np.clip(np.arcsinh(camrgb*5)/np.arcsinh(5),0,1)
        colors={'strict_core':(255,50,70),'strip_partial_edge':(255,205,30),'strip_unflagged_edge':(50,255,70),'component_outside_strip':(255,70,255),'ambiguous_housing':(30,155,255)}
        overlay=display.copy()
        for key,color in colors.items():
            m=masks[key];boundary=m&~ndimage.binary_erosion(m);overlay[boundary]=np.array(color)/255
        native_box=(75,80,135,130);left=Image.fromarray(np.uint8(np.rint(display*255))).crop(native_box).resize((720,600),Image.Resampling.NEAREST)
        right=Image.fromarray(np.uint8(np.rint(overlay*255))).crop(native_box).resize((720,600),Image.Resampling.NEAREST)
        sheet=Image.new('RGB',(1440,660),(16,16,16));sheet.paste(left,(0,60));sheet.paste(right,(720,60));draw=ImageDraw.Draw(sheet)
        draw.text((10,8),name+' measured RAW geometry: asinh display only',fill='white');draw.text((10,28),'red strict core | yellow partial strip | green measured strip | magenta component outside strip | blue ambiguous housing',fill='white')
        sheet.save(out/(name+'_masks.jpg'),quality=98)
        real=np.load(ROOT/(name+'_source.npz'))['real'];rgb=np.rint(np.clip(srgb_encode(np.nan_to_num(real)),0,1)*255).astype('uint8')
        from skimage.transform import ProjectiveTransform
        source_matrix=np.load(ROOT/(name+'_source.npz'))['matrix'];tf=ProjectiveTransform(source_matrix)
        im=Image.fromarray(rgb);d=ImageDraw.Draw(im)
        for key,color in [('housing_xy',(30,155,255)),('strip_xy',(50,255,70))]:
            p=tf(np.asarray(geometry[key]));d.line([tuple(v) for v in np.vstack([p,p[:1]])],fill=color,width=2)
        im.save(out/(name+'_scan_mask.jpg'),quality=98)
        print(name,json.dumps(results[name],ensure_ascii=False),flush=True)
    report={'regions':results,'units':'Camera values black/metadata-white normalized; RGB-to-camera inversion is the fixed digital-decoder proxy. direct/return_source are model sensitivity-calibrated layer exposure, not physical energy.',
        'limits':['The prior-only difference isolates the random highlight reconstruction after denoising/XYZ decoding; it does not attribute decoder clipping or denoising to the random prior.','No measured camera spectral sensitivity, emitter SPD, lens PSF or absolute radiance was obtained.','Unsaturated channels constrain only a local camera average, not an independently identified pure emitting material.','Mask and connected-component classes overlap; do not sum all table rows.','No film/scanner parameter or source amplitude was selected from real scan colours.']}
    (out/'results.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    return report


def decoder_ceiling(out,*,reuse=False):
    """Record the EXACT load_raw XYZ16 call before denoising/prior.

    ``reuse`` analyses the already saved capture without reopening RAW.
    The camera matrix came from digital constant-image probes, not scene fits.
    """
    from filmlab import generation,rawio
    target=out/'decoder-ceiling';target.mkdir(parents=True,exist_ok=True)
    if not reuse:
        import rawpy
        import oracle_timing as ot
        generation.use('next_sensor');src,_,_=ot.sources('R1-02-3');raw=rawpy.imread(str(src));captured={}
        class Borrow:
            def __enter__(self):return self
            def __exit__(self,*args):return False
            def __getattr__(self,k):return getattr(raw,k)
            def postprocess(self,**kw):
                answer=raw.postprocess(**kw);captured.update(xyz16=answer.copy(),kwargs=kw);return answer
        try:
            with patch.object(rawpy,'imread',lambda *a,**kw:Borrow()):rgb=rawio.load_raw(src,max_size=3000,highlights='clip',denoise_raw=False)
        finally:raw.close()
        xyz=captured['xyz16'];report={'raw':str(src),'kwargs':{k:str(v) if k=='output_color' else v for k,v in captured['kwargs'].items()},'native_shape':list(xyz.shape),'regions':{}}
        for name in GEOMETRY:
            a=np.load(out/(name+'.npz'));xl,yl,xh,yh=a['source_bounds'];sl=(slice(yl,yh),slice(xl,xh))
            matrix_xyz=np.linalg.solve(rawio.XYZ_TO_SRGB,a['cam_to_rgb'].T).T
            predicted=a['camera']@matrix_xyz;observed=xyz[sl];coded=observed.astype(np.float32)/65535.;xyzrgb=coded@rawio.XYZ_TO_SRGB.T
            closure=float(abs(xyzrgb-rgb[sl]).max())
            if closure:raise ValueError('XYZ16 replay must be bitwise equal to load_raw without denoise/prior')
            np.savez_compressed(target/(name+'.npz'),xyz16=observed,xyz16_rgb=xyzrgb,float_measured_rgb=a['camera']@a['cam_to_rgb'],float_measured_xyz=predicted,
                xyz_clip_mask=(observed==65535).any(-1),xyz_clip_channels=observed==65535,fully_measured_body_mask=a['body_mask']&~a['saturation'].any(-1),source_bounds=a['source_bounds'],matrix_xyz=matrix_xyz)
            report['regions'][name]={'load_raw_no_denoise_no_prior_replay_max':closure}
    else:report=json.loads((target/'results.json').read_text())
    for name in GEOMETRY:
        a=np.load(out/(name+'.npz'));b=np.load(target/(name+'.npz'));body=b['fully_measured_body_mask'];clip=b['xyz_clip_channels'];bodyclip=body&b['xyz_clip_mask']
        coded=b['xyz16'].astype(np.float32)/65535.;predicted=b['float_measured_xyz'];delta=coded-predicted;clipdiff=coded-np.clip(predicted,0,1)
        # This inversion is diagnostic only. Excluding final XYZ=65535 avoids
        # confusing output clipping artifacts with earlier camera plateaus.
        effective=coded@np.linalg.inv(b['matrix_xyz']);valid=body&~b['xyz_clip_mask'];plateaus=[]
        for c in range(3):
            changed=valid&(a['camera'][...,c]-effective[...,c]>.01)
            plateaus.append({'cells':int(changed.sum()),'measured_camera_min_median_max':np.quantile(a['camera'][...,c][changed],[0,.5,1]).tolist() if changed.any() else None,
                'effective_camera_min_median_max':np.quantile(effective[...,c][changed],[0,.5,1]).tolist() if changed.any() else None})
        row=report['regions'][name];row.update({'measurable_body_cells':int(body.sum()),'XYZ_any_code65535_cells':int(bodyclip.sum()),'XYZ_per_channel_code65535_cells':(clip&body[...,None]).sum((0,1)).tolist(),
            'float_XYZ_any_gt1_cells':int((body&(predicted>1).any(-1)).sum()),'codeclip_and_floatgt1_cells':int((bodyclip&(predicted>1).any(-1)).sum()),'codeclip_without_floatgt1_cells':int((bodyclip&~(predicted>1).any(-1)).sum()),
            'float_XYZ_max_channels':predicted[body].max(0).tolist(),'measured_RGB_max_channels':b['float_measured_rgb'][body].max(0).tolist(),'XYZ16_RGB_max_channels':b['xyz16_rgb'][body].max(0).tolist(),
            'denoised_preprior_RGB_max_channels':a['preprior_rgb'][body].max(0).tolist(),'XYZ_delta_median_p95_abs_by_channel':np.quantile(abs(delta[body]),[.5,.95],axis=0).tolist(),
            'capped_floatXYZ_residual_median_p95_abs_by_channel':np.quantile(abs(clipdiff[body]),[.5,.95],axis=0).tolist(),'non_XYZ_output_clipped_camera_reduction_by_RGB':plateaus})
        print(name,json.dumps(row),flush=True)
    wb=report['kwargs'].get('user_wb')
    if wb:report['daylight_wb_green_to_red_ratio']=float(wb[1]/wb[0])
    report['limits']=['Manual body geometry is not certification of pure emitter; raw measurements constrain received camera channels only.',
        'Digital constant-image decoder matrix is a local linear proxy below digital limits, not measured camera spectral sensitivity.',
        'XYZ16=65535 and earlier effective-camera plateaus are distinct; no individual LibRaw internal subroutine was instrumented.',
        'Typical low-signal residuals include digital quantization/half-size processing. Large upper-signal plateaus cannot be attributed to denoising/prior, which are absent.',
        'No material, illumination, development, scan or source SPD parameter is fitted here.']
    (target/'results.json').write_text(json.dumps(report,indent=2)+'\n');return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,default=DEFAULT_OUT);p.add_argument('--collect',action='store_true');p.add_argument('--decoder-ceiling',action='store_true');p.add_argument('--reuse-decoder-capture',action='store_true');a=p.parse_args()
    if a.decoder_ceiling:decoder_ceiling(a.out,reuse=a.reuse_decoder_capture)
    else:
        if a.collect:collect(a.out)
        analyze(a.out)

if __name__=='__main__':main()
