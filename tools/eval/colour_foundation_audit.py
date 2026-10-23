"""Controlled colour-chain contracts, independent of final photograph fitting.

Measured ColorChecker reflectances and known illuminants define radiance.
The ideal CIE camera is a qualified algorithm proxy, not a measured RAW SSF.
Model sensitivity supplies conditional layer-exposure truth, NOT measured 800T
C-41 development. Fixed scanner timing never balances individual test images.
"""
from pathlib import Path
from dataclasses import replace
import sys,json,hashlib
import numpy as np
REPO=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(REPO),str(REPO/'tools/eval')]
OUT=Path('/workspace/filmlab-audit/colour-foundation-audit')


def layer_error(inferred,reference):
    ev=np.log2(np.maximum(inferred,1e-30)/np.maximum(reference,1e-30))
    cast=ev-ev[...,1:2]
    return {'layer_abs_EV_median':np.median(abs(ev),axis=0).tolist(),
            'layer_abs_EV_max':np.max(abs(ev),axis=0).tolist(),
            'RG_BG_abs_EV_median':np.median(abs(cast[..., [0,2]]),axis=0).tolist(),
            'RG_BG_abs_EV_max':np.max(abs(cast[..., [0,2]]),axis=0).tolist()}


def expose_with_separate_gain(rgb,response,gain):
    """Eval-only exposure operation: hold reconstructed spectrum fixed."""
    from filmlab.film import expose
    if not np.isfinite(gain) or gain<0:raise ValueError('Exposure gain must be finite and nonnegative')
    return expose(rgb,response)*gain


def main():
    import colour
    from filmlab import generation,lights,spectral,film,chemistry
    from filmlab.engine import Engine,Params
    from filmlab.recipes import recipe
    from brightness_response_audit import lab
    from PIL import Image,ImageDraw,ImageFont
    generation.use('next_sensor');OUT.mkdir(exist_ok=True)
    shape=colour.SpectralShape(380,780,5)
    chart=colour.SDS_COLOURCHECKERS['ColorChecker N Ohta'];names=list(chart)
    R=np.array([sd.copy().align(shape).values for sd in chart.values()])
    e=Engine(Params(**{**recipe('cinestill_800t'),'exposure_ev':0.,'timing':'standard','scan_neutral':'film',
                      'scan_color_strength':0.,'roll':None,'frame':0,'grain':0,'output_conditioning':None}))
    S=10**e.e.stock.log_sensitivity
    report={'scope':__doc__,'reflectance_source':'colour-science ColorChecker N Ohta, 24 measured spectra',
            'known_light_checks':[],'same_patch_context_checks':[],'exposure_operation_checks':[],
            'limits':['This tests spectral inference using the stored material sensitivity, not its accuracy for real CineStill 800T.',
                      'Known reflectance/light controls cannot certify the true illuminants in the previously shown photographs.',
                      'RGB metamerism precludes exact recovery of every coloured reflectance; nonzero exposure error alone is not an implementation bug.',
                      'Exposure-gain contract holds an inferred spectrum fixed; changing surface reflectance brightness is a different operation.',
                      'Native-process characteristic reproduction tests are calibration closure, not independent nonneutral C-41 validation.']}
    all_H=[];saved={};tiles=[]
    luma=np.array([.2126,.7152,.0722])
    for light in ['D65','BB3200','LED-B3','FL11']:
        I=spectral.illuminant(light) if light in ['D65','BB3200'] else lights._sd(light)
        I=I/float(lights.camera_rgb(I)@luma)
        radiance=R*I;rgb=lights.camera_rgb(radiance);truth=radiance@S
        method='spectral' if light=='D65' else 'linear_wp'
        response=film.exposure_response(e.e.stock,I,method=method)
        known=film.expose(rgb,response)
        # Spatial chart, identical illumination everywhere; no automatic scan balancing.
        image=np.repeat(np.repeat(rgb.reshape(4,6,3),24,axis=0),24,axis=1)
        H,_=e._expose_mixed(image)
        auto=np.array([H[y*24+12,x*24+12] for y in range(4) for x in range(6)])
        row={'light':light,'known_mapping_method':method,'known_input_rgb_min':float(rgb.min()),
             'known_light_RGB_mapping':layer_error(known,truth),'automatic_light_RGB_mapping':layer_error(auto,truth),
             'inferred_global_class_shares':e.light_mix}
        report['known_light_checks'].append(row)
        saved[light+'_truth_H']=truth;saved[light+'_known_H']=known;saved[light+'_auto_H']=auto
        all_H.append(truth)
        outputs={}
        for key,h in [('known_spectrum',truth),('known_illuminant_RGB',known),('automatic_illumination_RGB',auto)]:
            q=e._log_exposure(h);st=e._develop(q);T=e._transmittance(st);Dn=e.scanner.normalized_density(T);out=e.scanner.to_rgb(Dn)
            saved[light+'_'+key+'_q']=q;saved[light+'_'+key+'_dye']=st.dye
            saved[light+'_'+key+'_spectral_density']=-np.log10(T);saved[light+'_'+key+'_scan_density']=Dn
            outputs[key]=out;saved[light+'_'+key+'_linear_RGB']=out
        row['fixed_scan_output_difference_from_spectral_reference']={key:{'median_deltaE00':float(np.median(colour.delta_E(lab(out),lab(outputs['known_spectrum']),method='CIE 2000'))),
                 'max_deltaE00':float(np.max(colour.delta_E(lab(out),lab(outputs['known_spectrum']),method='CIE 2000')))} for key,out in outputs.items()}
        tiles.append((light,outputs))
        for ev in [-2.,-float(np.log2(800/500)),2.]:
            gain=2**ev;pre=film.expose(rgb*gain,response);post=expose_with_separate_gain(rgb,response,gain)
            shift=np.log2(pre/post);cast=shift-shift[:,1:2]
            raw_base=response(rgb);raw_pre=response(rgb*gain)
            supported=(raw_base>1e-7)&(raw_pre>1e-7)
            unguarded=np.full_like(raw_base,np.nan)
            unguarded[supported]=np.log2(raw_pre[supported]/(gain*raw_base[supported]))
            report['exposure_operation_checks'].append({'light':light,'gain_EV':ev,
                'gain_before_vs_after_recovery_layer_abs_EV_max':abs(shift).max(0).tolist(),
                'gain_before_vs_after_RG_BG_abs_EV_max':abs(cast[:,[0,2]]).max(0).tolist(),
                'largest_layer_shift_patch':names[int(np.argmax(abs(shift).max(1)))],
                'tiny_or_floored_layer_samples':int((~supported).sum()),
                'unfloored_layer_abs_EV_max':np.nanmax(abs(unguarded),axis=0).tolist(),
                'separate_gain_physical_target_error':layer_error(post,truth*gain)})
    # A fixed 18% grey patch under unchanged known light, coloured reflectors around it.
    for light in ['D65','BB3200']:
        I=spectral.illuminant(light);I=I/float(lights.camera_rgb(I)@luma)
        center=lights.camera_rgb(.18*I);trueH=(.18*I)@S
        central=[];contexts=[]
        for background in ['neutral_18','blue','green','orange']:
            r=np.full(81,.18) if background=='neutral_18' else R[names.index(background)]
            im=np.broadcast_to(lights.camera_rgb(r*I),(96,96,3)).copy();im[40:56,40:56]=center
            h,_=e._expose_mixed(im);central.append(h[46:50,46:50].mean((0,1)))
            # Same exact camera pixel values, same actual SPD/radiance, same film.
            if not np.array_equal(im[48,48],center):raise ValueError('Controlled patch changed')
            contexts.append({'background_reflectance':background,'center_H':central[-1].tolist(),
                             'center_layer_EV_error_vs_spectral_integral':np.log2(central[-1]/trueH).tolist()})
        central=np.array(central);span=np.ptp(np.log2(central),axis=0)
        report['same_patch_context_checks'].append({'light':light,'fixed_actual_H':trueH.tolist(),
            'center_layer_EV_span_across_backgrounds':span.tolist(),'contexts':contexts})
    # Internal development and optical-density constraints; not a materials calibration.
    H=np.concatenate(all_H,axis=0);q=e._log_exposure(H);st=e._develop(q)
    T=e._transmittance(st);density=-np.log10(T)
    spectral_closure=float(abs(density-(st.dye@e._dye_spectra(st.dye_agent).T+e.e.stock.base_density)).max())
    monotone=[]
    for c in range(3):
        step=q.copy();step[:,c]+=np.log10(2);d=e._develop(step).dye-st.dye
        monotone.append(float(d[:,c].min()))
    report['development_internal_checks']={'minimum_own_dye_change_for_plus1EV_per_layer':monotone,
        'minimum_dye':float(st.dye.min()),'silver_after_full_bleach_max':float(abs(st.silver).max()),
        'halide_after_fix_max':float(abs(st.halide).max()),'transmittance_min_max':[float(T.min()),float(T.max())],
        'spectral_density_reconstruction_max':spectral_closure}
    snapshot=json.loads((Path('/workspace/filmlab-audit/lamp-edge-spectral-candidate/production_snapshot.json')).read_text())['production_files_sha256']
    changed=[p for p,h in snapshot.items() if hashlib.sha256((REPO/p).read_bytes()).hexdigest()!=h]
    if changed:raise ValueError(changed)
    report['production']={'verified_files':len(snapshot),'changed':changed}
    np.savez_compressed(OUT/'trace.npz',**saved,reflectances=R,patch_names=names)
    (OUT/'results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    sheet=Image.new('RGB',(1920,4*300),(18,18,18));draw=ImageDraw.Draw(sheet)
    font=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',21)
    for j,(light,outputs) in enumerate(tiles):
        for i,(key,title) in enumerate([('known_spectrum','已知光谱积分参考'),('known_illuminant_RGB','RGB＋已知光源'),('automatic_illumination_RGB','RGB＋自动光源估计')]):
            arr=np.uint8(np.rint(spectral.srgb_encode(outputs[key].reshape(4,6,3))*255))
            im=Image.fromarray(arr).resize((600,240),Image.Resampling.NEAREST);sheet.paste(im,(640*i+10,300*j+55))
            draw.text((640*i+10,300*j+8),light+' · '+title,font=font,fill='white')
    sheet.save(OUT/'controlled_chart.jpg',quality=98)
    print(json.dumps(report,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
