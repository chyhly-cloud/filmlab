"""No-refit fixed-negative replay with independently accepted ceiling geometry.

The affine is selected from remote RAW/scan lines and withheld corners only.
No film, lamp spectrum/strength, output response, or print timing is adjusted.
"""
from pathlib import Path
import json
import sys

import numpy as np
from skimage.transform import ProjectiveTransform

REPO=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(REPO),str(REPO/'tools/eval')]


def run():
    from catalogue_source_inverse import MomentCase
    from inverse_source_radiance import observation_operator
    from measurement_aperture_audit import native_support_overlap
    from light_acceptance import crop_warp
    from brightness_response_audit import lab
    from filmlab.spectral import srgb_encode
    from PIL import Image,ImageDraw,ImageFont
    root=Path('/workspace/filmlab-audit');out=root/'ceiling-geometry-replay';out.mkdir(exist_ok=True)
    geometries=json.loads((root/'ceiling-line-registration/results.json').read_text())
    report={'scope':__doc__,'refit':False,'regions':[]}
    font=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',20)
    for geometry in geometries['regions']:
        if not geometry['adopt_for_diagnostic']:continue
        name=geometry['name'];case=MomentCase(name);a,e=case.a,case.e
        source=np.load(root/'inverse-source-radiance'/(name+'_source.npz'))
        A=np.array(geometry['matrix_raw_warp_to_scan']);matrix=A@a['matrix']
        P=observation_operator(a['source_profile'].shape,matrix,a['points'],25/case.meta['scan_um_per_pixel'])
        overlap,_=native_support_overlap(P,np.isin(a['groups'],[0,2]),np.isin(a['groups'],[1,3]))
        outside=np.unique(P.indices)[~source['safe'].ravel()[np.unique(P.indices)]]
        if len(outside):raise ValueError('Corrected measurement requires source pixels outside audited safe crop')
        row={'name':name,'geometry':geometry,'matrix':matrix.tolist(),'source_safe_coverage':'All required native pixels inside existing safe mask',
             'new_P_native_overlap':overlap,'fixed_states':{}}
        states={
            'legacy_full':source['q_base'],
            'corrected_inputs_old_core':np.load(root/'raw-hdr-film-replay'/(name+'.npz'))['q'],
            'fixed112':np.load(root/'catalogue-source-inverse'/(name+'_catalogue.npz'))['q'],
            'field_original_sectors':np.load(root/'censored-source-field'/(name+'.npz'))['q'],
            'field_reverse_sectors':np.load(root/'censored-source-field/reverse-sectors'/(name+'.npz'))['q']}
        baseline,corrected={},{}
        for key,q in states.items():
            state=e._develop(q,blur_px=12/e.px_um);state.dye*=a['factor'][...,None]
            native=e.scanner.to_rgb(e.scanner.normalized_density(e._transmittance(state)))
            oldlab=lab(case.P@native.reshape(-1,3));newlab=lab(P@native.reshape(-1,3))
            baseline[key]=crop_warp(ProjectiveTransform(a['matrix']))(native)
            corrected[key]=crop_warp(ProjectiveTransform(matrix))(native)
            row['fixed_states'][key]={'original_geometry':case.stats(oldlab),'ceiling_geometry':case.stats(newlab)}
            np.savez_compressed(out/(name+'_'+key+'.npz'),old_image=baseline[key],corrected_image=corrected[key],native_rgb=native,
                                old_lab=oldlab,corrected_lab=newlab,matrix=matrix,points=a['points'],groups=a['groups'],target_lab=a['target_lab'])
        sheet=Image.new('RGB',(2048,560),(18,18,18));draw=ImageDraw.Draw(sheet)
        for col,(label,values) in enumerate([('真实800T扫描',source['real']),
            ('旧几何 · 112谱固定场',baseline['fixed112']),
            ('格线校验几何 · 同一固定场',corrected['fixed112']),
            ('格线校验几何 · 同一自由场',corrected['field_original_sectors'])]):
            image=Image.fromarray(np.uint8(np.rint(srgb_encode(np.nan_to_num(values))*255)))
            sheet.paste(image,(col*512,48));draw.text((col*512+8,12),label,font=font,fill='white')
        sheet.save(out/(name+'_comparison.jpg'),quality=98)
        report['regions'].append(row)
        print(name,json.dumps(row['fixed_states'],ensure_ascii=False),flush=True)
    (out/'results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    return report


if __name__=='__main__':run()
