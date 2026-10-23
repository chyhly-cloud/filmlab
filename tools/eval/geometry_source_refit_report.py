"""Aggregate the bounded geometry/source round without choosing new candidates."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
from scipy.ndimage import gaussian_filter

REPO=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(REPO),str(REPO/'tools/eval')]


def main():
    from brightness_response_audit import lab
    from local_background_contrast import rectangle
    from filmlab.spectral import srgb_encode
    from PIL import Image,ImageDraw,ImageFont
    root=Path('/workspace/filmlab-audit')
    out=root/'catalogue-source-inverse/ceiling-geometry'
    fit=json.loads((out/'ceiling_lamp_catalogue.json').read_text())
    strength=json.loads((root/'geometry-source-refit-review/strength_radius.json').read_text())
    ambiguity=json.loads((root/'source-observability-lp/results.json').read_text())
    old=np.load(root/'ceiling-geometry-replay/ceiling_lamp_fixed112.npz')
    new=np.load(out/'ceiling_lamp_catalogue.npz')
    historic=np.load(root/'halo-residual-direction/ceiling_lamp.npz')
    cfg=json.loads((REPO/'tools/eval/local_background_regions.json').read_text())['ceiling_lamp']
    mask=rectangle(historic['distance_um'].shape,cfg['target'])&historic['measured']&(historic['distance_um']>0)&(historic['distance_um']<=300)
    inner=mask&(historic['distance_um']<=100);outer=mask&(historic['distance_um']>200)
    sigma=25/json.loads((root/'brightness-local/R1-02-3.json').read_text())['scan_um_per_pixel']
    states={'real':historic['real'],'previous_source_same_geometry':old['corrected_image'],'new_source_same_geometry':new['image']}
    contrasts={}
    for key,image in states.items():
        colors=lab(gaussian_filter(np.nan_to_num(image),(sigma,sigma,0)))
        contrasts[key]=(np.median(colors[outer],0)-np.median(colors[inner],0)).tolist()
    snapshot=json.loads((root/'lamp-edge-spectral-candidate/production_snapshot.json').read_text())['production_files_sha256']
    changed=[p for p,h in snapshot.items() if hashlib.sha256((REPO/p).read_bytes()).hexdigest()!=h]
    if changed:raise ValueError(changed)
    result={'date_utc':'2026-10-21','goal_complete':False,'source_fit':fit,'fixed_spectrum_strength_check':strength,
            'conditional_source_observability':ambiguity,'same_ceiling_groups_outer_minus_inner_Lab':contrasts,
            'production':{'verified_files':len(snapshot),'changed_files':changed},
            'related_tests_passed':29,
            'decision':'Geometry-controlled refit does not pass orange-halo acceptance. Keep source inverses diagnostic; no current/next or material/scanner changes.',
            'limits':['Ceiling-plane geometry does not certify fixture parallax or physical radii.',
                      'Source Y is an engine input unit, not SI radiance; catalogue mixtures are witnesses, not lamp identities.',
                      'Repeated same-image fits and spatial holdouts do not constitute external source measurements.',
                      'LP freezes two model moments, not measured return exposures. Its Y range is conditional, not a physical confidence interval.',
                      'Near-white core compression does not establish the cause of orange-outskirts error.']}
    for path in [out/'round_results.json',REPO/'tools/eval/geometry_source_refit_round_results.json']:
        path.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    sheet=Image.new('RGB',(1536,560),(18,18,18));draw=ImageDraw.Draw(sheet)
    font=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',20)
    for col,((_,image),label) in enumerate(zip(states.items(),['真实800T扫描','上轮源反演 · 相同新配准','本轮源反演 · 相同新配准'])):
        sheet.paste(Image.fromarray(np.uint8(np.rint(srgb_encode(np.nan_to_num(image))*255))),(512*col,48))
        draw.text((512*col+8,12),label,font=font,fill='white')
    sheet.save(out/'real_previous_refit.jpg',quality=98)
    print(json.dumps({'contrasts':contrasts,'production':result['production'],'decision':result['decision']},ensure_ascii=False))


if __name__=='__main__':main()
