"""Bounded colour-balance audit using existing stable normal-area caches.

No fitting. The scan interventions are diagnostics, not device measurements or
proposed presets. Cache interpolation order is explicitly checked per region.
"""
from pathlib import Path
from dataclasses import replace
import sys,json,zlib,hashlib
import numpy as np
REPO=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(REPO),str(REPO/'tools/eval')]


def channel_neutral_timing(timing):
    """Remove only differential M balance, preserving its green-channel map."""
    if timing.M is None:return timing
    return replace(timing,M=tuple(np.full(3,np.asarray(x)[1]) for x in timing.M))


def run():
    from filmlab import generation,spectral,film
    from filmlab.engine import Engine,Params,GRAY
    from filmlab.recipes import recipe
    from filmlab.scanning.timing import PrintTiming
    from base_colour_audit import cell_rows,summarize
    from brightness_response_audit import lab
    from PIL import Image,ImageDraw,ImageFont
    generation.use('next_sensor');root=Path('/workspace/filmlab-audit');out=root/'colour-balance-audit';out.mkdir(exist_ok=True)
    report={'scope':__doc__,'scenes':[],'limits':['Only cached geometrically distant stable cells are primary; they are not proven free of real or modeled scattered light.',
        'Interpolated density then scanning does not commute with scanning then warping. Replay difference is checked, not silently discarded.',
        'No real negative density, local source SPD or scanner balance is independently measured.',
        'Removing automatic colour balance is not automatically the correct scanner model. No production change.']}
    font=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',22)
    for sid in ['cafe','R1-07-8','R1-06-7']:
        a=np.load(root/'base-colour-audit'/(sid+'.npz'));meta=json.loads((root/'base-colour-audit'/(sid+'.json')).read_text());td=meta['timing']
        e=Engine(Params(**{**recipe('cinestill_800t'),'timing':'frame','roll':1,'frame':(zlib.crc32(sid.encode())%1000)%36,'grain':0,'output_conditioning':None}))
        timing=PrintTiming(K=np.array(td['K']),s=np.array(td['s']),M=tuple(np.array(x) for x in td['M']),x_knee=td['x_knee'],x_white=td['x_white'])
        dn=a['nohalo_Dn'];safe=np.isfinite(dn).all(-1);valid=np.zeros(dn.shape[:2]+(3,));valid[safe]=dn[safe]
        states={'real':a['real'],'cached_baseline':a['nohalo']}
        for name,t in [('replayed_baseline',timing),('no_M',replace(timing,M=None)),('green_anchored_M',channel_neutral_timing(timing))]:
            e.scanner.timing=t;image=e.scanner.to_rgb(valid);image[~safe]=np.nan;states[name]=image
        mask=a['mask_far_geometric_1500']
        rows=cell_rows(states,mask,a['light_weights']);stats=summarize(rows)
        replay=np.array([np.array(r['Lab']['replayed_baseline'])-r['Lab']['cached_baseline'] for r in rows])
        errors=np.max(abs(replay),axis=1)
        if np.median(errors)>.1:
            report['scenes'].append({'scene':sid,'status':'excluded_cache_order_error','replay_cell_max_abs_Lab_p50_p95_max':np.percentile(errors,[50,95,100]).tolist()})
            continue
        # Controlled flat neutral at equal photopic input under known spectra.
        # Ratios use this material's existing sensitivity, not final-RGB fitting.
        sens=10**e.e.stock.log_sensitivity;ref=film._spec(e.e.stock.reference_illuminant);cmf=spectral.cmfs()
        href=e._expose_lin(np.full((1,3),GRAY),e.M_ref)
        witnesses=[]
        for light in ['BB3200','D65']:
            I=spectral.illuminant(light)
            ratio=(I@sens)/(ref@sens)*(ref@cmf[:,1])/(I@cmf[:,1])
            h=href*ratio
            q=e._log_exposure(h);st=e._develop(q);wd=e.scanner.normalized_density(e._transmittance(st))
            w={'light':light,'H':h.tolist(),'q':q.tolist(),'dye':st.dye.tolist(),'scan_density':wd.tolist(),'output_Lab':{}}
            for name,t in [('saved_frame',timing),('green_anchored_M',channel_neutral_timing(timing))]:
                e.scanner.timing=t;w['output_Lab'][name]=lab(e.scanner.to_rgb(wd)).tolist()
            witnesses.append(w)
        item={'scene':sid,'timing':td,'region_mask':'mask_far_geometric_1500; stricter negligible-return mask has no supported cells','normal_cells':len(rows),'normal_cell_stats':stats,
              'replay_cell_max_abs_Lab_p50_p95_max':np.percentile(errors,[50,95,100]).tolist(),
              'controlled_equal_photopic_neutrals':witnesses,'cell_records':rows}
        report['scenes'].append(item)
        sheet=Image.new('RGB',(2040,540),(18,18,18));d=ImageDraw.Draw(sheet)
        for i,(key,label) in enumerate([('real','真实800T扫描'),('cached_baseline','固定诊断负片 · 原灰平衡'),('green_anchored_M','仅撤销逐通道灰平衡 · 诊断')]):
            arr=np.uint8(np.rint(spectral.srgb_encode(np.nan_to_num(states[key]))*255));im=Image.fromarray(arr);im.thumbnail((680,470))
            sheet.paste(im,(680*i,70));d.text((680*i+8,8),label,font=font,fill='white');d.text((680*i+8,37),sid,font=font,fill='white')
        sheet.save(out/(sid+'.jpg'),quality=97)
    snapshot=json.loads((root/'lamp-edge-spectral-candidate/production_snapshot.json').read_text())['production_files_sha256']
    changed=[p for p,h in snapshot.items() if hashlib.sha256((REPO/p).read_bytes()).hexdigest()!=h]
    if changed:raise ValueError(changed)
    report['production']={'files_verified':len(snapshot),'changed':changed}
    (out/'results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    for item in report['scenes']:
        print(json.dumps({k:v for k,v in item.items() if k not in ['cell_records','controlled_equal_photopic_neutrals']},ensure_ascii=False),flush=True)


if __name__=='__main__':run()
