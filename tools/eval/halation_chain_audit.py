"""Night-light causal ablation: same input, chemistry, scanner and frozen timing.

No-halo and first-return-only are diagnostic interventions, not proposed films.
The source masks quantify dependence on RAW reconstruction, not true radiance.
"""
import argparse
from dataclasses import replace
import json
import multiprocessing as mp
from pathlib import Path
import sys
import zlib

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / 'tools/eval'))
from light_acceptance import crop_warp, pixel_scaling, region_statistics


def sheet(images, path, name):
    from PIL import Image, ImageDraw, ImageFont
    from filmlab.spectral import srgb_encode
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 22)
    keys = [('real', 'Real scan'), ('none', 'No secondary exposure'),
            ('single', 'First return only'), ('multi', 'Existing multi-return')]
    out = Image.new('RGB', (4096, 1090), (20, 20, 20))
    for i, (key, label) in enumerate(keys):
        im = Image.fromarray(np.floor(srgb_encode(np.nan_to_num(images[key])) * 255 + .5).astype('uint8'))
        im.save(path.with_name(path.name + '_' + key + '.png'))
        out.paste(im.resize((1024, 1024), Image.Resampling.NEAREST), (1024*i, 66))
        d = ImageDraw.Draw(out); d.text((1024*i+12, 8), label, font=font, fill='white')
        d.text((1024*i+12, 36), name + ' | fixed scan/timing', font=font, fill='white')
    out.save(path.with_name(path.name + '_compare.png'))


def job(args):
    sid, regions, cache, prior, out = args
    import colour, rawpy
    from PIL import Image
    from scipy.ndimage import distance_transform_edt, zoom
    from skimage.transform import ProjectiveTransform
    from filmlab import film, generation, lights, rawio
    from filmlab.engine import Engine, Params
    from filmlab.recipes import recipe
    from filmlab.scanning.timing import PrintTiming
    from epfl_highlights import digest
    import oracle_timing as ot
    generation.use('next_sensor')
    rawpath, scanpath, bias = ot.sources(sid)
    for p, field in ((rawpath, 'raw_sha256'), (scanpath, 'film_sha256')):
        if digest(p) != ot.MAN[sid][field]: raise ValueError('Checksum mismatch')
    rgb = rawio.load_raw(rawpath, max_size=3000)
    with rawpy.imread(rawpath) as raw: sat = rawio.sensor_saturation(raw, per_channel=True)
    if sat.shape != rgb.shape: raise ValueError('Unexpected RAW orientation/size')
    seed = zlib.crc32(sid.encode()) % 1000
    metered, note = rawio.film_metering(rgb, rng=np.random.default_rng(seed), bias_ev=bias)
    eng = Engine(Params(**{**recipe('cinestill_800t'), 'timing': 'frame', 'roll': 1,
                          'frame': seed % 36, 'grain': 0, 'output_conditioning': None}))
    captured = {}
    original = film.halation
    def capture(H, source, px, v, rho):
        captured.update(direct=H, source=source, rho=rho)
        return original(H, source, px, v, rho)
    film.halation = capture
    try: baseline = eng.develop_negative(metered)
    finally: film.halation = original
    del rgb, metered
    old = json.loads((Path(cache) / (sid+'.json')).read_text()); t = old['timing']
    eng.scanner.timing = PrintTiming(K=np.array(t['K']), s=np.array(t['s']), M=tuple(np.array(a) for a in t['M']),
                                    x_knee=t['x_knee'], x_white=t['x_white'])
    eng.scanner.conditioning = 'lab'
    oldreal = np.load(Path(ot.GEO) / ('geo_'+sid+'.npz'))['real']
    with Image.open(scanpath) as im: real = colour.models.eotf_sRGB(np.asarray(im.convert('RGB'), float)/255)
    h, w = real.shape[:2]; rawh, raww = sat.shape[:2]
    step = max(1, int(round(max(rawh, raww) / 1400)))
    matrix = pixel_scaling(w/oldreal.shape[1], h/oldreal.shape[0]) @ np.array(old['geometry']) @ pixel_scaling(1/step, 1/step)
    items = []
    for region in regions:
        name = sid+'_'+region['name']
        previous = np.load(Path(prior)/(name+'.npz'))
        x,y = np.array(region['center'])*[w/oldreal.shape[1], h/oldreal.shape[0]]
        x0,y0 = int(round(x))-256, int(round(y))-256
        wp = crop_warp(ProjectiveTransform(np.array([[1,0,-x0],[0,1,-y0],[0,0,1]]) @ matrix))
        core = previous['core']; dist = distance_transform_edt(~core) * 36000/w
        masks = {'core': core}
        for lo,hi in ((0,150),(150,300),(300,500),(500,800)):
            masks[f'ring_{lo}_{hi}'] = (dist>lo)&(dist<=hi)&previous['measured'] if core.any() else np.zeros_like(core)
        items.append(dict(name=name, warp=wp, images={'real':real[y0:y0+512,x0:x0+512]},
                          core=core, measured=previous['measured'], masks=masks, stages={}, source={},
                          previous=previous['exp_lab']))
    del real
    def collect(item, key, values):
        v = item['warp'](values)
        return {name: np.mean(v[mask], axis=0).tolist() for name, mask in item['masks'].items() if mask.sum()>=30}
    for variant in ('multi','single','none'):
        H = captured['direct'] if variant=='none' else original(captured['direct'], captured['source'], eng.px_um, eng.v,
                                                                captured['rho'] if variant=='multi' else (0,0,0))
        if variant=='multi': neg=baseline
        else:
            logH = eng._log_exposure(np.clip(H,1e-9,None))
            state = eng._develop(logH,blur_px=12/eng.px_um);eng._agitation(state);eng._grain(state)
            neg = replace(baseline,state=state,log_exposure=logH)
        dn=neg.measure(eng.scanner.normalized_density)
        rendered=eng.scanner.to_rgb(dn)
        for item in items:
            item['images'][variant]=item['warp'](rendered)
            item['stages'][variant] = {key:collect(item,key,value) for key,value in
                                       (('logH',neg.log_exposure),('dye',neg.dye),('scan_density',dn))}
            if variant=='multi':
                item['baseline_max_difference'] = float(np.nanmax(np.abs(item['images'][variant]-item['previous'])))
                if item['baseline_max_difference']>2e-6: raise ValueError('Baseline does not reproduce earlier crop')
                item['halo_total']=item['warp'](H-captured['direct'])
        del rendered,dn,H
    for label,mask in (('any_saturated_source',sat.any(-1)),('all_saturated_source',sat.all(-1))):
        contribution=original(np.zeros_like(captured['direct']),captured['source']*mask[...,None],eng.px_um,eng.v,captured['rho'])
        for item in items:
            v=item['warp'](contribution)
            item['source'][label]={k:(v[m].sum(0)/np.maximum(item['halo_total'][m].sum(0),1e-20)).tolist()
                                   for k,m in item['masks'].items() if m.sum()>=30}
        del contribution
    weights=zoom(eng.light_W,(rawh/eng.light_W.shape[0],raww/eng.light_W.shape[1],1),order=1)[:rawh,:raww]
    results=[];out=Path(out)
    for item in items:
        valid=np.isfinite(item['images']['multi']).all(-1)
        stats=region_statistics(item['images'],item['core'],item['measured'],valid,36000/w)
        localW=item['warp'](weights)
        mix={c.key:float(localW[...,i][item['core']].mean()) for i,c in enumerate(lights.classes())} if item['core'].any() else None
        sheet(item['images'],out/item['name'],item['name'])
        np.savez_compressed(out/(item['name']+'.npz'),**{k:v.astype('float32') for k,v in item['images'].items()})
        results.append({'name':item['name'],'baseline_max_difference':item['baseline_max_difference'],
                        'statistics':stats,'stages':item['stages'],'halo_source_fractions':item['source'],
                        'source_class_weights':mix})
    report={'scene':sid,'metering':note,'light_mix':eng.light_mix,'regions':results,
            'limits':__doc__,'scan_um_per_pixel':36000/w,'model_um_per_pixel':eng.px_um}
    (out/(sid+'.json')).write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False))
    return report


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cache',type=Path,required=True);ap.add_argument('--prior',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True);ap.add_argument('--jobs',type=int,default=2)
    ap.add_argument('--scenes',nargs='+',default=['R1-00-1','R1-04-5','R1-07-8'])
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    regions=json.loads((REPO/'tools/eval/light_acceptance_regions.json').read_text())['regions']
    jobs=[(s,[r for r in regions if r['scene']==s],str(args.cache),str(args.prior),str(args.out)) for s in args.scenes]
    with mp.get_context('spawn').Pool(args.jobs) as pool:
        result=[]
        for r in pool.imap_unordered(job,jobs): result.append(r);print(r['scene'],'done',flush=True)
    (args.out/'results.json').write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False))


if __name__=='__main__':main()
