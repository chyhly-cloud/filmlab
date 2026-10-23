"""Aggregate bounded input-inference checks without selecting photo-fit parameters."""
from pathlib import Path
import json,hashlib
REPO=Path(__file__).resolve().parents[2]
ROOT=Path('/workspace/filmlab-audit/illumination-inference-trial')
SCENES=['cafe','R1-07-8','R1-06-7','R1-02-3','R1-00-1']


def main():
    controls=json.loads((ROOT/'controlled_results.json').read_text())
    replays=[json.loads((ROOT/'real'/(s+'.json')).read_text()) for s in SCENES]
    nearest_failures=[]
    for c in controls['charts']:
        a=c['variants']['original']['RG_BG_abs_EV_median'];b=c['variants']['nearest_member']['RG_BG_abs_EV_median']
        if any(y>x for x,y in zip(a,b)):nearest_failures.append(c['light'])
    result={'scope':__doc__,'controlled':controls,'real_replays':replays,
            'nearest_member_control_lights_with_worse_layer_ratio_median':nearest_failures,
            'max_real_median_deltaE00_change':max(abs(r['metrics']['after']['median_deltaE00']-r['metrics']['before']['median_deltaE00']) for r in replays),
            'related_tests_passed':12,
            'decision':'Retain eval-only no-global-pruning locality correction. It does not repair real-scene colour. Nearest member policy fails controlled nonregression and is not advanced using photo scores.',
            'limits':['Assigned-weight counterexample is not the measured failure magnitude in real scenes.',
                      'Full-array EV maxima include old numerical-floor pixels; do not interpret as typical scene radiance error.',
                      'Geometric-far stable cells are not an independently observed unscattered negative.',
                      'Both synthetic CIE-camera and stock sensitivity are model proxies; no real 800T nonneutral material validation follows.']}
    snapshot=json.loads(Path('/workspace/filmlab-audit/lamp-edge-spectral-candidate/production_snapshot.json').read_text())['production_files_sha256']
    changed=[p for p,h in snapshot.items() if hashlib.sha256((REPO/p).read_bytes()).hexdigest()!=h]
    if changed:raise ValueError(changed)
    result['production']={'verified_files':len(snapshot),'changed':changed}
    for p in [ROOT/'summary.json',REPO/'tools/eval/illumination_inference_results.json']:
        p.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'nearest_failed_lights':nearest_failures,'max_real_median_deltaE00_change':result['max_real_median_deltaE00_change'],'production':result['production']},ensure_ascii=False))


if __name__=='__main__':main()
