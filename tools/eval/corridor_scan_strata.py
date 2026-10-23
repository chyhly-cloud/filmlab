"""Same-amplitude local cyan scan derivative on actual cached corridor negatives.

No synthetic exposure grid: this checks scanner gain separately from differences
in actual dye increments across model brightness strata. Fixed operator timing.
"""
from pathlib import Path
import sys,json,zlib,numpy as np
REPO=Path(__file__).resolve().parents[2];sys.path[:0]=[str(REPO),str(REPO/'tools/eval')]
from filmlab import generation
from filmlab.engine import Engine,Params
from filmlab.recipes import recipe
from filmlab.scanning.timing import PrintTiming
from corridor_dye_response import scanner
from corridor_cache_geometry import CacheGeometry
from brightness_response_audit import lab
generation.use('next_sensor');sid='R1-02-3';seed=zlib.crc32(sid.encode())%1000
e=Engine(Params(**{**recipe('cinestill_800t'),'timing':'frame','roll':1,'frame':seed%36,'grain':0,'output_conditioning':None}))
root=Path('/workspace/filmlab-audit/corridor-dye-response');allout=[]
td=json.load(open('/workspace/filmlab-audit/epfl-highlights/R1-02-3.json'))['timing'];t=PrintTiming(K=np.array(td['K']),s=np.array(td['s']),M=tuple(np.array(v) for v in td['M']),x_knee=td['x_knee'],x_white=td['x_white'])
for name in ['ceiling_lamp','middle_ceiling','glass_reflection']:
 a=np.load(root/'cache'/(name+'.npz'));g=CacheGeometry.from_cache(a);wp=g.warp
 old=np.load('/workspace/filmlab-audit/brightness-local/R1-02-3_'+name+'.npz');d=a['native_base_dye'];mask=(old['measured']>.5)&(old['distance_um']>0)&(old['distance_um']<=300)&(old['real_bin']==3)&a['output_trusted']
 derivative={};stability={};stable_masks={}
 for eps in [1e-5,1e-4]:
  p=scanner(e,d+[eps,0,0],t);m=scanner(e,d-[eps,0,0],t)
  for key in ['exp','paper']:
   j=(lab(wp(p[key]))-lab(wp(m[key])))/(2*eps)
   if eps==1e-5:derivative[key]=j
   else:
    difference=np.linalg.norm(j-derivative[key],axis=-1);relative=difference/np.maximum(np.linalg.norm(derivative[key],axis=-1),1e-8)
    stable_masks[key]=relative<.01
    stability[key]={'max_absolute_difference':float(np.max(difference[mask])),'fraction_relative_gt1pct':float((relative[mask]>.01).mean()),'median_relative_difference':float(np.median(relative[mask]))}
 rows=[]
 for bi,label in [(2,'model_Y_.05_.4'),(3,'model_Y_.4_.85'),(4,'model_Y_ge_.85')]:
  target=mask&(old['model_bin']==bi)
  if target.sum()<20:continue
  row={'model_bin':label,'pixels':int(target.sum()),'mean_scan_cyan_Lab_derivative':{key:np.mean(j[target],axis=0).tolist() for key,j in derivative.items()},'median_scan_cyan_chroma_gain':{key:float(np.median(np.linalg.norm(j[target,1:],axis=-1))) for key,j in derivative.items()},'stable_sample_fraction':{key:float(stable[target].mean()) for key,stable in stable_masks.items()},'stable_median_scan_cyan_chroma_gain':{key:float(np.median(np.linalg.norm(derivative[key][target&stable,1:],axis=-1))) for key,stable in stable_masks.items()},'inputs':{}}
  for case,field in [('red_return','native_minus_red_dye'),('all_return','native_alloff_dye')]:
   delta=d-a[field];delta[~a['native_trusted']]=0;dc=wp(delta)
   row['inputs'][case]={'mean_delta_dye':np.mean(dc[target],axis=0).tolist(),'mean_delta_D650':float(np.mean((dc@e._dye_spectra('CD-4')[54])[target]))}
  rows.append(row)
 allout.append({'name':name,'scan_derivative_unit':'Lab per uniform model cyan coefficient increment at actual final negative; floating display RGB domain','max_step_change_Lab_derivative':stability,'strata':rows})
(root/'actual_scan_strata.json').write_text(json.dumps(allout,indent=2));print(json.dumps(allout,indent=2))
