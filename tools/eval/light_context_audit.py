"""Known-spectrum source against changed surroundings: isolate S0 inference.

Ideal CIE camera, positive RGB sources, no sensor saturation, reconstruction,
metering or output rendering. Exact spectral exposures are the independent
reference; the known-member linear_wp result is a constructed white constraint.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from filmlab import film, generation, lights
from filmlab.engine import Engine, Params
from filmlab.recipes import recipe
from filmlab.incident import IncidentSpectrum


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    generation.use('next_sensor')
    engine=Engine(Params(**{**recipe('cinestill_800t'),'grain':0,'roll':1,'frame':0,'timing':'frame'}))
    yy,xx=np.mgrid[:256,:256];mask=(xx-128)**2+(yy-128)**2<=4**2
    backgrounds={'neutral':[1,1,1],'blue':[.15,.5,2],'warm':[2,.6,.15]}
    luma=np.array([.2126,.7152,.0722]);results=[]
    for name, spectrum in [('2700K',lights._blackbody(2700)),('LED-B3',lights._sd('LED-B3'))]:
        spectrum=spectrum*3/(lights.camera_rgb(spectrum)@luma)
        source=lights.camera_rgb(spectrum)
        if np.any(source<=0):raise ValueError('Source must lie in positive RGB domain')
        target=spectrum@(10**engine.e.stock.log_sensitivity)
        known=film.exposure_response(engine.e.stock,spectrum,method='linear_wp')(source)
        for bgname,bg in backgrounds.items():
            bg=np.array(bg)*.02/(np.array(bg)@luma)
            image=np.broadcast_to(bg,(256,256,3)).copy();image[mask]=source
            H,src=engine._expose_mixed(image)
            observed=H[128,128];halo=src[128,128]
            state=engine._develop(engine._log_exposure(observed[None,:]))
            trace={}
            engine.develop_negative(image,incident_sources=[IncidentSpectrum(spectrum,np.ones(mask.shape),mask,'controlled known source')],trace=trace)
            results.append({'light':name,'background':bgname,'source_RGB':source.tolist(),
                            'expected_direct_H':target.tolist(),'inferred_direct_H':observed.tolist(),
                            'direct_error_stops':np.log2(observed/target).tolist(),
                            'known_member_relative_error':((known-target)/target).tolist(),
                            'inferred_halo_source':halo.tolist(),'dye':state.dye[0].tolist(),
                            'explicit_direct_H':trace['direct_exposure'][128,128].tolist(),
                            'explicit_halo_source':trace['halo_source'][128,128].tolist(),
                            'source_class_weights':{c.key:float(engine.light_W[128,128,i]) for i,c in enumerate(lights.classes())}})
    ranges={}
    for name in ['2700K','LED-B3']:
        rows=[r for r in results if r['light']==name]
        h=np.array([r['inferred_direct_H'] for r in rows]);s=np.array([r['inferred_halo_source'] for r in rows])
        ranges[name]={'direct_range_stops':np.log2(h.max(0)/h.min(0)).tolist(),
                      'halo_source_range_stops':np.log2(s.max(0)/s.min(0)).tolist()}
        for key in ('explicit_direct_H','explicit_halo_source'):
            values=np.array([r[key] for r in rows])
            ranges[name][key+'_range_stops']=np.log2(values.max(0)/values.min(0)).tolist()
    report={'scope':__doc__,'fixed_source_Y':3,'background_Y':.02,'source_radius_px':4,
            'image_size':256,'ranges':ranges,'cases':results}
    (args.out/'context.json').write_text(json.dumps(report,indent=2,allow_nan=False));print(json.dumps(ranges,indent=2))


if __name__=='__main__':main()
