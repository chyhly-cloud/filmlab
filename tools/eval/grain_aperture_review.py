"""Known-input grain aperture covariance audit, not a fit to film/scan texture.

Measures the existing grain operator on a fixed uniform dye-equivalent channel.
The 48um target is the engine's stipulated estimate, not measured 800T density.
No production code or presets are changed. References use discrete convolution
variance of independent standard normals, rather than final-RGB comparisons.
"""
from __future__ import annotations
import argparse
import dataclasses
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.integrate import quad
from scipy.ndimage import convolve, gaussian_filter
from scipy.signal import convolve2d
from scipy.stats import chi2

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from filmlab.chemistry import FilmState
from filmlab.engine import Engine, Params, _iso
from filmlab.recipes import recipe


def aperture_weights(px_um, radius_um=24.0):
    """Uniform circular aperture integrated over square pixels, then normalised.

    This assumes each rendered pixel is a constant density tile; it does not
    infer unresolved subpixel grains. quad integrates the circle/pixel overlap.
    """
    n = int(np.ceil(radius_um / px_um + .5))
    weights = np.zeros((2*n+1, 2*n+1))
    for iy, y in enumerate(range(-n,n+1)):
        y0,y1=(y-.5)*px_um,(y+.5)*px_um
        for ix,x in enumerate(range(-n,n+1)):
            x0,x1=max(-radius_um,(x-.5)*px_um),min(radius_um,(x+.5)*px_um)
            if x0>=x1: continue
            def height(u):
                h=np.sqrt(max(0.,radius_um**2-u**2))
                return max(0., min(y1,h)-max(y0,-h))
            points=[]
            for v in (abs(y0),abs(y1)):
                if v<radius_um:
                    u=np.sqrt(radius_um**2-v**2)
                    points.extend([t for t in (-u,u) if x0<t<x1])
            weights[iy,ix]=quad(height,x0,x1,points=sorted(set(points)),epsabs=1e-9)[0]
    assert np.isclose(weights.sum(),np.pi*radius_um**2,rtol=1e-10)
    return weights/weights.sum()


def smoothing_kernel(blur_px):
    """Query fixed SciPy impulse response used by production, no scene fitting."""
    if blur_px<=.35:return np.ones((1,1))
    radius=int(4*blur_px+.5)
    impulse=np.zeros((2*radius+1,2*radius+1));impulse[radius,radius]=1.
    return gaussian_filter(impulse,blur_px,mode='constant')


def sample_stats(values, expected_sigma):
    values=np.asarray(values,float)
    n=len(values);variance=float(values.var(ddof=1))
    # Centres have nominal 128um spacing. Grid rounding can reduce this to
    # 125um while pixel footprints extend the composed support to +/-62.5um,
    # so neighbouring apertures may share an edge input pixel. This weak
    # overlap and global sample-std normalisation mean the Gaussian CI is
    # approximate, not a strict coverage guarantee.
    ci=np.sqrt((n-1)*variance/chi2.ppf([.975,.025],n-1))
    return {'n_apertures':n,'mean':float(values.mean()),'rms_about_mean':np.sqrt(variance),
            'approx_gaussian_independent_sigma_ci95':ci.tolist(),
            'ratio_to_analytic_prediction':np.sqrt(variance)/expected_sigma}


def run(out):
    out.mkdir(parents=True,exist_ok=True)
    engine=Engine(Params(**{**recipe('cinestill_800t'),'roll':None,'grain':1.}))
    iso=_iso(engine.e);blur_um=1+iso/100
    d=1.;sat=float(np.clip(1-d/(engine.e.halide_cap[0]*1.05),.05,1.))
    stipulated48=.0038*np.sqrt(iso/100)*np.sqrt(d*sat)
    pixel_sizes=[15.625,7.8125,3.90625]
    seeds=list(range(8));patch_um=3000.;rows=[]
    for px in pixel_sizes:
        engine.px_um=px
        side=int(round(patch_um/px));a=aperture_weights(px)
        g=smoothing_kernel(blur_um/px);norm=float(np.sqrt(np.sum(g*g)))
        g_unit=g/norm
        # If x_i are IID N(0,1), var(sum_i q_i x_i) = sum_i q_i^2.
        # Filtered noise is renormalised to unit *pixel* std in production;
        # averaging an aperture uses q = a * (g / ||g||_2), not only a.
        q=convolve2d(a,g_unit,mode='full')
        aperture_noise_l2=float(np.sqrt(np.sum(q*q)))
        sigma=.0038*np.sqrt(iso/100)*42.5/px*np.sqrt(d*sat)
        predicted=sigma*aperture_noise_l2
        correction=stipulated48/predicted
        xs=np.arange(128.,patch_um-96.,128.)
        loc=np.rint(xs/px).astype(int)
        samples=[];candidate=[];pixels=[];seed_stats=[];clipped=0
        for seed in seeds:
            engine.v=dataclasses.replace(engine.v,frame_seed=seed)
            shape=(side,side,3);z=np.zeros(shape)
            dye=z.copy();dye[...,0]=d
            st=FilmState(z,z.copy(),z.copy(),dye,z.copy(),image_silver=np.ones(shape))
            engine._grain(st)
            delta=st.dye[...,0]-d
            clipped+=int(np.count_nonzero(st.dye[...,0]==0))
            filt=convolve(delta,a,mode='constant')
            vals=filt[np.ix_(loc,loc)].ravel()
            samples.extend(vals);candidate.extend(vals*correction)
            interior=delta[loc[0]:loc[-1]+1,loc[0]:loc[-1]+1]
            pixels.append(float(interior.std()))
            seed_stats.append({'seed':seed,'n':len(vals),'sigma':float(vals.std(ddof=1))})
        row={'px_um':px,'image_width_at_36mm':int(round(36000/px)),
             'patch_pixels':side,'blur_um':blur_um,'blur_pixels':blur_um/px,
             'fixed_dye_amount':d,'fixed_saturation_factor':sat,
             'stipulated_aperture_rms_dye_units':stipulated48,
             'production_pixel_sigma_formula':sigma,
             'observed_pixel_sigma_mean':float(np.mean(pixels)),
             'gaussian_kernel_l2':norm,'aperture_on_unit_pixel_noise_l2':aperture_noise_l2,
             'analytic_aperture_rms_dye_units':predicted,
             'analytic_ratio_to_stipulated':predicted/stipulated48,
             'production':sample_stats(samples,predicted),
             'audit_only_covariance_normalisation':{'factor':correction,
                 'formula':'stipulated48 / (production_pixel_sigma * || aperture * (g / ||g||2) ||2)',
                 'observed':sample_stats(candidate,stipulated48)},
             'per_seed':seed_stats,'clipped_pixels':clipped}
        rows.append(row)
    # Independent exact variance identity for a uniform M-pixel average.
    m=16;square=np.ones((4,4))/m
    assert np.isclose(np.sum(square*square),1/m)
    # Exact overlapping two-tap example: averaging two adjacent unit-variance
    # (x_i+x_{i-1})/sqrt(2) samples has variance (1+4+1)/8 = 3/4,
    # rather than the independent-sample average's 1/2.
    two_tap=convolve2d(np.full((1,2),.5),np.ones((1,2))/np.sqrt(2),mode='full')
    assert np.isclose(np.sum(two_tap**2),.75)
    # Constant shift is preserved by a normalised measurement aperture.
    assert all(np.isclose(aperture_weights(px).sum(),1) for px in pixel_sizes)
    for r in rows:
        lo,hi=r['production']['approx_gaussian_independent_sigma_ci95']
        assert lo <= r['analytic_aperture_rms_dye_units'] <= hi
        lo,hi=r['audit_only_covariance_normalisation']['observed']['approx_gaussian_independent_sigma_ci95']
        assert lo <= stipulated48 <= hi
        assert r['clipped_pixels']==0
    assert rows[-1]['production']['rms_about_mean'] > 3*rows[0]['production']['rms_about_mean']
    record={'scope':__doc__, 'configuration':{'recipe':'cinestill_800t','generation_setting':'unchanged/default',
        'grain':1,'roll':None,'flat_red_dye_amount':d,'other_dye_channels':0,
        'grain_factor':1,'iso_from_current_name_parser':iso,'seeds':seeds,
        'physical_patch_um':patch_um,'aperture_diameter_um':48,'aperture_centres_spacing_um':128},
      'analytic_reference':'For IID zero-mean unit-variance x, Var(q*x)=sum(q**2). Pixel-normalised smoothing induces covariance; aperture variance uses convolution of the averaging and smoothing kernels.',
      'acceptance':{'square_average_variance':'pass: 1/16',
        'two_tap_correlated_average_variance':'pass: 3/4, not 1/2',
        'analytic_predictions_within_observed_approx_ci95':True,
        'audit_normalisation_targets_within_observed_approx_ci95':True,
        'no_clipped_pixels':True,
        'scope':'Acceptance of this fixed synthetic experiment, not a production-ready grain correction or measured film validation.'},
      'units':'dye-equivalent channel coefficient; not Status M or measured 800T density. A fixed monochromatic dye coefficient multiplies all these RMS values by the same factor but does not change their resolution ratios.',
      'limitations':['Uniform synthetic state, no scanner/noise/sharpening or real texture used.',
        '48um is a circular tile-average measurement; unresolved subpixel grain unknown.',
        'Engine rms48 and grain blur are stipulated estimates, not actual 800T granularity measurements.',
        'Nominal 128um aperture-centre spacing can round to 125um while composed pixel/filter support reaches +/-62.5um, so neighbouring apertures may weakly share edge inputs. Global per-realisation n.std() also induces common dependence; Gaussian confidence intervals are approximate, not a strict coverage guarantee.',
        'No identical realisation is expected across resolutions; compare ensemble statistics.',
        'Audit-only covariance correction targets this stipulated aperture; its PSD/material realism is not validated.'],
      'results':rows,
      'production_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__).resolve().parents[2]/'filmlab/engine.py']}}
    (out/'results.json').write_text(json.dumps(record,indent=2)+'\n')
    for r in rows:
        print(r['px_um'],r['analytic_ratio_to_stipulated'],r['production']['rms_about_mean'],r['audit_only_covariance_normalisation']['observed']['rms_about_mean'])
    return record

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path,required=True)
    run(ap.parse_args().out)
