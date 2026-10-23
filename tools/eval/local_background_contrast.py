"""Same-frame ceiling contrast, not measured physical halation.

Frozen real/nohalo/full RGB caches, geometry and timing. Backgrounds were chosen
from visible material/geometry before computing differences. Similar finish
does not imply equal illumination, BRDF or absence of returned light.
"""
from pathlib import Path
import json,sys,hashlib
import numpy as np
REPO=Path(__file__).resolve().parents[2];sys.path[:0]=[str(REPO),str(REPO/'tools/eval')]
from brightness_response_audit import lab,LUMA
ROOT=Path('/workspace/filmlab-audit');OUT=ROOT/'local-background'


def rectangle(shape,box):
    x0,y0,x1,y1=box;m=np.zeros(shape,bool);m[y0:y1,x0:x1]=True;return m


def relative_logs(rgb,background):
    """Output log2Y and log2 channel-ratio contrasts. Positive finite support only.

    Subtraction cancels multiplicative output luminance scaling (logY) or
    diagonal output channel gains (log ratios), provided there is no clipping.
    It does not undo nonlinear upstream scanner curves or colour mixing.
    """
    rgb=np.asarray(rgb,float)
    values=np.full(rgb.shape,np.nan);Y=rgb@LUMA
    valid=(Y>0)&np.isfinite(Y);values[...,0][valid]=np.log2(Y[valid])
    for col,channel in [(1,0),(2,2)]:
        valid=(rgb[...,channel]>0)&(rgb[...,1]>0)&np.isfinite(rgb[...,channel])&np.isfinite(rgb[...,1])
        values[...,col][valid]=np.log2(rgb[...,channel][valid]/rgb[...,1][valid])
    for col in range(3):
        # An undefined reference channel invalidates THIS contrast, without a
        # made-up epsilon floor or silently choosing a different background.
        if not np.isfinite(values[...,col][background]).all():values[...,col]=np.nan
        else:values[...,col]-=np.median(values[...,col][background])
    return values


def contrast_lab(Lab,background):return Lab-np.median(Lab[background],axis=0)


def circular_average(degrees):
    if not len(degrees):return None,None
    value=np.mean(np.exp(1j*np.radians(degrees)))
    return (float(np.degrees(np.angle(value))) if abs(value)>1e-8 else None),float(abs(value))


def summary(Lab,rgb,background,target):
    bg=np.median(Lab[background],axis=0);value=contrast_lab(Lab,background)[target]
    c=np.linalg.norm(Lab[...,1:],axis=-1);h=np.degrees(np.arctan2(Lab[...,2],Lab[...,1]));bc=float(np.median(c[background]))
    # Hue circular mean with finite, chromatic support; neutral points excluded.
    goodbg=background&(c>=5);goodtarget=target&(c>=5)
    bh,bhr=circular_average(h[goodbg])
    dh=(h[goodtarget]-bh+180)%360-180 if bh is not None else np.array([])
    dh_mean,dh_r=circular_average(dh)
    rel=relative_logs(rgb,background)[target];valid_logs=np.isfinite(rel)
    return {'pixels':int(target.sum()),'background_pixels':int(background.sum()),'background_median_Lab':bg.tolist(),
        'background_MAD_Lab':np.median(abs(Lab[background]-bg),axis=0).tolist(),
        'target_median_Lab':np.median(Lab[target],axis=0).tolist(),'median_delta_Lab':np.median(value,axis=0).tolist(),
        'median_delta_pixel_chroma':float(np.median(c[target]-bc)),
        'background_median_pixel_chroma':bc,'target_median_pixel_chroma':float(np.median(c[target])),
        'background_hue_deg':bh,'background_hue_resultant_length':bhr,'target_chromatic_fraction':float(goodtarget.sum()/max(target.sum(),1)),
        'circular_mean_hue_shift_deg_when_defined':dh_mean,'hue_shift_resultant_length':dh_r,
        'output_log_contrast_defined_fraction':valid_logs.mean(0).tolist(),
        'median_output_logY_logRG_logBG_contrast':[float(np.median(rel[valid_logs[:,c],c])) if valid_logs[:,c].any() else None for c in range(3)]}


def main():
    from scipy.ndimage import gaussian_filter,shift
    from filmlab.spectral import srgb_encode
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    OUT.mkdir(exist_ok=True)
    config_path=REPO/'tools/eval/local_background_regions.json';config=json.loads(config_path.read_text())
    um=json.loads((ROOT/'brightness-local/R1-02-3.json').read_text())['scan_um_per_pixel']
    report={'scope':__doc__,'region_definition_sha256':hashlib.sha256(config_path.read_bytes()).hexdigest(),'scan_um_per_pixel':um,'regions':[],
        'primary_background':'same_panel_right; alternatives retained without selecting best match',
        'limits':['local scene contrast is not pure real halation or a true unlit base',
            'two lamps one scene/roll/lab; same finish is a visual classification, not measured reflectance or equal illumination',
            'deltaLab cancels only additive Lab offsets, not a complete nonlinear scan white-balance operation',
            'relative logRGB ratios cancel diagonal final linear RGB gains only; nonlinear tone/matrix/local effects remain',
            'ceilings only: fixture/housing/core excluded; strips do not measure a complete radial annulus',
            'similar-finish above/below panels have illumination/BRDF gradients; no fitted background surface or offset chosen',
            'nominal core distances and 36mm scaling are geometric proxies; pixel samples are correlated']}
    fig,axes=plt.subplots(2,3,figsize=(13,8.5),layout='constrained')
    fprof,apro=plt.subplots(2,5,figsize=(17,7),layout='constrained')
    for row,name in enumerate(['ceiling_lamp','middle_ceiling']):
        path=ROOT/'halo-residual-direction'/(name+'.npz');a=np.load(path);c=config[name]
        images={key:a[key] for key in ['real','nohalo','halo']};dist=a['distance_um'];measured=a['measured'].astype(bool)
        target=rectangle(dist.shape,c['target'])&measured&(dist>0)&(dist<=300)
        backgrounds={key:rectangle(dist.shape,box) for key,box in c['backgrounds'].items()}
        if not target.any():raise ValueError('no measured ceiling target')
        reg={'name':name,'cache_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'target_pixels':int(target.sum()),'geometry_policy':c['policy'],
             'seam_checks':[],'scales':{},'registration_sensitivity':[]}
        # Structural seam audit estimates offsets only; it never changes geometry.
        for seam in c['seams']:
            x0,y0,x1,y1=seam['rect'];positions={};edge={}
            for key,rgb in images.items():
                lum=gaussian_filter(rgb@LUMA,1)[y0:y1,x0:x1]
                profile=lum.mean(0 if seam['axis']=='x' else 1)
                positions[key]=int(np.argmin(profile))+(x0 if seam['axis']=='x' else y0)
                edge[key]=bool(np.argmin(profile) in (0,len(profile)-1))
            reg['seam_checks'].append({'definition':seam,'dark_seam_position':positions,
                'minimum_at_window_boundary':edge,'model_minus_real_pixels':positions['halo']-positions['real']})
        def spatial_shape(Lab,rgb):
            inner=target&(dist>0)&(dist<=100);outer=target&(dist>200)&(dist<=300)
            if min(inner.sum(),outer.sum())<30:return {'status':'insufficient_ceiling_support','inner_pixels':int(inner.sum()),'outer_pixels':int(outer.sum())}
            logs=relative_logs(rgb,backgrounds['same_panel_right'])
            dl=np.median(Lab[outer],axis=0)-np.median(Lab[inner],axis=0)
            differences=[]
            for c in range(3):
                i=logs[...,c][inner];o=logs[...,c][outer]
                differences.append(float(np.median(o)-np.median(i)) if np.isfinite(i).all() and np.isfinite(o).all() else None)
            return {'status':'evaluated','inner_pixels':int(inner.sum()),'outer_pixels':int(outer.sum()),
                'outer_minus_inner_Lab':dl.tolist(),'outer_minus_inner_output_logs':differences,
                'meaning':'same-surface spatial contrast; background cancels, not a physical halo profile'}
        # Native and common 25um averaging are measurement controls, not rendered candidates.
        for scale,sigma in [('native',0),('25um',25/um)]:
            rgbs={k:gaussian_filter(v,(sigma,sigma,0)) if sigma else v for k,v in images.items()};labs={k:lab(v) for k,v in rgbs.items()}
            cases=[]
            for bgname,bg in backgrounds.items():
                states={key:summary(labs[key],rgbs[key],bg,target) for key in images}
                common=target&(np.linalg.norm(labs['real'][...,1:],axis=-1)>=5)&(np.linalg.norm(labs['halo'][...,1:],axis=-1)>=5)
                common_hue={'pixels':int(common.sum()),'target_fraction':float(common.sum()/target.sum()),'states':{}}
                for key in ['real','halo']:
                    base_hue=states[key]['background_hue_deg']
                    hues=np.degrees(np.arctan2(labs[key][...,2],labs[key][...,1]))[common]
                    angle,concentration=circular_average(hues-base_hue) if base_hue is not None else (None,None)
                    common_hue['states'][key]={'circular_mean_shift_deg':angle,'resultant_length':concentration}
                rings=[]
                for lo,hi in [(0,100),(100,200),(200,300)]:
                    m=target&(dist>lo)&(dist<=hi)
                    if m.sum()<30:continue
                    rings.append({'distance_um':[lo,hi],'states':{key:summary(labs[key],rgbs[key],bg,m) for key in images}})
                cases.append({'background':bgname,'states':states,'common_chromatic_hue_comparison':common_hue,'rings':rings})
            reg['scales'][scale]=cases
            reg.setdefault('spatial_shape',{})[scale]={key:spatial_shape(labs[key],rgbs[key]) for key in images}
            if scale=='25um':
                bg=backgrounds['same_panel_right'];saved={**images,'target':target,**{'bg_'+k:v for k,v in backgrounds.items()},'distance_um':dist}
                fields={key:contrast_lab(labs[key],bg) for key in images}
                logs={key:relative_logs(rgbs[key],bg) for key in images}
                saved.update({key+'_deltaLab':v for key,v in fields.items()});saved.update({key+'_relative_output_logs':v for key,v in logs.items()})
                # Every shift is reported; none selected as improved registration.
                # Structural residuals are 2..7 px: cover them conservatively,
                # including diagonal shifts. None becomes a new fitted image.
                for dy,dx in [(0,0),(-8,0),(8,0),(0,-8),(0,8),(-8,-8),(-8,8),(8,-8),(8,8)]:
                    shifted={key:gaussian_filter(shift(value,(dy,dx,0),order=1,mode='nearest'),(sigma,sigma,0)) for key,value in images.items() if key!='real'}
                    result={key:summary(lab(value),value,bg,target) for key,value in shifted.items()}
                    reg['registration_sensitivity'].append({'shift_yx':[dy,dx],'states':result,
                        'spatial_shape':{key:spatial_shape(lab(value),value) for key,value in shifted.items()}})
                np.savez_compressed(OUT/(name+'.npz'),**saved)
                # Profiles along same observed ceiling strip; references labelled.
                strip=rectangle(dist.shape,c['surface_strip']);labels=['delta L*','delta a*','delta b*','relative log2 R/G','relative log2 B/G']
                for col in range(5):
                    ax=apro[row,col]
                    for key,color in [('real','black'),('nohalo','#3b80b8'),('halo','#d78b27')]:
                        # Counterfactual red is zero or nearly zero in some
                        # backgrounds. Keep it in the numeric record, but do
                        # not present unstable log ratios as colour evidence.
                        if col>=3 and key=='nohalo':continue
                        values=fields[key][...,col] if col<3 else logs[key][...,col-2]
                        xx=[];yy=[]
                        for x in range(c['surface_strip'][0],c['surface_strip'][2],4):
                            m=strip.copy();m[:,:x]=False;m[:,x+4:]=False
                            xx.append(x);yy.append(np.median(values[m]))
                        ax.plot(xx,yy,label=key,color=color)
                    ax.axhline(0,color='gray',lw=.6);ax.set_title(name+' / '+labels[col],fontsize=8);ax.set_xlabel('Fixed crop x (ceiling strip)')
                    if col==0:ax.legend(fontsize=7)
        for col,key in enumerate(images):
            ax=axes[row,col];ax.imshow(np.clip(srgb_encode(images[key]),0,1),interpolation='nearest')
            x0,y0,x1,y1=c['target'];ax.add_patch(Rectangle((x0,y0),x1-x0,y1-y0,fill=False,ec='orange',lw=1.3));ax.text(x0,y0-4,'ceiling target',color='orange',fontsize=6)
            for j,(bgname,box) in enumerate(c['backgrounds'].items()):
                x0,y0,x1,y1=box;ax.add_patch(Rectangle((x0,y0),x1-x0,y1-y0,fill=False,ec=['lime','magenta','yellow'][j],lw=1));ax.text(x0,y0-4,bgname,color=['lime','magenta','yellow'][j],fontsize=5)
            ax.set_title(name+' / '+key,fontsize=9);ax.set_xticks([]);ax.set_yticks([])
        report['regions'].append(reg)
    fig.suptitle('Original fixed-cache photos; orange=ceiling target, green=primary local reference\nOther similar-finish references test background sensitivity; none is a measured unlit control',fontsize=10)
    fig.savefig(OUT/'reference_regions.png',dpi=180);plt.close(fig)
    fprof.suptitle('Local ceiling contrast relative to the SAME-IMAGE right reference, 25um measurement scale\nLog panels: real/full only (nohalo has clipped/near-zero channels); no image correction',fontsize=11)
    fprof.savefig(OUT/'local_profiles.png',dpi=180);plt.close(fprof)
    (OUT/'results.json').write_text(json.dumps(report,indent=2));print('saved',OUT)


if __name__=='__main__':main()
