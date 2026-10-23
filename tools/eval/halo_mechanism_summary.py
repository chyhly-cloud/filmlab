"""Plot fixed geometry profiles for predefined mechanism controls, without fitting."""
from pathlib import Path
import json
import numpy as np
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=Path('/workspace/filmlab-audit/halo-mechanisms/R1-02-3');report=json.loads((root/'results.json').read_text())
    keys=['real','baseline','red_only','first_return','independent_paper'];colors=['black','#d57a24','#d62728','#377eb8','#4daf4a']
    fig,axes=plt.subplots(3,3,figsize=(13,10),layout='constrained')
    for col,r in enumerate(report['regions']):
        p=r['radial_profiles'];x=[np.mean(v['radius']) for v in p]
        for row in range(3):
            ax=axes[row,col]
            for key,color in zip(keys,colors):ax.plot(x,[v['median_Lab'][key][row] for v in p],label=key,color=color)
            ax.set_title(r['name'] if row==0 else '');ax.set_xlabel('Nominal distance outside core/mask (µm)');ax.set_ylabel(['L*','a*','b*'][row])
            if row==0 and col==0:ax.legend(fontsize=7)
    fig.suptitle('Fixed timing / 50 µm median rings / unflagged surroundings; interventions are diagnostic controls')
    fig.savefig(root/'radial_profiles.png',dpi=160)
    # Keep the final full-resolution colour comparison readable at four panels.
    from PIL import Image,ImageDraw,ImageFont
    from filmlab.spectral import srgb_encode
    a=np.load('/workspace/filmlab-audit/printing-density/ceiling_lamp.npz')
    sheet=Image.new('RGB',(2048,562),(20,20,20));draw=ImageDraw.Draw(sheet)
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16)
    for i,(key,label) in enumerate([('real','Real scan'),('matrix_exp','Existing exp_plain'),('constrained_exp','Experimental neutral constraint'),('exact_exp','Experimental spectral integral')]):
        rgb=np.uint8(np.clip(srgb_encode(np.nan_to_num(a[key]))*255+.5,0,255))
        sheet.paste(Image.fromarray(rgb),(512*i,50));draw.text((512*i+8,12),label,font=font,fill='white')
    sheet.save(root.parent/'printing_compare.png')



if __name__=='__main__':main()
