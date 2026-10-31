import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
d=np.load('function_inference_k160.npz'); R=300.; r=d['x']*R
fig,ax=plt.subplots(figsize=(8,4.4),facecolor='#fcfcfb'); ax.set_facecolor('#fcfcfb')
ax.fill_between(r,np.exp(d['lo'])*1,np.exp(d['hi']),color='#eb6834',alpha=.25,lw=0,label='Banda 68 %, 16 realizaciones de ruido')
ax.plot(r,np.exp(d['true']),color='#2a78d6',lw=2,label='D(r) verdadero')
ax.plot(r,np.exp(d['med']),color='#eb6834',lw=2,ls='--',label='D(r) recuperado, sin suponer forma')
ax.set_yscale('log'); ax.set_xlim(0,200); ax.set_xlabel('Radio (pc)'); ax.set_ylabel('D (unidades adimensionales)')
ax.set_title('Inversión de D(r) a través del solver diferenciable',loc='left',fontsize=12)
for s in ('top','right'): ax.spines[s].set_visible(False)
ax.grid(alpha=.15); ax.legend(frameon=False,loc='lower right'); fig.tight_layout(); fig.savefig('fig_function_inference.png',dpi=130,facecolor=fig.get_facecolor())
