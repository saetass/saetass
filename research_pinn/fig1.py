import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
d=np.load('logs/round3/inv_dense.npz'); r=d['x']
fig,ax=plt.subplots(figsize=(8,4.4),facecolor='#fcfcfb'); ax.set_facecolor('#fcfcfb')
ax.fill_between(r,np.exp(d['low']),np.exp(d['high']),color='#eb6834',alpha=.25,lw=0,label='Banda 68 %, 32 realizaciones de ruido')
ax.plot(r,np.exp(d['truth']),color='#2a78d6',lw=2,label='D(r) verdadero')
ax.plot(r,np.exp(d['median']),color='#eb6834',lw=2,ls='--',label='D(r) recuperado, sin suponer forma')
ax.set_yscale('log'); ax.set_xlim(0,200); ax.set_xlabel('Radio (pc)'); ax.set_ylabel('D (unidades adimensionales)')
ax.set_title('Inversión de D(r) a través del solver diferenciable',loc='left',fontsize=12)
for s in ('top','right'): ax.spines[s].set_visible(False)
ax.grid(alpha=.15); ax.legend(frameon=False,loc='lower right'); fig.tight_layout(); fig.savefig('fig_function_inference.png',dpi=130,facecolor=fig.get_facecolor())
