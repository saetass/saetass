"""Mesh-free evaluation of the solver: interpolation to arbitrary radii, against a much finer reference."""
import numpy as np
from saetass.pinn.reference import solve
from saetass.pinn.wind_bubble import DimensionlessBubble as B
rng=np.random.default_rng(0); rows={}
for b in [B(22.,250.,6.4,3.5),B(1.,270.,6.4,3.5),B(100.,500.,15.,2.),B(.1,100.,3.,6.)]:
  for t in (10.,300.):
    f=solve(b,np.array([t]),resolution=8); q=np.sort(rng.uniform(0.02,b.outer_radius*.99,4000))
    ref=np.exp(np.interp(q,f.x,np.log(np.maximum(f.psi[0],1e-300))))
    for res in (1,2,4):
        s=solve(b,np.array([t]),resolution=res)
        with np.errstate(divide='ignore'):
            lg=np.exp(np.interp(q,s.x,np.log(np.maximum(s.psi[0],1e-300)))); li=np.interp(q,s.x,s.psi[0])
        m=ref>1e-6*ref.max(); rows.setdefault(res,[]).append((np.linalg.norm((lg-ref)[m])/np.linalg.norm(ref[m]),np.abs(lg/ref-1)[m].max(),np.linalg.norm((li-ref)[m])/np.linalg.norm(ref[m])))
for res,v in rows.items():
    v=np.array(v); print(f'res {res} ({300*res+700*res+400*res+1} nodes): L2 log-interp median {np.median(v[:,0]):.1e} worst {v[:,0].max():.1e}; max rel worst {v[:,1].max():.2f}; linear-interp L2 worst {v[:,2].max():.1e}')
