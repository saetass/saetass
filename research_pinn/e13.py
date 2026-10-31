"""How few contour nodes does the same accuracy as the network need? Against N=32 on the same grid."""
import math, time, numpy as np, torch
from e3lib import *
torch.manual_seed(5)
for name,fam in (('random',random_family),('bubble',bubble_family)):
    D,v,Q=fam(2048); lo,ma,up,F=operator(D,v,Q); t=torch.exp(math.log(1e-2)+math.log(1e4)*torch.rand(2048,device=dev,dtype=DD))
    ref=psi_at(lo,ma,up,F,t,32).clamp(min=0)
    print(name)
    for N in (2,4,6,8,10,12,16,20,24):
        torch.cuda.synchronize(); a=time.perf_counter(); p=psi_at(lo,ma,up,F,t,N); torch.cuda.synchronize(); c=(time.perf_counter()-a)/2048*1e3
        p=p.clamp(min=0); e=(torch.linalg.norm(p-ref,dim=1)/torch.linalg.norm(ref,dim=1)).cpu().numpy()
        print(f'  N={N:2d} ({N//2} complex solves): {c:.4f} ms/profile L2 median {np.median(e):.2e} p95 {np.percentile(e,95):.2e} max {e.max():.2e}',flush=True)
