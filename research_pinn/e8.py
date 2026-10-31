"""Is the FNO on arbitrary profiles independent of the mesh? And what does the exact solver cost at 256 nodes? usage: e8.py checkpoint"""
import sys, math, time, numpy as np, torch
import e3lib
from e3lib import *
# timing of the exact solver and of one network pass on this grid, batch 1024
D,v,Q=random_family(1024); lo,ma,up,F=operator(D,v,Q); t=torch.full((1024,),10.,device=dev,dtype=DD)
for N in (16,24):
    torch.cuda.synchronize(); a=time.perf_counter(); psi_at(lo,ma,up,F,t,N); torch.cuda.synchronize(); print('contour N',N,'ms/profile',(time.perf_counter()-a)/1024*1e3)
torch.cuda.synchronize(); a=time.perf_counter(); pcr(-lo,-ma,-up,F); torch.cuda.synchronize(); print('one real solve ms/profile',(time.perf_counter()-a)/1024*1e3)
m=FNO().to(dev); inp=torch.rand(1024,5,n-1,device=dev).float()
with torch.no_grad():
    m(inp); torch.cuda.synchronize(); a=time.perf_counter(); m(inp); torch.cuda.synchronize(); print('FNO pass ms/profile',(time.perf_counter()-a)/1024*1e3)
