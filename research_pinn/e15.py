import math, time, numpy as np, torch
from e3lib import *
D,v,Q=random_family(1024); lo,ma,up,F=operator(D,v,Q); t=torch.full((1024,),10.,device=dev,dtype=DD)
def T(f,r=5):
    torch.cuda.synchronize(); f(); torch.cuda.synchronize(); a=time.perf_counter()
    for _ in range(r): f()
    torch.cuda.synchronize(); return (time.perf_counter()-a)/r/1024*1e3
print('one real solve',T(lambda:pcr(-lo,-ma,-up,F)))
for N in (4,6,8,16,24): print('contour N',N,T(lambda:psi_at(lo,ma,up,F,t,N)))
m=FNO().to(dev); inp=torch.rand(1024,5,n-1,device=dev).float()
with torch.no_grad(): print('FNO pass',T(lambda:m(inp))); 
m2=torch.compile(m) if False else m
