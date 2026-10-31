"""Outliers of the contour on arbitrary profiles: how many, why, and whether symmetrizing the operator cures them."""
import math, time, numpy as np, torch
from e3lib import *
from gpusolve import psi_sym
s=open('e6.py').read(); 
torch.manual_seed(5); b=8192
D,v,Q=random_family(b); lo,ma,up,F=operator(D,v,Q); t=torch.exp(math.log(1e-2)+math.log(1e4)*torch.rand(b,device=dev,dtype=DD))
def step_ref(lo,ma,up,F,t,M=400):
    g=2-2**.5; s0=torch.log(t)-7.; ts=torch.exp(s0[:,None]+7.*torch.linspace(0,1,M+1,device=dev,dtype=DD)[None]); psi=0*F
    for k in range(M):
        dt=(ts[:,k+1]-ts[:,k])[:,None]; sh=.5*g*dt
        Ap=ma*psi+torch.cat([0*psi[:,:1],lo[:,1:]*psi[:,:-1]],1)+torch.cat([up[:,:-1]*psi[:,1:],0*psi[:,:1]],1)
        st=pcr(-sh*lo,1-sh*ma,-sh*up,psi+sh*(Ap+2*F)); w=1/(g*(2-g)); c=(1-g)/(2-g)*dt
        psi=pcr(-c*lo,1-c*ma,-c*up,w*st-(1-g)**2*w*psi+c*F)
    return psi
ref=step_ref(lo,ma,up,F,t)
print('reference: TR-BDF2, 400 steps; min/max',(ref.amin(1)/ref.amax(1)).min().item())

ss=pcr(-lo,-ma,-up,F)
for N in (8,16,24):
    a=time.perf_counter(); p=psi_at(lo,ma,up,F,t,N); e=torch.linalg.norm(p-ref,dim=1)/torch.linalg.norm(ref,dim=1)
    pm=p.amax(1).abs()+1e-300
    # maximum principle: 0 <= psi <= psi_ss, at nodes where psi is relevant
    over=((p-ss).amax(1)/pm); under=(-p.amin(1)/pm); nan=~torch.isfinite(e)
    for tol in (1e-3,1e-2):
        flag=(over>tol)|(under>tol)|nan
        badm=(e>.01)|nan
        print(f'N={N} tol={tol}: flagged {flag.float().mean()*100:.2f} %; bad(>1 %) {badm.float().mean()*100:.2f} %; bad caught {(flag&badm).sum().item()} of {badm.sum().item()}; worst unflagged error {e[~flag].max():.3e}',flush=True)
# repair: flagged profiles recomputed by stepping
p=psi_at(lo,ma,up,F,t,16); e=torch.linalg.norm(p-ref,dim=1)/torch.linalg.norm(ref,dim=1); pm=p.amax(1).abs()+1e-300
flag=(((p-ss).amax(1)/pm)>1e-3)|((-p.amin(1)/pm)>1e-3)|~torch.isfinite(e)
print('after replacing the flagged ones: all profiles error below',e[~flag].max().item(),' fraction to recompute',flag.float().mean().item())
