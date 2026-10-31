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
for N in (6,8,16,24,32):
    p=psi_at(lo,ma,up,F,t,N); e=torch.linalg.norm(p-ref,dim=1)/torch.linalg.norm(ref,dim=1)
    y,logs=psi_sym(lo,ma,up,F,t,N); q=y*torch.exp(logs.clamp(max=700)); e2=torch.linalg.norm(q-ref,dim=1)/torch.linalg.norm(ref,dim=1)
    print(f'N={N}: plain median {e.median():.2e}, >1%: {(e>.01).float().mean()*100:.2f} %, >100 %: {(e>1).sum().item()} of {b}; symmetrized >1%: {(e2>.01).float().mean()*100:.2f} %, >100 %: {(e2>1).sum().item()}, nan {torch.isnan(e2).sum().item()}',flush=True)
p=psi_at(lo,ma,up,F,t,24); e=torch.linalg.norm(p-ref,dim=1)/torch.linalg.norm(ref,dim=1); bad=torch.argsort(e,descending=True)[:6]
h=x[1]-x[0]; Dm=D[:,:n-1]; cell=(v[:,:n-1].abs()*h/Dm).amax(1)
print('worst: err, t, max cell Peclet, min D, max v'); 
for i in bad: print(f'  {e[i]:.1e} t={t[i]:.3g} cellPe={cell[i]:.1f} Dmin={D[i].min():.2e} vmax={v[i].max():.2f}')
print('cell Peclet of all: median',cell.median().item(),'p99',cell.quantile(.99).item())
for lo_,hi_ in [(0,1),(1,10),(10,100),(100,1e9)]:
    m=(cell>=lo_)&(cell<hi_); print(f' cellPe in [{lo_},{hi_}): n={m.sum().item()}, err>1%: {(e[m]>.01).float().mean()*100:.2f} %')
