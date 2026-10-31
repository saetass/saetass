"""Coarse to fine parameter search with exact correction. usage: e17.py"""
import sys, json, time, math, numpy as np, torch
from gen import *
import bubblefn as B
S=B.S
from saetass.cli import set_log_level
set_log_level("ERROR"); torch.manual_seed(0); rng=np.random.default_rng(0)
radii=np.linspace(16.,110.,S.N_RADII); data=[]
for energy in S.ENERGIES:
    psi,_=S.simulate(S.TRUTH,energy,radii); data.append(psi*np.exp(S.NOISE*rng.standard_normal(S.N_RADII)))
data=np.array(data); np.save('mock.npy',data)
LD=torch.as_tensor(np.log(data),device=dev,dtype=DD); low=torch.as_tensor(S.PRIOR[:,0],device=dev,dtype=DD); high=torch.as_tensor(S.PRIOR[:,1],device=dev,dtype=DD); q0=torch.as_tensor(radii/S.R_MAX,device=dev,dtype=DD)
@torch.no_grad()
def loglike(theta,fw,sigma=S.NOISE,chunk=2048):
    out=[]
    for th in theta.split(chunk):
        n=th.shape[0]; D,v,F,t,lnu,ts=B.fields(th,fw); p=fw.psi(D,v,F,t); lp=fw.sample(p,q0[None].expand(n*B.NE,-1))+lnu
        chi=(((lp.reshape(n,B.NE,-1)-LD)/sigma)**2).sum((1,2)); inside=(th>=low).all(1)&(th<=high).all(1)&(ts.reshape(n,B.NE)[:,0]<.3)
        out.append(torch.where(inside&torch.isfinite(chi),-.5*chi,torch.full_like(chi,-torch.inf)))
    return torch.cat(out)
def timeit(fw,th):
    torch.cuda.synchronize(); a=time.perf_counter(); loglike(th,fw); torch.cuda.synchronize(); return (time.perf_counter()-a)/(th.shape[0]*B.NE)*1e3
th0=low+(high-low)*torch.rand(1024,4,dtype=DD,device=dev)
fine=Forward(4096,16)
true=torch.as_tensor(S.TRUTH,device=dev,dtype=DD)[None]
print('fine at truth: chi2',-2*loglike(true,fine).item(),' ms/profile',timeit(fine,th0),flush=True)
for n,N in [(2048,16),(1024,12),(512,8),(256,6),(128,4)]:
    c=Forward(n,N); print(f'n={n} N={N}: chi2 at truth {-2*loglike(true,c).item():.1f}, ms/profile {timeit(c,th0):.4f}',flush=True)
