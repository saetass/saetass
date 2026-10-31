"""Coarse to fine: tempered coarse sampling, linear discrepancy fitted with a few fine runs, exact importance correction.
usage: e18.py nc Nc nf Nf"""
import sys, json, time, math, numpy as np, torch
from gen import *
import bubblefn as B
S=B.S
from saetass.cli import set_log_level
set_log_level("ERROR"); torch.manual_seed(0)
nc,Nc,nf,Nf=map(int,sys.argv[1:5]); radii=np.linspace(16.,110.,S.N_RADII); data=np.load('mock.npy')
LD=torch.as_tensor(np.log(data),device=dev,dtype=DD).reshape(-1); low=torch.as_tensor(S.PRIOR[:,0],device=dev,dtype=DD); high=torch.as_tensor(S.PRIOR[:,1],device=dev,dtype=DD); q0=torch.as_tensor(radii/S.R_MAX,device=dev,dtype=DD)
@torch.no_grad()
def out(theta,fw,chunk=1024):
    r=[]; ok=[]
    for th in theta.split(chunk):
        n=th.shape[0]; D,v,F,t,lnu,ts=B.fields(th,fw); p=fw.psi(D,v,F,t); r.append((fw.sample(p,q0[None].expand(n*B.NE,-1))+lnu).reshape(n,-1)); ok.append((ts.reshape(n,B.NE)[:,0]<.3))
    return torch.cat(r),torch.cat(ok)
def inside(th): return (th>=low).all(1)&(th<=high).all(1)
def logpost(lp,ok,th,sigma):
    chi=(((lp-LD)/sigma)**2).sum(1); return torch.where(inside(th)&ok&torch.isfinite(chi),-.5*chi,torch.full_like(chi,-torch.inf))
coarse,fine=Forward(nc,Nc),Forward(nf,Nf); count=dict(coarse=0,fine=0); W,STEPS=2048,400
def sample(fn,start):
    chain,vals=S.stretch_sampler(fn,start,STEPS); return chain[STEPS//2:].reshape(-1,4),vals[STEPS//2:].reshape(-1)
def stage(sigma,delta,start):
    def lp(th):
        o,ok=out(th,coarse); count['coarse']+=th.shape[0]*B.NE
        return logpost(o+delta(th),ok,th,sigma)
    return sample(lp,start)
def ess(lw):
    lw=lw-lw.max(); w=torch.exp(lw); return (w.sum()**2/(w**2).sum()).item(),w/w.sum()
def fit(th,lpf,lpc):
    """linear discrepancy per observation, ridge"""
    m=th.mean(0); sc=th.std(0)+1e-9; X=torch.cat([torch.ones(th.shape[0],1,device=dev,dtype=DD),(th-m)/sc],1); Y=lpf-lpc; good=torch.isfinite(Y).all(1)&torch.isfinite(X).all(1); X,Y=X[good],Y[good]
    A=X.T@X+1e-6*X.shape[0]*torch.eye(5,device=dev,dtype=DD); b=torch.linalg.solve(A,X.T@Y)
    return lambda t:(torch.cat([torch.ones(t.shape[0],1,device=dev,dtype=DD),(t-m)/sc],1)@b)
t0=time.perf_counter(); res=dict(nc=nc,Nc=Nc,nf=nf,Nf=Nf); delta=lambda t:0.
start=low+(high-low)*torch.rand(W,4,dtype=DD,device=dev)
SIG=[S.NOISE*6,S.NOISE*2,S.NOISE]
for it,sig in enumerate(SIG):
    th,val=stage(sig,delta,start); g=torch.isfinite(val); th=th[g]; idx=torch.randperm(th.shape[0],device=dev)[:256]; pts=th[idx]
    lc,_=out(pts,coarse); lf,_=out(pts,fine); count['fine']+=256*B.NE; count['coarse']+=256*B.NE
    prev=delta; delta=fit(pts,lf,lc); start=th[torch.randperm(th.shape[0],device=dev)[:W]]
    print(f'round {it} sigma {sig:.3f}: mean {[round(x,3) for x in th.mean(0).tolist()]} std {[round(x,4) for x in th.std(0).tolist()]}  discrepancy rms {(lf-lc).pow(2).mean().sqrt():.3f} after fit {(lf-lc-delta(pts)).pow(2).mean().sqrt():.4f} t={time.perf_counter()-t0:.0f}s',flush=True)
th,val=stage(S.NOISE,delta,start); g=torch.isfinite(val); th,val=th[g],val[g]
K=1024; idx=torch.randperm(th.shape[0],device=dev)[:K]; pts=th[idx]; lf,ok=out(pts,fine); count['fine']+=K*B.NE; lw=logpost(lf,ok,pts,S.NOISE)-val[idx]; e,w=ess(lw)
mean=(w[:,None]*pts).sum(0); std=((w[:,None]*(pts-mean)**2).sum(0)).sqrt(); best=pts[(logpost(lf,ok,pts,S.NOISE)).argmax()].cpu().numpy()
el=time.perf_counter()-t0
fitp=np.array([S.simulate(best,e_,radii)[0] for e_ in S.ENERGIES]); chi=float(np.sum((np.log(fitp/data)/S.NOISE)**2))
res.update(seconds=el,ess=e,K=K,mean=mean.tolist(),std=std.tolist(),pull=((mean.cpu().numpy()-S.TRUTH)/std.cpu().numpy()).tolist(),coarse_profiles=count['coarse'],fine_profiles=count['fine'],chi2_best_saetass=chi)
print(json.dumps(res)); open('S_results.jsonl','a').write(json.dumps(res)+'\n')
