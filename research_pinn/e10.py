"""Experiment F: the parameter search of validation/cases/pinn_parameter_search.py with the GPU solver instead of the network.
usage: e10.py res walkers steps [N]"""
import sys, json, time, math, numpy as np, torch
import astropy.units as u
sys.path.insert(0,'/home/cpozogonzalez/saetass/validation/cases')
import pinn_parameter_search as S
from saetass.cli import set_log_level
from saetass.pinn.wind_bubble import FastScales
from gpusolve import psi_at
src=open('/tmp/claude-1008/-home-cpozogonzalez-saetass/9a06d848-4bb5-4ea1-994a-78c0e4527b54/scratchpad/ops/e2.py').read().split('# speed')[0]
exec(src)   # assemble(th,res)
res,walkers,steps=int(sys.argv[1]),int(sys.argv[2]),int(sys.argv[3]); N=int(sys.argv[4]) if len(sys.argv)>4 else 24
set_log_level("ERROR"); rng=np.random.default_rng(0); torch.manual_seed(0)
radii=np.linspace(16.,110.,S.N_RADII); data=[]; 
for energy in S.ENERGIES:
    psi,_=S.simulate(S.TRUTH,energy,radii); data.append(psi*np.exp(S.NOISE*rng.standard_normal(S.N_RADII)))
data=np.array(data)
fast=FastScales(1*u.PeV); E=torch.as_tensor(S.ENERGIES,device=dev,dtype=DD); P=(E**2+2*E*fast.mc2)**.5
RR=torch.as_tensor(radii,device=dev,dtype=DD); LD=torch.as_tensor(np.log(data),device=dev,dtype=DD)
low=torch.as_tensor(S.PRIOR[:,0],device=dev,dtype=DD); high=torch.as_tensor(S.PRIOR[:,1],device=dev,dtype=DD)
@torch.no_grad()
def logp(theta):
    n=theta.shape[0]; v=10.**theta; L,M,rho,eta=[v[:,None,i] for i in range(4)]
    sc=fast(P[None,:],L,M,rho,torch.full_like(L,S.AGE),torch.full_like(L,S.R_MAX),eta_B=eta)
    Pe,lam,a,b=[getattr(sc,k).expand(n,P.numel()).reshape(-1) for k in ('peclet','diffusion_ratio','bubble_ratio','domain_ratio')]
    ok=(b>1.05)&(a>1.05)&(theta>=low).all(1).repeat_interleave(P.numel())
    th=torch.stack([Pe.log(),lam.log(),a.log(),b.clamp(min=1.05).log()],1); lo,ma,up,F,x=assemble(th,res)
    t=(S.AGE/sc.time_unit).expand(n,P.numel()).reshape(-1)
    psi=psi_at(lo,ma,up,F,t,N).clamp(min=1e-300)
    q=(RR[None]/sc.R_TS.expand(n,P.numel()).reshape(-1,1)).expand(n*P.numel(),-1)
    j=torch.searchsorted(x[:,:-1],q).clamp(1,psi.shape[1]-1); xa,xb=x.gather(1,j-1),x.gather(1,j); fa,fb=psi.gather(1,j-1).log(),psi.gather(1,j).log(); w=(q-xa)/(xb-xa)
    lp=fa+w*(fb-fa)+sc.psi_unit.expand(n,P.numel()).reshape(-1,1).log()
    chi=(((lp.reshape(n,P.numel(),-1)-LD)/S.NOISE)**2).sum((1,2)); inside=(theta>=low).all(1)&(theta<=high).all(1)&ok.reshape(n,-1).all(1)
    return torch.where(inside,-.5*chi,torch.full_like(chi,-torch.inf))
start=low+(high-low)*torch.rand(walkers,4,dtype=DD,device=dev)
torch.cuda.synchronize(); t0=time.perf_counter(); chain,vals=S.stretch_sampler(logp,start,steps); torch.cuda.synchronize(); el=time.perf_counter()-t0
set_=chain[steps//2:].reshape(-1,4); g=vals[steps//2:].reshape(-1); f=torch.isfinite(g); best=set_[f][g[f].argmax()].cpu().numpy(); mean=set_[f].mean(0).cpu().numpy(); std=set_[f].std(0).cpu().numpy()
fit=np.array([S.simulate(best,e,radii)[0] for e in S.ENERGIES]); chi=float(np.sum((np.log(fit/data)/S.NOISE)**2)); tr=np.array([S.simulate(S.TRUTH,e,radii)[0] for e in S.ENERGIES]); chit=float(np.sum((np.log(tr/data)/S.NOISE)**2))
r=dict(res=res,walkers=walkers,steps=steps,seconds=el,profiles=walkers*steps*4,ms_per_profile=el/(walkers*steps*4)*1e3,mean=mean.tolist(),std=std.tolist(),pull=((mean-S.TRUTH)/std).tolist(),chi2_best=chi,chi2_truth=chit)
print(json.dumps(r)); open('F_results.jsonl','a').write(json.dumps(r)+'\n')
