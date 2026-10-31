"""SAETASS as the final, exact stage: posterior from the GPU solver, corrected by importance with real SAETASS runs in a sweep."""
import sys, json, time, math, numpy as np, torch
from gen import *
import bubblefn as B
S=B.S
import astropy.units as u, astropy.constants as const
from saetass.cli import set_log_level
from saetass.sweep import run_sweep
from saetass import units as su
from saetass.pinn.confirm import simulate as run_solver
set_log_level("ERROR")
RAD=np.linspace(16.,110.,S.N_RADII)
def one(task):
    th,e=task; L,M,rho,eta=10.0**np.asarray(th); r=np.linspace(0.,S.R_MAX,1200)
    psi=run_solver(r*u.pc,e*u.GeV,S.AGE*u.Myr,2*int(6000+9000*np.log10(e)),L_wind=L*u.erg/u.s,M_dot=M*const.M_sun/u.yr,rho_0=rho*u.g/u.cm**3,t_b=S.AGE*u.Myr,E_max=1*u.PeV,eta_B=eta).to_value(su.PSI_P)
    return np.interp(RAD,r,np.log(np.maximum(psi,1e-300)))
if __name__=="__main__":
    K=int(sys.argv[1]); torch.manual_seed(0); data=np.load('mock.npy'); LD=torch.as_tensor(np.log(data),device=dev,dtype=DD).reshape(-1)
    low=torch.as_tensor(S.PRIOR[:,0],device=dev,dtype=DD); high=torch.as_tensor(S.PRIOR[:,1],device=dev,dtype=DD); q0=torch.as_tensor(RAD/S.R_MAX,device=dev,dtype=DD); fw=Forward(1024,12)
    @torch.no_grad()
    def out(theta,chunk=1024):
        r=[]; ok=[]
        for th in theta.split(chunk):
            n=th.shape[0]; D,v,F,t,lnu,ts=B.fields(th,fw); p=fw.psi(D,v,F,t); r.append((fw.sample(p,q0[None].expand(n*B.NE,-1))+lnu).reshape(n,-1)); ok.append(ts.reshape(n,B.NE)[:,0]<.3)
        return torch.cat(r),torch.cat(ok)
    def lp(th):
        o,ok=out(th); chi=(((o-LD)/S.NOISE)**2).sum(1); ins=(th>=low).all(1)&(th<=high).all(1)&ok&torch.isfinite(chi); return torch.where(ins,-.5*chi,torch.full_like(chi,-torch.inf))
    start=low+(high-low)*torch.rand(512,4,dtype=DD,device=dev); a=time.perf_counter(); chain,vals=S.stretch_sampler(lp,start,400); gpu=time.perf_counter()-a
    th=chain[200:].reshape(-1,4); v=vals[200:].reshape(-1); g=torch.isfinite(v); th,v=th[g],v[g]; print(f'GPU posterior {gpu:.0f}s mean {[round(x,4) for x in th.mean(0).tolist()]} std {[round(x,4) for x in th.std(0).tolist()]}',flush=True)
    idx=torch.randperm(th.shape[0],device=dev)[:K]; pts=th[idx]; base=v[idx]
    tasks=[(p,e) for p in pts.cpu().numpy() for e in S.ENERGIES]
    a=time.perf_counter(); res=run_sweep(one,tasks,max_workers=16); sw=time.perf_counter()-a
    Y=np.array(res).reshape(K,-1); ld=np.log(data).reshape(-1); ll=-.5*(((Y-ld)/S.NOISE)**2).sum(1); lw=ll-base.cpu().numpy(); lw-=lw.max(); w=np.exp(lw); e=w.sum()**2/(w**2).sum(); w/=w.sum(); P=pts.cpu().numpy(); m=(w[:,None]*P).sum(0); s=np.sqrt((w[:,None]*(P-m)**2).sum(0))
    r=dict(K=K,runs=K*4,sweep_seconds=sw,gpu_seconds=gpu,ess=float(e),mean=m.tolist(),std=s.tolist(),pull=((m-S.TRUTH)/s).tolist(),rms_lnpsi_diff=float(np.sqrt(np.mean((Y-out(pts)[0].cpu().numpy())**2))))
    print(json.dumps(r)); open('E20_results.jsonl','a').write(json.dumps(r)+'\n')
