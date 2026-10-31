"""Function-space inference: recover D(r) with no assumed form, through the differentiable solver. usage: e19.py n_real noise"""
import sys, json, time, math, numpy as np, torch
from gen import *
import bubblefn as B
S=B.S
from saetass.cli import set_log_level
set_log_level("ERROR"); torch.manual_seed(0); rng=np.random.default_rng(1)
NR,NOISE=int(sys.argv[1]),float(sys.argv[2]); KN=int(sys.argv[5]) if len(sys.argv)>5 else 40; ITER=int(sys.argv[3]) if len(sys.argv)>3 else 800
radii=np.geomspace(3.,200.,40)
clean=np.array([S.simulate(S.TRUTH,e,radii)[0] for e in S.ENERGIES]); print('mock done',flush=True)
fw=Forward(1024,12); true=torch.as_tensor(S.TRUTH,device=dev,dtype=DD)[None]
with torch.no_grad():
    D0,v0,F0,t0,lnu0,ts0=B.fields(true,fw)
ne=B.NE; v=v0[:1]; F=F0[:1]; t=t0[:1]; lnu=lnu0; 
Dt=D0                                        # (ne,n): true D for each energy
scale=(Dt/Dt[:1]).mean(1,keepdim=True)        # energy scaling of the coefficient, treated as known
ln_true=Dt[:1].log()[0]
xk=torch.linspace(0,1,KN,device=dev,dtype=DD)  # knots, uniform, placed without knowing the jumps
idx=((fw.x*(KN-1)).floor().long()).clamp(0,KN-2); w=(fw.x*(KN-1)-idx)
def Dof(u):
    """u (b,KN) -> ln D on the nodes, piecewise linear"""
    return u[:,idx]*(1-w)+u[:,idx+1]*w
q0=torch.as_tensor(radii/S.R_MAX,device=dev,dtype=DD)
noise=torch.as_tensor(rng.standard_normal((NR,)+clean.shape),device=dev,dtype=DD)
LD=torch.as_tensor(np.log(clean),device=dev,dtype=DD)[None]+NOISE*noise
u=(ln_true.mean()*torch.ones(NR,KN,device=dev,dtype=DD)).requires_grad_(True)
def forward(u):
    b=u.shape[0]; lnD=Dof(u)[:,None]+scale.log().T[None].transpose(1,2).reshape(1,ne,1)
    D=torch.exp(lnD).reshape(b*ne,-1); vv=v.expand(b*ne,-1); FF=F.expand(b*ne,-1); tt=t.expand(b*ne)
    p=fw.psi(D,vv,FF,tt); lp=fw.sample(p,q0[None].expand(b*ne,-1))+lnu[:ne].repeat(b,1)
    return lp.reshape(b,ne,-1)
lam=float(sys.argv[4]) if len(sys.argv)>4 else 3.0
def loss(u):
    r=(forward(u)-LD)/NOISE; chi=(r**2).sum((1,2)); du=u[:,1:]-u[:,:-1]; tv=torch.sqrt(du**2+1e-3).sum(1); return chi,tv
# gradient check against finite differences
g=torch.autograd.grad(loss(u)[0].sum(),u)[0]; e=1e-5; up=u.detach().clone(); up[0,10]+=e; dn=u.detach().clone(); dn[0,10]-=e
with torch.no_grad(): fd=(loss(up)[0].sum()-loss(dn)[0].sum())/(2*e)
print(f'gradient check: autograd {g[0,10]:.6e} finite differences {fd:.6e}',flush=True)
torch.cuda.synchronize(); a=time.perf_counter(); chi,tv=loss(u); (chi.sum()).backward(); torch.cuda.synchronize(); print(f'one gradient: {time.perf_counter()-a:.3f}s for {NR} realizations, {KN} unknowns each',flush=True)
opt=torch.optim.Adam([u],.08); sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,ITER,1e-3); t1=time.perf_counter()
for it in range(1,ITER+1):
    chi,tv=loss(u); l=(chi+lam*tv).sum(); opt.zero_grad(); l.backward(); opt.step(); sch.step()
    if it%(ITER//5)==0: print(it,f'{time.perf_counter()-t1:.0f}s chi2/N {(chi/(ne*len(radii))).mean():.3f} TV {tv.mean():.2f}',flush=True)
with torch.no_grad():
    rec=Dof(u); med=rec.median(0).values; lo,hi=rec.quantile(.16,0),rec.quantile(.84,0); rec_ln=med
    xr=fw.x; ts=ts0[0,0].item(); tb=None
    R=lambda a,b:((xr>=a)&(xr<b)); 
    jmp=[(.002,.9*ts,'wind'),(1.15*ts,.26,'bubble'),(.30,.66,'medium')]
    for a,b,name in jmp:
        m=R(a,b); dd=(rec_ln-ln_true)[m]; cov=((ln_true>=lo)&(ln_true<=hi))[m].float().mean(); print(f'{name}: median |dlnD| {dd.abs().median():.3f} max {dd.abs().max():.3f}, inside the 68 % band {cov:.2f}, median bandwidth {(hi-lo)[m].median():.3f}',flush=True)
    print('elapsed',time.perf_counter()-t1); np.savez(f'function_inference_k{KN}.npz',x=xr.cpu().numpy(),true=ln_true.cpu().numpy(),med=med.cpu().numpy(),lo=lo.cpu().numpy(),hi=hi.cpu().numpy())
