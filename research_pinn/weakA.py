"""Experiment A: same bubble, same network, only the loss changes. usage: weakA.py kind Pe lam a b seed steps tag"""
import sys, json, time, math, numpy as np, torch
from saetass.pinn.reference import reference_grid, solve, _operator
from saetass.pinn.wind_bubble import DimensionlessBubble as B
from gpusolve import pcr
kind,Pe,lam,a,b,seed,steps,tag=sys.argv[1],*map(float,sys.argv[2:6]),int(sys.argv[6]),int(sys.argv[7]),sys.argv[8]
LOGL=kind.endswith('log'); kindname=kind; kind=kind[:-3] if LOGL else kind
dev='cuda'; DD=torch.float64; torch.set_default_dtype(DD); torch.manual_seed(seed)
bb=B(Pe,lam,a,b); delta=1/3; X=a*b; DT=float(bb.wind_diffusion); T=300.
xf=reference_grid(bb,1); i1=300; i2=1000
xn=xf[::2]; I1,I2=150,500                 # 700 elements, nodes on the interfaces
assert abs(xn[I1]-1)<1e-12 and abs(xn[I2]-a)<1e-9
x_=torch.as_tensor(xn,device=dev); h=x_[1:]-x_[:-1]; K=h.numel()
g=torch.tensor([.5-.5/3**.5,.5+.5/3**.5],device=dev)
xq=(x_[:-1,None]+h[:,None]*g)                                        # (K,2)
zone=torch.zeros(K,dtype=torch.long,device=dev); zone[I1:I2]=1; zone[I2:]=2
def coeffs(x):
    z=zone[:,None]; D=torch.where(z==0,DT*x**delta,torch.where(z==1,torch.full_like(x,1/Pe),torch.full_like(x,lam/Pe)))
    v=torch.where(z==0,torch.full_like(x,4.),torch.where(z==1,x**-2,0*x)); return D,v
Dq,vq=coeffs(xq)
def zeta(x,z):
    return torch.where(z==0,x,torch.where(z==1,1+(x-1)/(a-1),2+(x-a)/(X-a)))
class Net(torch.nn.Module):
    def __init__(s,w=256,d=5):
        super().__init__(); s.f=torch.nn.ModuleList([torch.nn.Linear(14,w)]+[torch.nn.Linear(w,w) for _ in range(d-1)]); s.o=torch.nn.Linear(w,1)
    def forward(s,zt,tau):
        d1,d2=(zt-1).abs(),(zt-2).abs()
        i=[zt-1.5,tau,d1,d2]+[torch.exp(-d1/c) for c in (.3,.1,.03,.01,.003)]+[torch.exp(-d2/c) for c in (.3,.1,.03,.01,.003)]
        u=torch.stack(i,-1)
        u=torch.tanh(s.f[0](u))
        for l in s.f[1:]: u=u+torch.tanh(l(u))
        return s.o(u).squeeze(-1)
net=Net().to(dev)
def psi(x,t,z):
    tau=(torch.log(t)-math.log(1e-2))/math.log(T/1e-2)*2-1
    return t/(1+t)*(1-x/X)*torch.exp(5*net(zeta(x,z),tau))
Zq=zone[:,None].expand(-1,2)
# tridiagonal ingredients on the mesh
def gram_energy(t):
    Dm,_=coeffs(.5*(x_[:-1,None]+x_[1:,None])); Dm=Dm[:,0]; m=.5*(x_[:-1]+x_[1:]); kap=m**2*Dm/h; M=m**2*h/6
    ti=(1/t)[:,None]
    n=K; ma=torch.zeros(t.numel(),n,device=dev); lo=ma.clone(); up=ma.clone()
    ma[:,:-1]+=kap[:-1]+2*ti*M[:-1]; ma[:,1:]+=kap[:-1]+2*ti*M[:-1]; ma[:,-1]+=kap[-1]+2*ti[:,0]*M[-1]
    lo[:,1:]=-kap[:-1]+ti*M[:-1]; up[:,:-1]=-kap[:-1]+ti*M[:-1]
    ma[:,0]+=ti[:,0]*(x_[1]**3/3)*0+1e-12
    return lo,ma,up
lo_,ma_,up_,src=_operator(xn,bb,I1,I2); lo_,ma_,up_=[torch.as_tensor(v,device=dev) for v in (lo_,ma_,up_)]
e=torch.cat([x_[:1]*0,.5*(x_[:-1]+x_[1:]),x_[-1:]]); V=((e[1:]**3-e[:-1]**3)/3)[:K]
def weak_residual(t):
    tt=t[:,None,None]; x=xq[None].expand(t.numel(),-1,-1).clone().requires_grad_(True); tv=tt.expand_as(x).clone().requires_grad_(True)
    p=psi(x,tv,Zq[None].expand_as(x))
    px,pt=torch.autograd.grad(p.sum(),(x,tv),create_graph=True)
    J=vq*p-Dq*px; w=(h/2)[None,:,None]
    phiL,phiR=1-g,g
    a1=x**2*pt; a2=x**2*J
    Rl=(w*(a1*phiL[None,None]+a2/h[None,:,None])).sum(-1); Rr=(w*(a1*phiR[None,None]-a2/h[None,:,None])).sum(-1)
    R=torch.zeros(t.numel(),K+1,device=dev); R[:,:-1]+=Rl; R[:,1:]+=Rr; R=R[:,:-1].clone(); R[:,I1]-=1.0
    return R,p,px,pt,x,J
def strong(t):
    tt=t[:,None,None]; x=xq[None].expand(t.numel(),-1,-1).clone().requires_grad_(True); tv=tt.expand_as(x).clone().requires_grad_(True)
    p=psi(x,tv,Zq[None].expand_as(x)); px,pt=torch.autograd.grad(p.sum(),(x,tv),create_graph=True)
    D=Dq; F=x**2*(vq*p-D*px); Fx=torch.autograd.grad(F.sum(),x,create_graph=True)[0]
    r=tv*(x**2*pt+Fx)/x.detach()**2
    # interface jumps: flux jump at x=1 equals 1, flux continuity at a
    def flux(xv,zv,t_):
        xv=xv.clone().requires_grad_(True); p=psi(xv,t_,zv); px=torch.autograd.grad(p.sum(),xv,create_graph=True)[0]
        z=zv; D=torch.where(z==0,DT*xv**delta,torch.where(z==1,torch.full_like(xv,1/Pe),torch.full_like(xv,lam/Pe)))
        v=torch.where(z==0,torch.full_like(xv,4.),torch.where(z==1,xv**-2,0*xv)); return xv**2*(v*p-D*px)
    one=torch.ones_like(t)
    j1=flux(one,0*one.long()+1,t)-flux(one,0*one.long(),t)-1
    j2=flux(a*one,0*one.long()+2,t)-flux(a*one,0*one.long()+1,t)
    return (r**2).mean()+((j1**2)+(j2**2)).mean()
ref_t=np.array([1.,10.,100.,300.]); ref=solve(bb,ref_t,resolution=4)
xr=torch.as_tensor(ref.x[:-1],device=dev); zr=torch.as_tensor(np.searchsorted(np.array([1.,a]),ref.x[:-1],side='right'),device=dev); pr=torch.as_tensor(ref.psi[:,:-1],device=dev)
wr=torch.as_tensor(np.gradient(ref.x)[:-1],device=dev)
@torch.no_grad()
def metric():
    o=[]
    for j,t in enumerate(ref_t):
        p=psi(xr,torch.full_like(xr,t),zr); true=pr[j]
        l2=((wr*(p-true)**2).sum()/(wr*true**2).sum()).sqrt().item()
        m=true>1e-6*true.max(); o.append((l2,(torch.log(p[m]/true[m]).abs().max().item())))
    return o
opt=torch.optim.Adam(net.parameters(),1e-3); sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,steps,1e-5)
t0=time.perf_counter(); B_T=12
for it in range(1,steps+1):
    t=torch.exp(math.log(1e-2)+math.log(T/1e-2)*torch.rand(B_T,device=dev))
    if kind=='strong': l=strong(t)
    else:
        R,*_=weak_residual(t); Rt=t[:,None]*R
        if kind=='vpinn': l=(Rt**2).mean()*K
        elif kind=='energy':
            lo,ma,up=gram_energy(t); y=pcr(lo,ma,up,Rt); l=(Rt*y).sum(1).mean()
        elif kind=='sg':
            y=pcr(-t[:,None]*lo_,1-t[:,None]*ma_,-t[:,None]*up_,Rt/V); l=(V*y**2).sum(1).mean()*1e2
    if LOGL: l=torch.log(l)
    opt.zero_grad(set_to_none=True); l.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(),10.); opt.step(); sch.step()
    if it%(steps//5)==0: print(kind,it,f'{time.perf_counter()-t0:.0f}s loss {l.item():.3e}',' '.join(f'{a_:.2e}/{b_:.2f}' for a_,b_ in metric()),flush=True)
m=metric(); open('A_results.jsonl','a').write(json.dumps(dict(tag=tag,kind=kindname,Pe=Pe,lam=lam,a=a,b=b,seed=seed,steps=steps,seconds=time.perf_counter()-t0,l2=[x[0] for x in m],logmax=[x[1] for x in m]))+'\n')
print('done',flush=True)
