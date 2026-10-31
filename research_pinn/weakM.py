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
K=256; GP=torch.tensor(np.polynomial.legendre.leggauss(5)[0],device=dev); GW=torch.tensor(np.polynomial.legendre.leggauss(5)[1],device=dev)
IF=torch.tensor([1.,a],device=dev)
def Dv(x):
    z=(x>=1).long()+(x>=a).long()
    D=torch.where(z==0,DT*x.clamp(min=1e-9)**delta,torch.where(z==1,torch.full_like(x,1/Pe),torch.full_like(x,lam/Pe)))
    v=torch.where(z==0,torch.full_like(x,4.),torch.where(z==1,x.clamp(min=1e-9)**-2,0*x)); return D,v,z
def centers(n):
    """No mesh: centers near each interface at log-uniform distances, or uniform in the zones; widths follow the distance and the local layer."""
    u=torch.rand(n,device=dev); which=torch.randint(2,(n,),device=dev); sign=torch.where(torch.rand(n,device=dev)<.5,-1.,1.).to(DD)
    dist=torch.exp(math.log(1e-3)+math.log(3e3)*torch.rand(n,device=dev))
    near=(IF[which]+sign*dist*IF[which]).clamp(1e-3,X*.999)
    c=torch.where(u<.6,near,X*torch.rand(n,device=dev)**2)
    D,v,_=Dv(c); layer=(D/(v.abs()+1e-6)).clamp(1e-3,X)
    d=(c[:,None]-IF).abs().amin(1)
    w=torch.maximum(.7*d,3*layer.clamp(max=.3*X))*torch.exp(.5*torch.randn(n,device=dev)); w=w.clamp(5e-4,X)
    return c,w
from saetass.pinn.reference import reference_grid as _rg
_xg=torch.as_tensor(_rg(bb,1),device=dev); _ug=torch.linspace(0,1,_xg.numel(),device=dev,dtype=DD)
def cloud(n):
    """Random stratified points with the density of the layers: a new cloud every step, without a mesh."""
    u=(torch.arange(n,device=dev,dtype=DD)+torch.rand(n,device=dev))/n; j=torch.searchsorted(_ug,u).clamp(1,_xg.numel()-1)
    dx=(_xg[j]-_xg[j-1])/(_ug[j]-_ug[j-1]); return _xg[j-1]+(u-_ug[j-1])*dx, dx/n
KB=192; NQ=4096
def rbf_loss(t):
    c,w=centers(KB); xp,wq=cloud(NQ); s=(xp[None]-c[:,None])/w[:,None]; ph=((1-s**2).clamp(min=0))**3; dp=-6*s*((1-s**2).clamp(min=0))**2/w[:,None]
    D,v,z=Dv(xp); W=(wq*xp**2)[None]
    x_=xp[None].expand(t.numel(),-1).clone().requires_grad_(True); tv=t[:,None].expand_as(x_).clone().requires_grad_(True); zz=z[None].expand_as(x_)
    p=psi(x_,tv,zz); px,pt=torch.autograd.grad(p.sum(),(x_,tv),create_graph=True); J=v[None]*p-D[None]*px
    R=(xp[None]**2*wq[None]*pt)@ph.T-(xp[None]**2*wq[None]*J)@dp.T
    near=((c-1).abs()<w); R=R-torch.where(near,((1-((1-c)/w)**2).clamp(min=0))**3,0*c)[None]
    M=(ph*W)@ph.T; Bop=(dp*W*D[None])@dp.T-(dp*W*v[None])@ph.T                # B(phi_j, phi_i) without the time term
    I=torch.eye(KB,device=dev,dtype=DD); G=t[:,None,None]**-1*M+Bop[None]+1e-8*(torch.diagonal(M).mean())*I
    y=torch.linalg.solve(G,R[...,None])[...,0]
    err=(y@M*y).sum(1); nrm=((p.detach()**2*xp**2*wq).sum(1)); return (err/(nrm+1e-30)).mean()
def residuals(t):
    n=K; c,w=centers(n); l,u=(c-w).clamp(min=0.),(c+w).clamp(max=X)
    # sub-intervals split at the interfaces: only the two jump locations are known, there is no mesh
    b=torch.stack([l,torch.minimum(torch.maximum(IF[0].expand(n),l),u),torch.minimum(torch.maximum(IF[1].expand(n),l),u),u],1)
    a0,b0=b[:,:-1],b[:,1:]; xq_=.5*(a0+b0)[...,None]+.5*(b0-a0)[...,None]*GP; wq=.5*(b0-a0)[...,None]*GW
    xq_=xq_.reshape(n,-1); wq=wq.reshape(n,-1); mid=.5*(a0+b0)[...,None].expand(-1,-1,5).reshape(n,-1)
    s=(xq_-c[:,None])/w[:,None]; ins=(s.abs()<1)
    phi=((1-s**2).clamp(min=0))**3; dphi=-6*s*((1-s**2).clamp(min=0))**2/w[:,None]
    Dc,vc,_=Dv(c)
    if TILT:
        g=(vc/Dc).clamp(max=50/w)[:,None]; e=torch.exp((-g*(xq_-c[:,None])).clamp(-30,30)); dphi=(dphi-g*phi)*e; phi=phi*e
    D,v,z=Dv(mid); x_=xq_[None].expand(t.numel(),-1,-1).clone().requires_grad_(True); tv=t[:,None,None].expand_as(x_).clone().requires_grad_(True)
    zz=z[None].expand_as(x_); p=psi(x_,tv,zz); px,pt=torch.autograd.grad(p.sum(),(x_,tv),create_graph=True)
    J=v[None]*p-D[None]*px
    R=(wq[None]*(x_**2*pt*phi[None]-x_**2*J*dphi[None])).sum(-1)-(torch.where(((c-1).abs()<w),((1-((1-c)/w)**2).clamp(min=0))**3*(torch.exp((-(vc/Dc).clamp(max=50/w)*(1-c)).clamp(-30,30)) if TILT else 1.),0*c))[None]
    Rt=t[:,None]*R
    if NORM:
        E=(wq*xq_**2*(D*dphi**2)).sum(1)[None]+(wq*xq_**2*phi**2).sum(1)[None]/t[:,None]; Rt=Rt/(t[:,None]*E).clamp(min=1e-300).sqrt()*0+Rt/(E.clamp(min=1e-300)).sqrt()
    return Rt
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
TILT='tilt' in kind; NORM='norm' in kind
opt=torch.optim.Adam(net.parameters(),1e-3); sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,steps,1e-5)
t0=time.perf_counter(); B_T=6 if kind=='rbf' else 12
for it in range(1,steps+1):
    t=torch.exp(math.log(1e-2)+math.log(T/1e-2)*torch.rand(B_T,device=dev))
    if kind=='rbf': l=rbf_loss(t)
    else:
        Rt=residuals(t); l=(Rt**2).mean()
    opt.zero_grad(set_to_none=True); l.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(),10.); opt.step(); sch.step()
    if it%(steps//5)==0: print(kind,it,f'{time.perf_counter()-t0:.0f}s loss {l.item():.3e}',' '.join(f'{a_:.2e}/{b_:.2f}' for a_,b_ in metric()),flush=True)
m=metric(); open('M_results.jsonl','a').write(json.dumps(dict(tag=tag,kind=kind,Pe=Pe,lam=lam,a=a,b=b,seed=seed,steps=steps,seconds=time.perf_counter()-t0,l2=[x[0] for x in m],logmax=[x[1] for x in m]))+'\n')
print('done',flush=True)
