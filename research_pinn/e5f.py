import sys, time, math, numpy as np, torch
sys.path.insert(0,'/home/cpozogonzalez/saetass/benchmarks')
import problems as P
from gpusolve import pcr, talbot
from saetass.cli import set_log_level
set_log_level("ERROR")
dev='cuda'; DD=torch.float64; CD=torch.complex128
def bern(z):
    s=z.abs()<1e-4; safe=torch.where(s,torch.ones_like(z),z)
    return torch.where(s,1-.5*z+z*z/12,safe/torch.expm1(safe.clamp(max=700.)))
def setup(n_r,n_p):
    r_max,r_ts=18.,2.; r=np.linspace(0,r_max,n_r); p=np.logspace(0,7,n_p)
    v=P.wind_velocity(r,r_ts); D=P.D_0*p[:,None]**(1/3)*np.ones(n_r)
    Pd=-P.B_LOSS*p**2
    Q=np.zeros((n_p,n_r)); Q[:,np.searchsorted(r,r_ts)]=p**-2*np.exp(-p/1e6)
    psi0=np.outer(p**-2.,P._initial_profile(r,r_max))
    return r,p,v,D,Pd,Q,psi0
def solve2d(n_r,n_p,t,N=24):
    r,p,v,D,Pd,Q,psi0=setup(n_r,n_p)
    T=lambda a: torch.as_tensor(a,device=dev,dtype=DD)
    r,p,v,D,Pd,Q,psi0=map(T,(r,p,v,D,Pd,Q,psi0))
    h=r[1]-r[0]; De=2*D[:,:-1]*D[:,1:]/(D[:,:-1]+D[:,1:]); ve=.5*(v[:-1]+v[1:]); m=.5*(r[:-1]+r[1:])
    z=ve*h/De; fw=m**2*De/h*bern(-z); bw=m**2*De/h*bern(z)
    e=torch.cat([r[:1]*0,m,r[-1:]]); Vol=(e[1:]**3-e[:-1]**3)/3
    Nr=n_r-1; lo=torch.zeros(n_p,Nr,device=dev,dtype=DD); ma=lo.clone(); up=lo.clone()
    ma[:,:-1]-=fw[:,:-1]; up[:,:-1]+=bw[:,:-1]; lo[:,1:]+=fw[:,:-1]; ma[:,1:]-=bw[:,:-1]; ma[:,-1]-=fw[:,-1]
    lo,ma,up=lo/Vol[:Nr],ma/Vol[:Nr],up/Vol[:Nr]
    dy=math.log10(p[1]/p[0]); V=Pd/(p*math.log(10)); Vf=.5*(V[:-1]+V[1:])                 # generalized velocity at interior faces
    Vlow=V[0]; 
    zs,dz=talbot(N,dev); out=torch.zeros(n_p,Nr,device=dev,dtype=DD)
    F=Q[:,:Nr]; p0=psi0[:,:Nr]
    for zk,dzk in zip(zs,dz):
        s=zk*N/t; rhs0=F.to(CD)/s+p0.to(CD)
        x=[None]*n_p
        for j in range(n_p-1,-1,-1):
            Vd=Vf[j-1] if j>0 else Vlow; Vu=Vf[j] if j<n_p-1 else 0*Vlow
            # dU_j/dt = -(F_up-F_dn)/dy ; second order upwind from higher momentum
            Uj1=x[j+1]*p[j+1] if j+1<n_p else 0
            Uj2=x[j+2]*p[j+2] if j+2<n_p else 0
            Fup=Vu*(1.5*Uj1-.5*Uj2) if j+2<n_p else Vu*Uj1
            if j+1<n_p: diag=-1.5*Vd/dy; extra=Vd*(-.5*Uj1)/dy
            else: diag=-Vd/dy; extra=0
            rhs=rhs0[j]-Fup/dy/p[j]+extra/p[j]
            x[j]=pcr((-lo[j]).to(CD)[None],(s+diag-ma[j])[None],(-up[j]).to(CD)[None],rhs[None])[0]
        w=torch.exp(zk*N)*dzk*(N/t)*(2*math.pi/N)/(2j*math.pi)
        out=out+2*(w*torch.stack(x)).real
    return r.cpu().numpy(),p.cpu().numpy(),torch.cat([out,torch.zeros(n_p,1,device=dev,dtype=DD)],1).cpu().numpy()
def saet(n_r,n_p,n_steps,dt):
    pr=P.full_2d(n_r=n_r,n_p=n_p,n_steps=n_steps)
    # keep the physical time fixed
    s=pr.solver
    a=time.perf_counter(); o=pr.run().copy(); return o,time.perf_counter()-a,float(s.state.t_val)
def compare(a,ra,pa,b,rb,pb):
    """L2 and quantiles of a vs b on the grid of a, log interpolation of b"""
    from scipy.interpolate import RegularGridInterpolator as RI
    f=RI((np.log(pb),rb),np.log(np.maximum(b,1e-300))); R,Pp=np.meshgrid(ra,np.log(pa))
    bi=np.exp(f(np.stack([Pp.ravel(),R.ravel()],1))).reshape(a.shape)
    dr=ra[1]; mask=(ra>2*dr)&(ra<ra[-1]-3*dr)
    al=[];q=[]
    for j in range(a.shape[0]):
        s=bi[j][mask]; c=a[j][mask]
        if s.max()>0: al.append(np.linalg.norm(c-s)/np.linalg.norm(s))
    return np.median(al),np.max(al)


import sys
T_END=float(sys.argv[1]); LOSS=float(sys.argv[2]); P.B_LOSS=LOSS; nr,np_=400,71
P._initial_profile=lambda r,m: 0.0*r
r,p,v,D,Pd,Q,psi0=setup(nr,np_)
e=np.concatenate([[0.],.5*(r[1:]+r[:-1]),[r[-1]]]); Vn=(e[1:]**3-e[:-1]**3)/3
fr=np.concatenate([[0.],.5*(r[1:]+r[:-1]),[r[-1]+.5*(r[1]-r[0])]]); Vc=(fr[1:]**3-fr[:-1]**3)/3
_,_,c=solve2d(nr,np_,T_END,24)
ninj=T_END*np.sum(Q*Vn,1); J=(0,20,40,55)
print('contour  N/Ninj',' '.join(f'{np.sum(c[j]*Vn)/ninj[j]:.4f}' for j in J))
for mult in (1,10,100):
    ns=int(round(T_END/1e-4))*mult
    pr=P.full_2d(n_r=nr,n_p=np_,n_steps=ns)
    from saetass import units as su
    import numpy as _n
    dt=T_END/ns; pr.solver.grid.t_grid=None if False else pr.solver.grid.t_grid
    # the builder fixes dt=1e-4: rebuild the time step of every subsolver
    for sub in pr.solver.operator_subsolvers:
        sub._dt=sub._dt/mult
    pr.solver.quiet=True; a=time.perf_counter(); o=pr.run().copy(); cc=time.perf_counter()-a
    al=[np.linalg.norm((o[j]-c[j])[3:-3])/np.linalg.norm(c[j][3:-3]) for j in range(0,60)]
    print(f'SAETASS dt/{mult}: {cc:.1f}s N/Ninj',' '.join(f'{np.sum(o[j]*Vc)/ninj[j]:.4f}' for j in J),f' L2 vs contour: median {np.median(al):.4f} p=1 {al[0]:.4f} p=100 {al[20]:.4f} p=1e4 {al[40]:.4f} p=3e5 {al[55]:.4f}',flush=True)
