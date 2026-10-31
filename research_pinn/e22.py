"""electrons_2d and protons_2d with the fluxes fitted in the Laplace domain, against plain SG, and SAETASS. usage: e22.py problem t_end"""
import sys, time, math, numpy as np, torch
sys.path.insert(0,'/home/cpozogonzalez/saetass/benchmarks')
import problems as P
from saetass.pinn.radial_solver import ScaledSystem, solve_tridiagonal, talbot_contour, log_bernoulli
from saetass import units as su
from saetass.cli import set_log_level
set_log_level("ERROR")
dev='cuda'; DD=torch.float64; CD=torch.complex128
name,T_END=sys.argv[1],float(sys.argv[2]); make=getattr(P,name)
pr=make(n_steps=1); s_=pr.solver; op=s_.operator_params; g=s_.grid
r=g.r_centers.to_value(su.LENGTH); p=g.p_centers_phys.to_value(su.MOMENTUM); n_r,n_p=r.size,p.size
D=op['diffusion']['D_values'].to_value(su.DIFFUSION_COEFFICIENT); v=np.asarray(op['advection']['v_centers'].to_value(su.VELOCITY))[0]; Pd=op['loss']['P_dot'].to_value(su.MOMENTUM_LOSS_RATE); Q=op['source']['source'].to_value(su.SOURCE_PSI_P); psi0=s_.state.psi_p.to_value(su.PSI_P).copy()
T=lambda a: torch.as_tensor(a,device=dev,dtype=DD)
rt,pt,vt,Dt,Pdt,Qt,p0=map(T,(r,p,v,D,Pd,Q,psi0)); Nr=n_r-1
h=rt[1]-rt[0]; m=.5*(rt[:-1]+rt[1:]); De=2*Dt[:,:-1]*Dt[:,1:]/(Dt[:,:-1]+Dt[:,1:]); ve=.5*(vt[:-1]+vt[1:])
e=torch.cat([rt[:1]*0,m,rt[-1:]]); Vol=((e[1:]**3-e[:-1]**3)/3)[:Nr]; Cap=(rt[1:]**3-rt[:-1]**3)/3
dy=torch.log10(pt[1]/pt[0]).item(); V=(Pdt/(pt[:,None]*math.log(10)))[:,:Nr]; Vf=.5*(V[:-1]+V[1:])
z=ve*h/De; logR=(h/(m**2*De)).log()
class Cells:
    pass
def fitted(j,s):
    """diagonals of level j at the cell-dependent shift s"""
    S=ScaledSystem(lower=None,main=None,upper=None,forcing=None,log_steady=torch.zeros(Nr,device=dev,dtype=DD)[None],log_resistance=logR[j][None],exponent=z[j][None],volumes=Vol[None],capacities=Cap[None])
    return S._fitted(s)
def sg(j):
    lf=log_bernoulli(-z[j])-logR[j]; lb=log_bernoulli(z[j])-logR[j]; f=lf.exp(); b=lb.exp(); zero=torch.zeros(1,device=dev,dtype=DD)
    return torch.cat([zero,f[:-1]]),f+torch.cat([zero,b[:-1]]),torch.cat([b[:-1],zero])
def sweep(rhs0,s,mode,order=2):
    x=[None]*n_p
    for j in range(n_p-1,-1,-1):
        Vd=Vf[j-1] if j>0 else V[0]; Vu=Vf[j] if j<n_p-1 else 0*V[0]
        U1=x[j+1]*pt[j+1] if j+1<n_p else 0; U2=x[j+2]*pt[j+2] if j+2<n_p else 0
        Fup=Vu*(1.5*U1-.5*U2) if (order==2 and j+2<n_p) else Vu*U1
        if order==2 and j+1<n_p: diag=-1.5*Vd/dy; extra=-.5*Vd*U1/dy
        else: diag=-Vd/dy; extra=0
        rhs=(rhs0[j]-Fup/dy/pt[j]+extra/pt[j])*Vol
        if mode=='fitted':
            shift=s+.5*(diag+torch.cat([diag[1:],diag[-1:]]))     # per cell
            lo,ma,up=fitted(j,shift[None]); x[j]=solve_tridiagonal(lo,ma,up,rhs[None])[0]
        else:
            lo,ma,up=sg(j); 
            x[j]=solve_tridiagonal(-lo.to(CD)[None],(ma+(s+diag)*Vol)[None],-up.to(CD)[None],rhs[None])[0]
    return torch.stack(x)
def contour(t,N,mode):
    zs,w=talbot_contour(N,dev); out=0
    for zk,wk in zip(zs,w):
        s=zk/t; X=sweep(Qt[:,:Nr].to(CD)/s+p0[:,:Nr].to(CD),s*torch.ones(Nr,device=dev,dtype=CD),mode); out=out+(wk/t*X).real
    return out
res={}
for mode in ('fitted','sg'):
    for N in (24,32):
        a=time.perf_counter(); c=contour(T_END,N,mode); torch.cuda.synchronize(); print(mode,N,f'{time.perf_counter()-a:.1f}s min/max {c.min()/c.max():.2e}',flush=True); res[mode,N]=c.cpu().numpy()
    print(mode,'N24 vs N32 L2 median',np.median(np.linalg.norm(res[mode,24]-res[mode,32],axis=1)/np.linalg.norm(res[mode,32],axis=1)),flush=True)
w=r**2; 
n0=int(round(T_END/1e-4))
def cmp(o,c,label):
    c=np.concatenate([c,np.zeros((n_p,1))],1); sel=[j for j in range(n_p) if np.sum(c[j]*w)>0]
    a=np.array([np.abs(np.sum(o[j]*w)/np.sum(c[j]*w)-1) for j in sel]); l=np.array([np.linalg.norm((o[j]-c[j])[3:-3])/np.linalg.norm(c[j][3:-3]) for j in sel])
    print(f'  {label}: spectrum diff median {np.median(a):.3f} p95 {np.percentile(a,95):.3f}; L2 per p median {np.median(l):.3f} p95 {np.percentile(l,95):.3f}',flush=True)
np.savez('e22_'+name+'.npz',fitted=res['fitted',32],sg=res['sg',32],r=r,p=p)
for mult in (1,4,16,64):
    pr=make(n_steps=n0*mult)
    for sub in pr.solver.operator_subsolvers: sub._dt=sub._dt/mult
    pr.solver.quiet=True; a=time.perf_counter(); o=pr.run().copy(); print(f'SAETASS dt/{mult}: {time.perf_counter()-a:.0f}s',flush=True)
    cmp(o,res['fitted',32],'fitted'); cmp(o,res['sg',32],'plain SG')
