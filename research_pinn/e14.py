"""Stress test of the contour on the benchmarks electrons_2d and protons_2d. usage: e14.py problem t_end"""
import sys, time, math, numpy as np, torch
sys.path.insert(0,'/home/cpozogonzalez/saetass/benchmarks')
import problems as P
from gpusolve import pcr, talbot
from saetass import units as su
from saetass.cli import set_log_level
set_log_level("ERROR")
dev='cuda'; DD=torch.float64; CD=torch.complex128
def bern(z):
    s=z.abs()<1e-4; safe=torch.where(s,torch.ones_like(z),z)
    return torch.where(s,1-.5*z+z*z/12,safe/torch.expm1(safe.clamp(max=700.)))
name,T_END=sys.argv[1],float(sys.argv[2]); make=getattr(P,name)
pr=make(n_steps=1); s=pr.solver; op=s.operator_params; g=s.grid
r=g.r_centers.to_value(su.LENGTH); p=g.p_centers_phys.to_value(su.MOMENTUM); n_r,n_p=r.size,p.size
D=op['diffusion']['D_values'].to_value(su.DIFFUSION_COEFFICIENT); v=np.asarray(op['advection']['v_centers'].to_value(su.VELOCITY))[0]; Pd=op['loss']['P_dot'].to_value(su.MOMENTUM_LOSS_RATE); Q=op['source']['source'].to_value(su.SOURCE_PSI_P); psi0=s.state.psi_p.to_value(su.PSI_P).copy()
print(name,'grid',n_p,n_r,'p',p[0],p[-1],'Pd sign all negative',bool((Pd<0).all()),'cell Pe max %.3g'%(np.abs(v)[None]*(r[1]-r[0])/D).max(),'loss time min %.3g Myr'%(p[:,None]/np.abs(Pd)).min(),flush=True)
T=lambda a: torch.as_tensor(a,device=dev,dtype=DD)
rt,pt,vt,Dt,Pdt,Qt,p0=map(T,(r,p,v,D,Pd,Q,psi0))
h=rt[1]-rt[0]; m=.5*(rt[:-1]+rt[1:]); De=2*Dt[:,:-1]*Dt[:,1:]/(Dt[:,:-1]+Dt[:,1:]); ve=.5*(vt[:-1]+vt[1:]); z=ve*h/De
fw=m**2*De/h*bern(-z); bw=m**2*De/h*bern(z); e=torch.cat([rt[:1]*0,m,rt[-1:]]); Vol=(e[1:]**3-e[:-1]**3)/3; Nr=n_r-1
lo=torch.zeros(n_p,Nr,device=dev,dtype=DD); ma=lo.clone(); up=lo.clone(); ma[:,:-1]-=fw[:,:-1]; up[:,:-1]+=bw[:,:-1]; lo[:,1:]+=fw[:,:-1]; ma[:,1:]-=bw[:,:-1]; ma[:,-1]-=fw[:,-1]
lo,ma,up=lo/Vol[:Nr],ma/Vol[:Nr],up/Vol[:Nr]
dy=torch.log10(pt[1]/pt[0]).item(); V=(Pdt/(pt[:,None]*math.log(10)))[:,:Nr]; Vf=.5*(V[:-1]+V[1:])
def sweep(rhs0,s,order=2):
    x=[None]*n_p
    for j in range(n_p-1,-1,-1):
        Vd=Vf[j-1] if j>0 else V[0]; Vu=Vf[j] if j<n_p-1 else 0*V[0]
        U1=x[j+1]*pt[j+1] if j+1<n_p else 0; U2=x[j+2]*pt[j+2] if j+2<n_p else 0
        Fup=Vu*(1.5*U1-.5*U2) if (order==2 and j+2<n_p) else Vu*U1
        if order==2 and j+1<n_p: diag=-1.5*Vd/dy; extra=-.5*Vd*U1/dy
        else: diag=-Vd/dy; extra=0
        rhs=rhs0[j]-Fup/dy/pt[j]+extra/pt[j]
        x[j]=pcr((-lo[j]).to(rhs.dtype)[None],(s+diag-ma[j]).to(rhs.dtype)[None],(-up[j]).to(rhs.dtype)[None],rhs[None])[0]
    return torch.stack(x)
def contour(t,N=24,order=2):
    zs,dz=talbot(N,dev); out=0
    for zk,dzk in zip(zs,dz):
        s_=zk*N/t; X=sweep(Qt[:,:Nr].to(CD)/s_+p0[:,:Nr].to(CD),s_,order); w=torch.exp(zk*N)*dzk*(N/t)*(2*math.pi/N)/(2j*math.pi); out=out+2*(w*X).real
    return out
torch.cuda.synchronize(); a=time.perf_counter(); c=contour(T_END); torch.cuda.synchronize(); cc=time.perf_counter()-a; c=c.cpu().numpy(); print(f'contour N=24: {cc:.1f}s, negative fraction {(c<-1e-12*c.max()).mean():.3f}, min/max {c.min()/c.max():.2e}',flush=True)
c32=contour(T_END,32).cpu().numpy(); print('N=32 vs N=24: L2 over p, median %.2e max %.2e'%(np.median(np.linalg.norm(c32-c,axis=1)/np.linalg.norm(c32,axis=1)),(np.linalg.norm(c32-c,axis=1)/np.linalg.norm(c32,axis=1)).max()),flush=True)
c=np.concatenate([c32,np.zeros((n_p,1))],1)
n0=int(round(T_END/1e-4))
for mult in (4,16,64):
    pr=make(n_steps=n0*mult); 
    for sub in pr.solver.operator_subsolvers: sub._dt=sub._dt/mult
    pr.solver.quiet=True; a=time.perf_counter(); o=pr.run().copy(); t_=time.perf_counter()-a
    weights=r**2; sel=[j for j in range(n_p) if np.sum(c[j]*weights)>0]
    e=np.array([np.abs(np.sum(o[j]*weights)/np.sum(c[j]*weights)-1) for j in sel]); l=np.array([np.linalg.norm((o[j]-c[j])[3:-3])/np.linalg.norm(c[j][3:-3]) for j in sel])
    print(f'SAETASS dt/{mult}: {t_:.0f}s; spectrum diff: median {np.median(e):.3f} p95 {np.percentile(e,95):.3f}; L2 per p: median {np.median(l):.3f} p95 {np.percentile(l,95):.3f}',flush=True)
