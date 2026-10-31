"""Experiment E: diffusive shock acceleration with the implicit operator. ṗ changes sign, the sweep alone is not exact.
Steady state by sparse LU (reference), and by GMRES preconditioned with the downward sweep (cost per iteration n_p tridiagonal solves)."""
import sys, time, numpy as np, scipy.sparse as sp, scipy.sparse.linalg as sl
sys.path.insert(0,'/home/cpozogonzalez/saetass/validation/cases')
R=1.; U=1.; EPS=.02; INJ=4; FIT=(10.,300.)
def radial_grid(length, cells, growth=1.05, coarsest=0.02):
    h=length/cells; fine=np.arange(R-12*length,R+4*length+h/2,h); inner,step,x=[],h,fine[0]
    while x>0:
        step=min(step*growth,coarsest*R); x-=step; inner.append(x)
    inner=np.array(inner[::-1]); inner[0]=0.
    if inner[1]<.5*(inner[2]-inner[1]): inner=np.delete(inner,1)
    outer,step,x=[],h,fine[-1]
    while x<2*R:
        step=min(step*growth,coarsest*R); x+=step; outer.append(x)
    return np.concatenate([inner,fine,outer])
def bern(z):
    z=np.asarray(z,float); s=np.abs(z)<1e-4; safe=np.where(s,1.,z)
    with np.errstate(over='ignore'): e=safe/np.expm1(np.minimum(safe,700.))
    return np.where(s,1-.5*z+z*z/12,e)
def build(cells=20,per_decade=16,pmax=1e3,comp=4.,order=2):
    D=EPS*U*R; r=radial_grid(D/U,cells); shock=int(np.argmin(np.abs(r-R))); r[shock]=R
    p=np.logspace(0,np.log10(pmax),int(round(np.log10(pmax)*per_decade))+1); dy=np.log10(p[1]/p[0])
    v=np.where(r<R,U,U/comp*(R/np.maximum(r,R))**2)
    h=np.diff(r); m=.5*(r[1:]+r[:-1]); ve=np.where(m<R,U,U/comp*(R/np.maximum(m,R))**2); z=ve*h/D
    fw=m**2*D/h*bern(-z); bw=m**2*D/h*bern(z)
    e=np.concatenate([[0.],m,[r[-1]]]); V=(e[1:]**3-e[:-1]**3)/3; n=r.size-1
    lo=np.zeros(n); ma=np.zeros(n); up=np.zeros(n)
    ma[:-1]-=fw[:-1]; up[:-1]+=bw[:-1]; lo[1:]+=fw[:-1]; ma[1:]-=bw[:-1]; ma[-1]-=fw[-1]
    Ar=sp.diags([lo[1:]/V[1:n],ma/V[:n],up[:-1]/V[:n-1]],[-1,0,1],format='csr')
    # flow through the dual cell, conservative: the jump of r^2 v falls on the shock node
    fl=m**2*ve; div=(np.concatenate([fl,[0]])[:n]-np.concatenate([[0],fl])[:n])/V[:n]; div[-1]=0
    Vy=-div/(3*np.log(10))                                             # generalized velocity at each node
    N=p.size; Ip=sp.identity(N,format='csr'); Dn=sp.csr_matrix(([],([],[])),shape=(N,N))
    pos,neg=(sp.diags(np.maximum(Vy,0)),sp.diags(np.minimum(Vy,0)))
    # upwind operators on U = p psi, in y
    def up_pos():  # V>0: F_{j+1/2}=V(1.5U_j-.5U_{j-1}), first order at the first face
        L=sp.lil_matrix((N,N))
        for j in range(N):
            if order==2:
                if j>=1: L[j,j]-=1.5/dy; L[j,j-1]+=.5/dy
                else: L[j,j]-=1/dy
                if j>=2: L[j,j-1]+=1.5/dy; L[j,j-2]-=.5/dy
                elif j==1: L[j,j-1]+=1/dy
            else:
                L[j,j]-=1/dy
                if j>=1: L[j,j-1]+=1/dy
        return L.tocsr()
    def up_neg():  # V<0: F_{j-1/2}=V(1.5U_j-.5U_{j+1})
        L=sp.lil_matrix((N,N))
        for j in range(N):
            if order==2:
                if j<=N-2: L[j,j]+=1.5/dy; L[j,j+1]-=.5/dy
                else: L[j,j]+=1/dy
                if j<=N-3: L[j,j+1]-=1.5/dy; L[j,j+2]+=.5/dy
                elif j==N-2: L[j,j+1]-=1/dy
            else:
                L[j,j]+=1/dy
                if j<=N-2: L[j,j+1]-=1/dy
        return L.tocsr()
    P=sp.diags(p); Pi=sp.diags(1/p)
    Lp=Pi@up_pos()@P; Ln=Pi@up_neg()@P
    # sign conventions: dU/dt = -(F+ - F-)/dy
    A=sp.kron(Ip,Ar,format='csr')+sp.kron(Lp,pos,format='csr')+sp.kron(Ln,neg,format="csr")
    Mn=sp.kron(Ip,Ar,format='csr')+sp.kron(Ln,neg,format='csr')
    F=np.zeros((N,n)); F[int(round(INJ*per_decade/16)),shock]=per_decade/16
    return r,p,n,N,shock,A,Mn,F,Vy

import os, math
os.environ['MPLBACKEND']='Agg'
sys.path.insert(0,'/home/cpozogonzalez/saetass/validation')
import adiabatic_shock_acceleration as c
from saetass import Grid, Particle, Solver, State, units as su
from saetass.cli import set_log_level
set_log_level("ERROR")
def talbot(N):
    th=(2*np.arange(N//2)+1)*np.pi/N; z=-.6122+.5017*th/np.tan(.6407*th)+.2645j*th
    dz=.5017*(1/np.tan(.6407*th)-.6407*th/np.sin(.6407*th)**2)+.2645j; return z,dz
def contour(A,F,t,N=32):
    zs,dz=talbot(N); out=0; I=sp.identity(A.shape[0],format='csc',dtype=complex); Ac=A.tocsc().astype(complex)
    for z,d in zip(zs,dz):
        s=z*N/t; x=sl.splu(s*I-Ac).solve(F.ravel().astype(complex)/s)
        w=np.exp(z*N)*d*(N/t)*(2*np.pi/N)/(2j*np.pi); out=out+2*(w*x).real
    return out
def saetass(t_end,cells=20,per_decade=16):
    D=EPS; length=D; r=c.radial_grid(length,cells); p=np.logspace(0,3,3*per_decade+1); dt=(2.5/cells)*(length/cells)
    n_t=int(np.ceil(t_end/dt)); dt=t_end/n_t
    g=Grid(r_centers=r*su.LENGTH,p_centers=p*su.MOMENTUM,t_grid=np.arange(n_t+1)*dt*su.TIME)
    v=np.where(r<R,U,U/4*(R/np.maximum(r,R))**2); sh=int(np.argmin(np.abs(r-R))); Q=np.zeros((p.size,r.size)); Q[INJ,sh]=1.
    S=Solver(state=State(psi_p=np.zeros((p.size,r.size))*su.PSI_P,grid=g,particle=Particle.PROTON),problem_type="advection-diffusion-loss-source",
      operator_params={"advection":{"v_centers":v*su.VELOCITY,"inflow_value_U":0.0*su.AREA*su.PSI_P},"diffusion":{"D_values":np.full((p.size,r.size),D)*su.DIFFUSION_COEFFICIENT,"psi_end":0.0*su.PSI_P},"loss":{"adiabatic_losses":True},"source":{"source":Q*su.SOURCE_PSI_P}},quiet=True)
    a=time.perf_counter(); o=S.run().psi_p.to_value(su.PSI_P); return p,o[:,sh],time.perf_counter()-a

u1,u2=1.,.25
def predicted(p,Pinj,total):
    # planar: (u1-u2)/3 p0 psi(p0) = rate per unit area
    a=-1-3*u2/(u1-u2); return 3*total/((u1-u2)*Pinj)*(p/Pinj)**a
print('steady state; ratio to the planar amplitude at p = 30 GeV/c')
for cells,pd in [(20,16),(20,64),(40,64),(80,64)]:
    r,p,n,N,shock,A,Mn,F,Vy=build(cells,pd); x=sl.splu((-A).tocsc()).solve(F.ravel()).reshape(N,n)
    j=int(round(INJ*pd/16)); dp=p[j]*np.log(10)*np.log10(p[1]/p[0]); e=np.concatenate([[0.],.5*(r[1:]+r[:-1]),[r[-1]]]); Vs=(e[shock+1]**3-e[shock]**3)/3
    tot=F[j,shock]*dp*Vs/R**2; ii=np.argmin(np.abs(p-30)); print(f' implicit cells {cells} per_decade {pd}: psi(30)={x[ii,shock]:.5e} planar {predicted(p[ii],p[j],tot):.5e} ratio {x[ii,shock]/predicted(p[ii],p[j],tot):.4f}',flush=True)
for cells in (20,40):
    a=time.perf_counter(); rr,pp,psi,sh=c.run(cells=cells); t=time.perf_counter()-a
    j=INJ; dpf=pp[j]*np.log(10)*np.log10(pp[1]/pp[0]); g=Grid(r_centers=rr*su.LENGTH); Vs=g.volumes.to_value(su.VOLUME)[sh]/(4*np.pi); tot=dpf*Vs/R**2
    ii=np.argmin(np.abs(pp-30)); print(f' SAETASS cells {cells}: {t:.0f}s psi(30)={psi[ii,sh]:.5e} planar {predicted(pp[ii],pp[j],tot):.5e} ratio {psi[ii,sh]/predicted(pp[ii],pp[j],tot):.4f}',flush=True)
