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
    F=np.zeros((N,n)); F[INJ,shock]=1.
    return r,p,n,N,shock,A,Mn,F,Vy
for cells,pd,o in [(20,16,1),(20,16,2),(20,32,2),(20,64,2),(20,128,2),(40,64,2),(80,64,2),(160,64,2)]:
    r,p,n,N,shock,A,Mn,F,Vy=build(cells,pd,order=o)
    a=time.perf_counter(); lu=sl.splu((-A).tocsc()); x=lu.solve(F.ravel()); c=time.perf_counter()-a
    psi=x.reshape(N,n); sel=(p>=FIT[0])&(p<=FIT[1]); s_=np.polyfit(np.log(p[sel]),np.log(np.maximum(psi[sel,shock],1e-300)),1)[0]
    print(f'cells {cells} per_decade {pd} order {o}: {A.shape[0]} unknowns {c:.2f}s slope {s_:.4f} (theory -2.0563)',flush=True)
