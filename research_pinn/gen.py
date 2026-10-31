"""Generic forward model: any D(r), v(r), source, on a uniform grid, batched and differentiable. No family, no parameterization."""
import math, numpy as np, torch
from gpusolve import pcr, psi_at
dev='cuda'; DD=torch.float64
def bern(z):
    s=z.abs()<1e-4; safe=torch.where(s,torch.ones_like(z),z)
    return torch.where(s,1-.5*z+z*z/12,safe/torch.expm1(safe.clamp(max=700.)))
class Forward:
    def __init__(s,n,N):
        s.n,s.N=n,N; s.x=torch.linspace(0,1,n,device=dev,dtype=DD); s.h=s.x[1]-s.x[0]; s.m=.5*(s.x[:-1]+s.x[1:])
        e=torch.cat([s.x[:1]*0,s.m,s.x[-1:]]); s.V=((e[1:]**3-e[:-1]**3)/3)[:n-1]
    def operator(s,D,v):
        """D, v at the nodes, shape (b,n), dimensionless. Tridiagonal A of the free nodes."""
        De=2*D[:,:-1]*D[:,1:]/(D[:,:-1]+D[:,1:]); ve=.5*(v[:,:-1]+v[:,1:]); z=ve*s.h/De; k=s.m**2*De/s.h; fw=k*bern(-z); bw=k*bern(z); n=s.n-1
        lo=torch.zeros_like(D[:,:n]); ma=lo.clone(); up=lo.clone()
        ma=ma-torch.cat([fw[:,:-1],fw[:,-1:]],1)-torch.cat([0*bw[:,:1],bw[:,:-1]],1)
        up=torch.cat([bw[:,:-1],0*bw[:,:1]],1); lo=torch.cat([0*fw[:,:1],fw[:,:-1]],1)
        return lo/s.V,ma/s.V,up/s.V
    def shell(s,rs):
        """Unit injection (per 4 pi) at radius rs, shared linearly by its two nodes: independent of the grid."""
        q=(rs/s.h).reshape(-1); i=q.floor().long().clamp(0,s.n-3); w=(q-i).clamp(0,1); F=torch.zeros(q.numel(),s.n-1,device=dev,dtype=DD)
        F=F.scatter(1,i[:,None],((1-w)/s.V[i])[:,None]); F=F.scatter_add(1,(i+1)[:,None],(w/s.V[i+1])[:,None]); return F
    def psi(s,D,v,F,t):
        lo,ma,up=s.operator(D,v); return psi_at(lo,ma,up,F,t,s.N)
    def sample(s,p,q):
        """Log-interpolated ln psi at dimensionless radii q (b,m)."""
        j=torch.searchsorted(s.x[:s.n-1].expand(q.shape[0],-1).contiguous(),q.contiguous()).clamp(1,s.n-2)
        xa,xb=s.x[j-1],s.x[j]; p=p.clamp(min=1e-300).log(); fa,fb=p.gather(1,j-1),p.gather(1,j); w=(q-xa)/(xb-xa); return fa+w*(fb-fa)
