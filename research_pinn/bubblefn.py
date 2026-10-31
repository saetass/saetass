"""Wind bubble written as plain functions for gen.Forward, in units of r_max and of the bubble velocity."""
import sys, math, numpy as np, torch, astropy.units as u
sys.path.insert(0,'/home/cpozogonzalez/saetass/validation/cases')
import pinn_parameter_search as S
from saetass.pinn.wind_bubble import FastScales
from gen import dev,DD
fast=FastScales(1*u.PeV); E=torch.as_tensor(S.ENERGIES,device=dev,dtype=DD); P=(E**2+2*E*fast.mc2)**.5; NE=P.numel()
def fields(theta,fw,Dscale=None):
    """theta (n,4): log10 L, Mdot, rho, eta. Returns D, v, F, t, lnunit for every (walker, energy), flattened, and the radial unit."""
    n=theta.shape[0]; val=10.**theta; L,M,rho,eta=[val[:,None,i] for i in range(4)]
    sc=fast(P[None,:],L,M,rho,torch.full_like(L,S.AGE),torch.full_like(L,S.R_MAX),eta_B=eta)
    Rt=sc.R_TS.expand(n,NE).reshape(-1,1); Rb=(sc.bubble_ratio*sc.R_TS).expand(n,NE).reshape(-1,1); Tu=sc.time_unit.expand(n,NE).reshape(-1,1)
    Pe=sc.peclet.expand(n,NE).reshape(-1,1); lam=sc.diffusion_ratio.expand(n,NE).reshape(-1,1); vb=Rt/Tu; Db=vb*Rt/Pe
    R=S.R_MAX; ts=Rt/R; tb=Rb/R; Tn=R/vb                                # time unit R/vb
    x=fw.x[None].expand(n*NE,-1); xx=x.clamp(min=fw.h.item()*.5)
    Dn=Db*Tn/R**2                                                        # D_b in units of R^2/Tn
    D=torch.where(xx<ts,11**(fast.delta/2)*Dn*(xx/ts)**fast.delta,torch.where(xx<=tb,Dn,lam*Dn))
    v=torch.where(xx<ts,4*torch.ones_like(xx),torch.where(xx<=tb,(ts/xx)**2,0*xx))
    t=(S.AGE/Tn).reshape(-1)
    lnu=sc.psi_unit.expand(n,NE).reshape(-1,1).log()+2*(ts/1).log().reshape(-1,1)   # rate/(4 pi vb R^2), since the source here has unit flux
    return D,v,fw.shell(ts),t,lnu,ts
