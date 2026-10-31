"""Batched GPU solver: Laplace inversion on a Talbot contour + parallel cyclic reduction."""
import numpy as np, torch, time, math
def pcr(a,b,c,d):
    """Solve batched tridiagonal systems; tensors (batch,n). a[:,0], c[:,-1] ignored."""
    n=b.shape[-1]; a=a.clone(); c=c.clone(); a[...,0]=0; c[...,-1]=0
    levels=math.ceil(math.log2(n))
    idx=torch.arange(n,device=b.device)
    for k in range(levels):
        s=1<<k
        im=(idx-s).clamp(min=0); ip=(idx+s).clamp(max=n-1)
        hasm=(idx-s)>=0; hasp=(idx+s)<n
        am=a[...,im]; bm=b[...,im]; cm=c[...,im]; dm=d[...,im]
        ap=a[...,ip]; bp=b[...,ip]; cp=c[...,ip]; dp=d[...,ip]
        al=torch.where(hasm,-a/bm,torch.zeros_like(a)); ga=torch.where(hasp,-c/bp,torch.zeros_like(c))
        b=b+al*cm+ga*ap; d=d+al*dm+ga*dp; a=al*am; c=ga*cp
    return d/b
def talbot(N,device):
    th=(2*torch.arange(N//2,device=device,dtype=torch.float64)+1)*math.pi/N   # upper half, midpoints
    z=-0.6122+0.5017*th/torch.tan(0.6407*th)+0.2645j*th
    dz=0.5017*(1/torch.tan(0.6407*th)-0.6407*th/torch.sin(0.6407*th)**2)+0.2645j
    return z,dz
def psi_at(lower,main,upper,forcing,t,N=16):
    """psi(t) for psi'=A psi + F, psi(0)=0. A tridiagonal (batch,n). t tensor (batch,)."""
    z,dz=talbot(N,main.device); out=0
    tt=t[...,None]
    for zk,dzk in zip(z,dz):
        s=zk*N/tt            # (batch,1) complex
        rhs=forcing.to(torch.complex128)/s
        x=pcr((-lower).to(torch.complex128),s-main,(-upper).to(torch.complex128),rhs)
        w=torch.exp(zk*N)*dzk*(N/tt)*(2*math.pi/N)/(2j*math.pi)
        out=out+2*(w*x).real
    return out

def psi_sym(lower,main,upper,forcing,t,N=32):
    """Same as psi_at, after the diagonal similarity that makes the tridiagonal symmetric: S^-1 A S = T."""
    off=(lower[...,1:]*upper[...,:-1]).sqrt()
    logs=torch.cat([torch.zeros_like(lower[...,:1]),torch.cumsum(0.5*(lower[...,1:].log()-upper[...,:-1].log()),-1)],-1)
    # relative to the source node, where the forcing is
    i=forcing.abs().argmax(-1,keepdim=True); logs=logs-logs.gather(-1,i)
    a=torch.cat([torch.zeros_like(off[...,:1]),off],-1)
    c=torch.cat([off,torch.zeros_like(off[...,:1])],-1)
    y=psi_at(a,main,c,forcing*torch.exp(-logs),t,N)
    return y,logs
