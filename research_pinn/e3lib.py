import numpy as np, torch, math, time
from gpusolve import pcr, psi_at
dev='cuda'; DD=torch.float64; n=256
x=torch.linspace(0,1,n,device=dev,dtype=DD)
def bern(z):
    s=z.abs()<1e-4; safe=torch.where(s,torch.ones_like(z),z)
    return torch.where(s,1-.5*z+z*z/12,safe/torch.expm1(safe.clamp(max=700.)))
def smooth(b,k=6,scale=1.):
    """random smooth fields (b,n) from low Fourier modes"""
    m=torch.arange(1,k+1,device=dev,dtype=DD); a=torch.randn(b,k,device=dev,dtype=DD)/m; ph=2*math.pi*torch.rand(b,k,device=dev,dtype=DD)
    return scale*torch.sum(a[:,None]*torch.cos(math.pi*x[None,:,None]*m[None,None]+ph[:,None]),-1)
def random_family(b):
    logD=math.log(1e-2)+math.log(3e2)*torch.rand(b,1,device=dev,dtype=DD)+smooth(b,scale=.5)
    # up to two jumps of D, one of v
    for _ in range(2):
        c=.1+.8*torch.rand(b,1,device=dev,dtype=DD); j=math.log(30.)*(2*torch.rand(b,1,device=dev,dtype=DD)-1)
        logD=logD+j*(x[None]>c)*(torch.rand(b,1,device=dev,dtype=DD)>.3)
    v0=3*torch.rand(b,1,device=dev,dtype=DD)
    v=v0*torch.exp(smooth(b,scale=.3)); c=.1+.6*torch.rand(b,1,device=dev,dtype=DD)
    v=v*torch.where(x[None]<c,torch.ones_like(v),.25*(c/x[None].clamp(min=1e-3))**2)*(x[None]/.02).clamp(max=1)
    r0=.1+.7*torch.rand(b,1,device=dev,dtype=DD); w=.02+.1*torch.rand(b,1,device=dev,dtype=DD)
    Q=torch.exp(-((x[None]-r0)/w)**2)
    return torch.exp(logD),v,Q
def bubble_family(b):
    """Out of distribution: a wind bubble profile"""
    rt=.08+.1*torch.rand(b,1,device=dev,dtype=DD); rb=.35+.3*torch.rand(b,1,device=dev,dtype=DD)
    Db=10**(-2+1.5*torch.rand(b,1,device=dev,dtype=DD)); vw=2+4*torch.rand(b,1,device=dev,dtype=DD)
    xx=x[None].clamp(min=1e-3)
    D=torch.where(xx<rt,1.49*Db*(xx/rt)**(1/3),torch.where(xx<rb,Db,100*Db))
    v=torch.where(xx<rt,vw*(xx/.02).clamp(max=1),torch.where(xx<rb,.25*vw*(rt/xx)**2,0*xx))
    Q=torch.exp(-((x[None]-rt)/.015)**2)
    return D,v,Q
def operator(D,v,Q):
    h=x[1]-x[0]; De=2*D[:,:-1]*D[:,1:]/(D[:,:-1]+D[:,1:]); ve=.5*(v[:,:-1]+v[:,1:]); m=.5*(x[:-1]+x[1:])
    z=ve*h/De; fw=m**2*De/h*bern(-z); bw=m**2*De/h*bern(z)
    e=torch.cat([x[:1]*0,m,x[-1:]]); V=(e[1:]**3-e[:-1]**3)/3
    N=n-1; lo=torch.zeros(D.shape[0],N,device=dev,dtype=DD); ma=lo.clone(); up=lo.clone()
    ma[:,:-1]-=fw[:,:-1]; up[:,:-1]+=bw[:,:-1]; lo[:,1:]+=fw[:,:-1]; ma[:,1:]-=bw[:,:-1]; ma[:,-1]-=fw[:,-1]
    return lo/V[:N],ma/V[:N],up/V[:N],Q[:,:N]
def batch(fam,b,N=20):
    D,v,Q=fam(b); lo,ma,up,F=operator(D,v,Q)
    lt=math.log(1e-2)+math.log(1e4)*torch.rand(b,device=dev,dtype=DD)
    p=psi_at(lo,ma,up,F,torch.exp(lt),N).clamp(min=0)
    s=p.amax(1,keepdim=True)
    y=torch.log(p/s+1e-6)  # log of normalized profile, floor 1e-6
    inp=torch.stack([torch.log(D)/3,v/2,Q,lt[:,None].expand(-1,n-1+1)[:,:n]/3,x[None].expand(b,-1)],1)[...,:n-1]
    return inp.float(),y.float(),torch.log(s[:,0]).float(),(lo,ma,up,F,lt,s)
class Spectral(torch.nn.Module):
    def __init__(s,w,m):
        super().__init__(); s.m=m; s.w=torch.nn.Parameter(torch.randn(w,w,m,dtype=torch.cfloat)/w)
    def forward(s,u):
        U=torch.fft.rfft(u); O=torch.zeros_like(U); O[...,:s.m]=torch.einsum('bix,iox->box',U[...,:s.m],s.w)
        return torch.fft.irfft(O,n=u.shape[-1])
class FNO(torch.nn.Module):
    def __init__(s,w=96,m=64,l=6):
        super().__init__(); s.lift=torch.nn.Conv1d(5,w,1); s.sp=torch.nn.ModuleList(Spectral(w,m) for _ in range(l)); s.lo=torch.nn.ModuleList(torch.nn.Conv1d(w,w,3,padding=1) for _ in range(l))
        s.out=torch.nn.Sequential(torch.nn.Conv1d(w,256,1),torch.nn.GELU(),torch.nn.Conv1d(256,2,1))
    def forward(s,i):
        u=s.lift(i)
        for a,b in zip(s.sp,s.lo): u=u+torch.nn.functional.gelu(a(u)+b(u))
        o=s.out(u); return o[:,0],o[:,1].mean(-1)
class DeepONet(torch.nn.Module):
    def __init__(s,w=512,p=256):
        super().__init__()
        mk=lambda a,b: torch.nn.Sequential(torch.nn.Linear(a,w),torch.nn.GELU(),torch.nn.Linear(w,w),torch.nn.GELU(),torch.nn.Linear(w,w),torch.nn.GELU(),torch.nn.Linear(w,w),torch.nn.GELU(),torch.nn.Linear(w,b))
        s.br=mk(255*4,p+1); s.tr=mk(1,p)
    def forward(s,i):
        c=s.br(i[:,:4].flatten(1)); t=s.tr(i[0,4][:,None]); return c[:,:-1]@t.T, c[:,-1]
def score(model,fam,tests=32):
    L=[];G=[]
    model.eval()
    with torch.no_grad():
        for _ in range(tests):
            inp,y,ls,(lo,ma,up,F,lt,s)=batch(fam,64,28)
            p,lsp=model(inp); tru=(torch.exp(y.double())-1e-6)
            pred=(torch.exp(p.double())-1e-6)*torch.exp((lsp-ls).double())[:,None]
            L.append(torch.linalg.norm(pred-tru,dim=1)/torch.linalg.norm(tru,dim=1)); m=y>math.log(2e-6); G.append(((p-y).abs()*m).amax(1))
    model.train(); L=torch.cat(L).cpu().numpy();G=torch.cat(G).cpu().numpy()
    return f'L2 med {np.median(L):.4f} p95 {np.percentile(L,95):.4f} max {L.max():.3f}; logmax med {np.median(G):.3f} p95 {np.percentile(G,95):.3f}'
