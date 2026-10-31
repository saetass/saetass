"""Experiment D: 2D neural operator on (r, p) with losses. usage: e7.py name env loss steps
env: none | ss ; loss: data | both"""
import sys, json, time, math, numpy as np, torch, torch.nn.functional as Fn
from gpusolve import pcr, talbot
name,env,lossk,steps=sys.argv[1],sys.argv[2],sys.argv[3],int(sys.argv[4]); torch.manual_seed(0)
dev='cuda'; DD=torch.float64; CD=torch.complex128; NR,NP=128,48; FLOOR=1e-8; YMAX=5.0
r=torch.linspace(0,1,NR,device=dev,dtype=DD); y=torch.linspace(0,YMAX,NP,device=dev,dtype=DD); dy=(y[1]-y[0]).item(); p=10**y
h=r[1]-r[0]; m_=.5*(r[:-1]+r[1:]); e=torch.cat([r[:1]*0,m_,r[-1:]]); Vol=(e[1:]**3-e[:-1]**3)/3
def bern(z):
    s=z.abs()<1e-4; safe=torch.where(s,torch.ones_like(z),z)
    return torch.where(s,1-.5*z+z*z/12,safe/torch.expm1(safe.clamp(max=700.)))
def smooth(b,k=5,scale=1.):
    q=torch.arange(1,k+1,device=dev,dtype=DD); a=torch.randn(b,k,device=dev,dtype=DD)/q; ph=2*math.pi*torch.rand(b,k,device=dev,dtype=DD)
    return scale*torch.sum(a[:,None]*torch.cos(math.pi*r[None,:,None]*q[None,None]+ph[:,None]),-1)
U=lambda b,lo,hi: lo+(hi-lo)*torch.rand(b,1,device=dev,dtype=DD)
def family(b,bubble=False):
    if bubble:
        rt=U(b,.08,.18); rb=U(b,.35,.65); xx=r[None].clamp(min=1e-3); vw=U(b,2,6); Db=10**U(b,-2,-.5)
        Dr=torch.where(xx<rt,1.49*Db*(xx/rt)**(1/3),torch.where(xx<rb,Db,100*Db)); v=torch.where(xx<rt,vw*(xx/.02).clamp(max=1),torch.where(xx<rb,.25*vw*(rt/xx)**2,0*xx))
        Qr=torch.exp(-((r[None]-rt)/.03)**2); br=torch.where(xx<rt,U(b,3,10)*torch.ones_like(xx),torch.where(xx<rb,torch.ones_like(xx),.1*torch.ones_like(xx)))*10**U(b,-3,-1)
    else:
        Dr=torch.exp(math.log(1e-2)+math.log(3e2)*torch.rand(b,1,device=dev,dtype=DD)+smooth(b,scale=.5))
        for _ in range(2):
            c=U(b,.1,.9); Dr=Dr*torch.exp(U(b,-3.4,3.4)*(r[None]>c)*(torch.rand(b,1,device=dev,dtype=DD)>.3))
        c=U(b,.1,.7); v=U(b,0,3)*torch.exp(smooth(b,scale=.3))*torch.where(r[None]<c,torch.ones(b,NR,device=dev,dtype=DD),.25*(c/r[None].clamp(min=1e-3))**2)*(r[None]/.02).clamp(max=1)
        Qr=torch.exp(-((r[None]-U(b,.1,.8))/U(b,.03,.12))**2); br=10**U(b,-3,-1)*torch.exp(smooth(b,scale=.5))
    delta=U(b,.25,.6); alpha=U(b,1.2,2.); sp=U(b,1.5,2.6); yc=U(b,3,5)
    D=Dr[:,None]*torch.exp(delta[:,None]*math.log(10)*y[None,:,None])            # (b,np,nr)
    Pd=-br[:,None]*torch.exp(alpha[:,None]*math.log(10)*y[None,:,None])          # p_dot
    Q=Qr[:,None]*torch.exp(-sp[:,None]*math.log(10)*y[None,:,None]-(10**(y[None,:,None]-yc[:,None])))
    return D,v,Pd,Q
def assemble(D,v,Pd):
    De=2*D[...,:-1]*D[...,1:]/(D[...,:-1]+D[...,1:]); ve=(.5*(v[:,:-1]+v[:,1:]))[:,None]; z=ve*h/De
    fw=m_**2*De/h*bern(-z); bw=m_**2*De/h*bern(z); Nr=NR-1
    lo=torch.zeros(*D.shape[:2],Nr,device=dev,dtype=DD); ma=lo.clone(); up=lo.clone()
    ma[...,:-1]-=fw[...,:-1]; up[...,:-1]+=bw[...,:-1]; lo[...,1:]+=fw[...,:-1]; ma[...,1:]-=bw[...,:-1]; ma[...,-1]-=fw[...,-1]
    V=Pd/(p[None,:,None]*math.log(10)); Vf=.5*(V[:,:-1]+V[:,1:])
    return lo/Vol[:Nr],ma/Vol[:Nr],up/Vol[:Nr],V[...,:Nr],Vf[...,:Nr]
def sweep(lo,ma,up,V,Vf,rhs0,s,order=2,cplx=True):
    """(s - A_r - A_p) x = rhs0, top momentum first: A_p is upwind, losses only go down"""
    T=CD if cplx else DD; b=rhs0.shape[0]; x=[None]*NP
    for j in range(NP-1,-1,-1):
        Vd=Vf[:,j-1] if j>0 else V[:,0]; Vu=Vf[:,j] if j<NP-1 else 0*V[:,0]
        U1=x[j+1]*p[j+1] if j+1<NP else 0; U2=x[j+2]*p[j+2] if j+2<NP else 0
        so=order==2 and j+2<NP
        Fup=Vu*(1.5*U1-.5*U2) if so else Vu*U1
        if order==2 and j+1<NP: diag=-1.5*Vd/dy; extra=-.5*Vd*U1/dy
        else: diag=-Vd/dy; extra=0
        rhs=rhs0[:,j]-Fup/dy/p[j]+extra/p[j]
        x[j]=pcr((-lo[:,j]).to(T),(s+diag-ma[:,j]).to(T),(-up[:,j]).to(T),rhs)
    return torch.stack(x,1)
def solve_t(lo,ma,up,V,Vf,F,t,N=20):
    zs,dz=talbot(N,dev); out=0; tt=t[:,None]
    for zk,dzk in zip(zs,dz):
        s=zk*N/tt; rhs=F.to(CD)/s[:,None]; X=sweep(lo,ma,up,V,Vf,rhs,s)
        w=torch.exp(zk*N)*dzk*(N/tt)*(2*math.pi/N)/(2j*math.pi); out=out+2*(w[:,None]*X).real
    return out.clamp(min=0)
def prepare(b,bubble=False):
    D,v,Pd,Q=family(b,bubble); lo,ma,up,V,Vf=assemble(D,v,Pd); F=Q[...,:NR-1]
    lt=math.log(1e-1)+math.log(1e3)*torch.rand(b,device=dev,dtype=DD); t=torch.exp(lt)
    ss=sweep(lo,ma,up,V,Vf,F,0*t[:,None],order=1,cplx=False).clamp(min=0)
    tr=solve_t(lo,ma,up,V,Vf,F,t)
    if env=='ss': E=ss+FLOOR*ss.amax(2,keepdim=True)+1e-300
    else: E=torch.minimum(ss.amax(2,keepdim=True),t[:,None,None]*F.amax(2,keepdim=True)).expand_as(ss)+1e-300
    ln=lambda u:torch.log(u+1e-300)
    grid=torch.stack([ln(D)[...,:NR-1]/4,(v[:,None,:NR-1]/2).expand(-1,NP,-1),ln(-V)/6,ln(F)/20,(lt/3)[:,None,None].expand(-1,NP,NR-1),r[None,None,:NR-1].expand(b,NP,-1),(y/YMAX)[None,:,None].expand(b,-1,NR-1),ln(E/E.amax(2,keepdim=True))/5],1).float().clamp(-10,10)
    return dict(inp=grid,tr=tr,E=E,S=ss.amax(2,keepdim=True))
class Sp(torch.nn.Module):
    def __init__(s,w,a,c):
        super().__init__(); s.a,s.c=a,c; s.w1=torch.nn.Parameter(torch.randn(w,w,a,c,dtype=torch.cfloat)/w); s.w2=torch.nn.Parameter(torch.randn(w,w,a,c,dtype=torch.cfloat)/w)
    def forward(s,u):
        Ff=torch.fft.rfft2(u); O=torch.zeros_like(Ff); A,C=s.a,s.c
        O[...,:A,:C]=torch.einsum('biac,ioac->boac',Ff[...,:A,:C],s.w1); O[...,-A:,:C]=torch.einsum('biac,ioac->boac',Ff[...,-A:,:C],s.w2)
        return torch.fft.irfft2(O,s=u.shape[-2:])
class Net(torch.nn.Module):
    def __init__(s,w=64,a=16,c=32,l=5):
        super().__init__(); s.lift=torch.nn.Conv2d(8,w,1); s.sp=torch.nn.ModuleList(Sp(w,a,c) for _ in range(l)); s.loc=torch.nn.ModuleList(torch.nn.Conv2d(w,w,3,padding=1) for _ in range(l)); s.o=torch.nn.Sequential(torch.nn.Conv2d(w,192,1),torch.nn.GELU(),torch.nn.Conv2d(192,1,1))
    def forward(s,i):
        u=s.lift(i)
        for a,b in zip(s.sp,s.loc): u=u+Fn.gelu(a(u)+b(u))
        return s.o(u)[:,0]
m=Net().to(dev); print(name,sum(q.numel()*(2 if q.is_complex() else 1) for q in m.parameters()),flush=True)
o=torch.optim.AdamW(m.parameters(),1e-3,weight_decay=1e-5); sch=torch.optim.lr_scheduler.OneCycleLR(o,1e-3,total_steps=steps,pct_start=.05)
def score(bubble,tests=4):
    L=[];S=[]; m.eval()
    with torch.no_grad():
        for _ in range(tests):
            P=prepare(256,bubble); pr=torch.cat([P['E'][i:i+32]*torch.exp(m(P['inp'][i:i+32]).double()) for i in range(0,256,32)]); tr=P['tr']
            ok=tr.amax(2)>1e-30*tr.amax((1,2),keepdim=False)[:,None]
            l=torch.linalg.norm(pr-tr,dim=2)/torch.linalg.norm(tr,dim=2).clamp(min=1e-300); L.append(l[ok]); 
            sp=(tr*r[None,None,:NR-1]**2).sum(2); sq=(pr*r[None,None,:NR-1]**2).sum(2); S.append(((sq/sp.clamp(min=1e-300)-1).abs())[ok])
    m.train(); L=torch.cat(L).cpu().numpy(); S=torch.cat(S).cpu().numpy()
    return dict(l2_med=float(np.median(L)),l2_p95=float(np.percentile(L,95)),spec_med=float(np.median(S)),spec_p95=float(np.percentile(S,95)))
t0=time.perf_counter()
for it in range(1,steps+1):
    if (it-1)%16==0: POOL=prepare(512)
    i=(it-1)%16*32; P={k:v[i:i+32] for k,v in POOL.items()}
    Y=torch.log((P['tr']+FLOOR*P['S'])/P['E']); l=((m(P['inp']).double()-Y)**2).mean()
    o.zero_grad(set_to_none=True); l.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); o.step(); sch.step()
    if it%(steps//4)==0: print(it,f'{time.perf_counter()-t0:.0f}s {l.item():.3e}','rand',score(False,2),'bub',score(True,2),flush=True)
with torch.no_grad():
    D,v,Pd,Q=family(256); lo,ma,up,V,Vf=assemble(D,v,Pd); F=Q[...,:NR-1]; t=torch.full((256,),10.,device=dev,dtype=DD); torch.cuda.synchronize(); a=time.perf_counter(); solve_t(lo,ma,up,V,Vf,F,t); torch.cuda.synchronize(); ms=(time.perf_counter()-a)/256*1e3
    a=time.perf_counter(); ss=sweep(lo,ma,up,V,Vf,F,0*t[:,None],order=1,cplx=False); torch.cuda.synchronize(); me=(time.perf_counter()-a)/256*1e3
    P=prepare(32); a=time.perf_counter(); m(P['inp']); torch.cuda.synchronize(); mn=(time.perf_counter()-a)/32*1e3
res=dict(name=name,env=env,steps=steps,seconds=time.perf_counter()-t0,random=score(False),bubble=score(True),ms_solver=ms,ms_envelope=me,ms_net=mn)
open('D_results.jsonl','a').write(json.dumps(res)+'\n'); print(res,flush=True); print('done',flush=True)
