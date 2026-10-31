"""Experiment C: time-dependent D(t), Q(t), where the contour does not apply. usage: e6.py name env steps
env: none | ss | be (frozen coefficients at the final time)"""
import sys, json, time, math, numpy as np, torch
from e3lib import *
name,env,steps=sys.argv[1],sys.argv[2],int(sys.argv[3]); torch.manual_seed(0); FLOOR=1e-8; NH=16
def mix(b):
    k=b//2; a=random_family(b-k); c=bubble_family(k); return tuple(torch.cat([u,w]) for u,w in zip(a,c))
def hist(b):
    a=.2+.8*torch.rand(b,2,device=dev,dtype=DD); w=.3+1.5*torch.rand(b,2,device=dev,dtype=DD); ph=2*math.pi*torch.rand(b,2,device=dev,dtype=DD)
    return lambda s,i: a[:,i]*torch.sin(w[:,i]*s+ph[:,i])       # ln factor as a function of ln t
def stepper(D,v,Q,tend,f,g,M=60):
    """batched TR-BDF2 on log steps with time-dependent coefficients, frozen within each step"""
    b=D.shape[0]; lo,ma,up,F=operator(D,v,Q); s0=torch.log(tend)-6.0
    ts=torch.exp(torch.linspace(0,1,M+1,device=dev,dtype=DD)[None]*6.0+s0[:,None]); gam=2-2**.5
    psi=torch.zeros(b,n-1,device=dev,dtype=DD)
    A=lambda fac,y: fac*(ma*y+torch.cat([0*y[:,:1],lo[:,1:]*y[:,:-1]],1)+torch.cat([up[:,:-1]*y[:,1:],0*y[:,:1]],1))
    for k in range(M):
        dt=(ts[:,k+1]-ts[:,k])[:,None]; sm=torch.log(.5*(ts[:,k]+ts[:,k+1]))
        fac=torch.exp(f(sm,0))[:,None]; src=F*torch.exp(g(sm,1))[:,None]   # D advective part kept: only the operator scale
        sh=.5*gam*dt*fac; stage=pcr(-sh*lo,1-sh*ma,-sh*up,psi+sh*(ma*psi+torch.cat([0*psi[:,:1],lo[:,1:]*psi[:,:-1]],1)+torch.cat([up[:,:-1]*psi[:,1:],0*psi[:,:1]],1))+gam*dt*src)
        w=1/(gam*(2-gam)); c=(1-gam)/(2-gam)*dt
        psi=pcr(-c*fac*lo,1-c*fac*ma,-c*fac*up,w*stage-(1-gam)**2*w*psi+c*src)
    return psi
def prepare(fam,b,full=True):
    D,v,Q=fam(b); lt=math.log(.1)+math.log(1e3)*torch.rand(b,device=dev,dtype=DD); t=torch.exp(lt); h=hist(b)
    lo,ma,up,F=operator(D,v,Q); ff=torch.exp(h(lt,0))[:,None]; gg=torch.exp(h(lt,1))[:,None]
    ss=pcr(-lo,-ma,-up,F).clamp(min=0)*gg/ff
    S=torch.minimum(ss.amax(1),t*(F*gg).amax(1))[:,None]
    if env=='ss': E=ss+FLOOR*ss.amax(1,keepdim=True)
    elif env=='be':
        E=pcr(-t[:,None]*ff*lo,1-t[:,None]*ff*ma,-t[:,None]*ff*up,t[:,None]*gg*F).clamp(min=0); E=E+FLOOR*E.amax(1,keepdim=True)
    else: E=S.expand(-1,n-1)
    grid=torch.linspace(-5,0,NH,device=dev,dtype=DD)[None]+lt[:,None]
    H0=h(grid.T,0).T if False else torch.stack([h(grid[:,j],0) for j in range(NH)],1); H1=torch.stack([h(grid[:,j],1) for j in range(NH)],1)
    base=torch.stack([torch.log(D[:,:n-1])/3,v[:,:n-1]/2,Q[:,:n-1],(lt/3)[:,None].expand(-1,n-1),x[None,:n-1].expand(b,-1),torch.log(E/E.amax(1,keepdim=True))/5],1)
    inp=torch.cat([base,H0[...,None].expand(-1,-1,n-1),H1[...,None].expand(-1,-1,n-1)],1).float()
    tr=stepper(D,v,Q,t,h,h).clamp(min=0) if full else None
    return dict(inp=inp,E=E,S=S,tr=tr)
class Net(FNO):
    def __init__(s,*a,**k):
        super().__init__(*a,**k); s.lift=torch.nn.Conv1d(6+2*NH,96,1)
    def forward(s,i):
        u=s.lift(i)
        for a,b in zip(s.sp,s.lo): u=u+torch.nn.functional.gelu(a(u)+b(u))
        return s.out(u)[:,0]
m=Net().to(dev); o=torch.optim.AdamW(m.parameters(),1e-3,weight_decay=1e-5); sch=torch.optim.lr_scheduler.OneCycleLR(o,1e-3,total_steps=steps,pct_start=.05)
def score(f,tests=16):
    L=[]; m.eval()
    with torch.no_grad():
        for _ in range(tests):
            P=prepare(f,64); pr=P['E']*torch.exp(m(P['inp']).double()); L.append(torch.linalg.norm(pr-P['tr'],dim=1)/torch.linalg.norm(P['tr'],dim=1))
    m.train(); L=torch.cat(L).cpu().numpy(); return dict(l2_med=float(np.median(L)),l2_p95=float(np.percentile(L,95)),l2_max=float(L.max()))
t0=time.perf_counter()
for it in range(1,steps+1):
    if (it-1)%16==0: POOL=prepare(random_family,1024)
    i=(it-1)%16*64; P={k:v[i:i+64] for k,v in POOL.items()}
    y=torch.log((P['tr']+FLOOR*P['S'])/P['E']); l=((m(P['inp']).double()-y)**2).mean()
    o.zero_grad(set_to_none=True); l.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); o.step(); sch.step()
    if it%(steps//4)==0: print(it,f'{time.perf_counter()-t0:.0f}s {l.item():.3e}','rand',score(random_family,6),'bub',score(bubble_family,6),flush=True)
# timing: one batch of the solver against the network with its envelope
with torch.no_grad():
    D,v,Q=random_family(1024); t=torch.full((1024,),10.,device=dev,dtype=DD); h=hist(1024); torch.cuda.synchronize(); a=time.perf_counter(); stepper(D,v,Q,t,h,h); torch.cuda.synchronize(); ts=time.perf_counter()-a
    P=prepare(random_family,1024,False); torch.cuda.synchronize(); a=time.perf_counter(); P=prepare(random_family,1024,False); m(P['inp']); torch.cuda.synchronize(); tn=time.perf_counter()-a
res=dict(name=name,env=env,steps=steps,seconds=time.perf_counter()-t0,random=score(random_family),bubble=score(bubble_family),ms_solver=ts/1024*1e3,ms_net=tn/1024*1e3)
open('C_results.jsonl','a').write(json.dumps(res)+'\n'); print(res,flush=True); print('done',flush=True)
