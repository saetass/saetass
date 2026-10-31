"""Experiment B: neural operator on arbitrary profiles. usage: e4.py name env loss train steps [seed]
env: none | ss | be ; loss: data | weak | both ; train: random | mix"""
import sys, json, time, math, numpy as np, torch
from e3lib import *
name,env,lossk,train_on,steps=sys.argv[1:5]+[int(sys.argv[5])]; seed=int(sys.argv[6]) if len(sys.argv)>6 else 0
torch.manual_seed(seed); FLOOR=1e-8
def mix(b):
    k=b//2; a=random_family(b-k); c=bubble_family(k); return tuple(torch.cat([u,w]) for u,w in zip(a,c))
fam={'random':random_family,'mix':mix}[train_on]
def prepare(f,b):
    D,v,Q=f(b); lo,ma,up,F=operator(D,v,Q)
    lt=math.log(1e-2)+math.log(1e4)*torch.rand(b,device=dev,dtype=DD); t=torch.exp(lt)
    ss=pcr(-lo,-ma,-up,F).clamp(min=0)                               # steady state: A psi + F = 0
    S=torch.minimum(ss.amax(1),t*F.amax(1))[:,None]                  # natural scale at that time
    if env=='ss': E=ss+FLOOR*ss.amax(1,keepdim=True)
    elif env=='be':
        E=pcr(-t[:,None]*lo,1-t[:,None]*ma,-t[:,None]*up,t[:,None]*F).clamp(min=0); E=E+FLOOR*E.amax(1,keepdim=True)
    else: E=S.expand(-1,n-1)
    extra=torch.log(E/E.amax(1,keepdim=True))/5
    inp=torch.stack([torch.log(D[:,:n-1])/3,v[:,:n-1]/2,Q[:,:n-1],(lt/3)[:,None].expand(-1,n-1),x[None,:n-1].expand(b,-1),extra],1).float()
    return dict(inp=inp,lo=lo,ma=ma,up=up,F=F,lt=lt,t=t,E=E,S=S)
def truth(P,N=24): return psi_at(P['lo'],P['ma'],P['up'],P['F'],P['t'],N).clamp(min=0)
class Net(FNO):
    def __init__(s,*a,**k):
        super().__init__(*a,**k); s.lift=torch.nn.Conv1d(6,96,1)
    def forward(s,i):
        u=s.lift(i)
        for a,b in zip(s.sp,s.lo): u=u+torch.nn.functional.gelu(a(u)+b(u))
        return s.out(u)[:,0]
m=Net().to(dev); print(name,sum(p.numel()*(2 if p.is_complex() else 1) for p in m.parameters()),flush=True)
o=torch.optim.AdamW(m.parameters(),1e-3,weight_decay=1e-5); sch=torch.optim.lr_scheduler.OneCycleLR(o,1e-3,total_steps=steps,pct_start=.05)
def weak(P):
    ds=.05; i1=P['inp'].clone(); i2=P['inp'].clone(); i1[:,3]-=ds/3; i2[:,3]+=ds/3
    p1=P['E']*torch.exp(m(i1).double()); p2=P['E']*torch.exp(m(i2).double()); pb=.5*(p1+p2)
    t=P['t'][:,None]; lo,ma,up=P['lo'],P['ma'],P['up']
    Ap=ma*pb+torch.cat([0*pb[:,:1],lo[:,1:]*pb[:,:-1]],1)+torch.cat([up[:,:-1]*pb[:,1:],0*pb[:,:1]],1)
    r=t*(Ap+P['F'])-(p2-p1)/(2*ds)
    y=pcr(-t*lo,1-t*ma,-t*up,r)                                       # error-like, discrete dual norm
    e=x[1:]; vol=(.5*(x[:-1]+x[1:]))**2
    rel=(vol*y**2).sum(1)/((vol*pb.detach()**2).sum(1)+1e-300)
    return rel.sqrt().mean() if lossk=='weakr' else rel.mean()
def score(f,tests=24):
    L=[];G=[]; m.eval()
    with torch.no_grad():
        for _ in range(tests):
            P=prepare(f,64); tr=truth(P,28); pr=P['E']*torch.exp(m(P['inp']).double())
            L.append(torch.linalg.norm(pr-tr,dim=1)/torch.linalg.norm(tr,dim=1)); ma_=tr>1e-6*tr.amax(1,keepdim=True)
            G.append(((torch.log(pr.clamp(min=1e-300)/tr.clamp(min=1e-300))).abs()*ma_).amax(1))
    m.train(); L=torch.cat(L).cpu().numpy();G=torch.cat(G).cpu().numpy()
    return dict(l2_med=float(np.median(L)),l2_p95=float(np.percentile(L,95)),l2_max=float(L.max()),lg_med=float(np.median(G)),lg_p95=float(np.percentile(G,95)))
t0=time.perf_counter()
for it in range(1,steps+1):
    P=prepare(fam,64); l=torch.zeros((),device=dev,dtype=DD)
    if lossk in('data','both'):
        tr=truth(P); y=torch.log((tr+FLOOR*P['S'])/P['E']); out=m(P['inp']).double()
        l=l+((out-y)**2).mean()
    if lossk in('weak','both','weakr'): l=l+weak(P)*(10 if lossk=='weak' else 1)
    o.zero_grad(set_to_none=True); l.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); o.step(); sch.step()
    if it%(steps//4)==0:
        print(it,f'{time.perf_counter()-t0:.0f}s {l.item():.3e}','rand',score(random_family,8),'bub',score(bubble_family,8),flush=True)
res=dict(name=name,env=env,loss=lossk,train=train_on,steps=steps,seed=seed,seconds=time.perf_counter()-t0,random=score(random_family),bubble=score(bubble_family))
open('B_results.jsonl','a').write(json.dumps(res)+'\n'); torch.save(m.state_dict(),'logs/'+name+'.pt'); print('done',flush=True)
