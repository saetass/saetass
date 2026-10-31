"""Robustness to the contrast of D, without retraining. usage: e12.py checkpoint env"""
import sys, math, numpy as np, torch
ck,env=sys.argv[1],sys.argv[2]
src=open('e4.py').read(); head=src[:src.index("m=Net().to(dev)")]
sys.argv=['e4.py','x',env,'data','random','4']; exec(head)
m=Net().to(dev); m.load_state_dict(torch.load(ck,map_location=dev)); m.eval()
def bubble_c(contrast):
    def f(b):
        rt=.08+.1*torch.rand(b,1,device=dev,dtype=DD); rb=.35+.3*torch.rand(b,1,device=dev,dtype=DD)
        Db=10**(-2+1.5*torch.rand(b,1,device=dev,dtype=DD)); vw=2+4*torch.rand(b,1,device=dev,dtype=DD); xx=x[None].clamp(min=1e-3)
        D=torch.where(xx<rt,1.49*Db*(xx/rt)**(1/3),torch.where(xx<rb,Db,contrast*Db)); v=torch.where(xx<rt,vw*(xx/.02).clamp(max=1),torch.where(xx<rb,.25*vw*(rt/xx)**2,0*xx))
        return D,v,torch.exp(-((x[None]-rt)/.015)**2)
    return f
print(ck,env)
with torch.no_grad():
    for c in (10,100,1e3,1e4,1e5,1e6):
        L=[]
        for _ in range(16):
            P=prepare(bubble_c(c),64); tr=truth(P,28); pr=P['E']*torch.exp(m(P['inp']).double()); L.append(torch.linalg.norm(pr-tr,dim=1)/torch.linalg.norm(tr,dim=1))
        L=torch.cat(L).cpu().numpy(); print(f' contrast {c:.0e}: L2 median {np.median(L):.4f} p95 {np.percentile(L,95):.3f}',flush=True)
