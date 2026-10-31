"""The generic neural operator as the coarse level, on a physical bubble it never saw, against low-resolution numerics."""
import sys, json, time, math, numpy as np, torch
src=open('e4.py').read(); head=src[:src.index("m=Net().to(dev)")]
sys.argv=['e4.py','x','be','data','mix','4']; exec(head)
import gen, bubblefn as B
S=B.S; net=Net().to(dev); net.load_state_dict(torch.load('logs/be_data_long.pt',map_location=dev)); net.eval()
torch.manual_seed(0); RAD=np.linspace(16.,110.,S.N_RADII); q0=torch.as_tensor(RAD/S.R_MAX,device=dev,dtype=DD)
TH=torch.as_tensor(S.TRUTH,device=dev,dtype=DD)+torch.randn(64,4,device=dev,dtype=DD)*torch.tensor([.02,.3,.02,.1],device=dev,dtype=DD)
def lnpsi(fw,th):
    n=th.shape[0]; Di,vi,Fi,ti,lnu,ts=B.fields(th,fw); p=fw.psi(Di,vi,Fi,ti); return (fw.sample(p,q0[None].expand(n*B.NE,-1))+lnu),(Di,vi,ti,ts,lnu)
fine=gen.Forward(2048,16)
with torch.no_grad():
    ref,_=lnpsi(fine,TH)
    for nn,N in [(128,4),(256,6),(512,8),(1024,12)]:
        fw=gen.Forward(nn,N); torch.cuda.synchronize(); a=time.perf_counter(); o,_=lnpsi(fw,TH); torch.cuda.synchronize(); c=(time.perf_counter()-a)/(64*B.NE)*1e3
        d=(o-ref); print(f'n={nn} N={N}: rms dln psi {d.pow(2).mean().sqrt():.4f}, after removing the mean per observation {(d-d.mean(0)).pow(2).mean().sqrt():.4f}, {c:.4f} ms',flush=True)
    fw=gen.Forward(256,6); Di,vi,Fi,ti,lnu,ts=B.fields(TH,fw); b=Di.shape[0]
    # network on the grid it knows: same operator, Gaussian shell of width 0.015 with unit flux
    lo,ma,up,_=operator(Di,vi,0*Di); Qs=torch.exp(-((x[None]-ts)/.015)**2); Fq=Qs/((Qs[:,:n-1]*(.5*(x[:-1]+x[1:]))[None]**2).sum(1,keepdim=True)*(x[1]-x[0]))
    F=Fq[:,:n-1]; lt=torch.log(ti); t=ti
    E=pcr(-t[:,None]*lo,1-t[:,None]*ma,-t[:,None]*up,t[:,None]*F).clamp(min=0); E=E+FLOOR*E.amax(1,keepdim=True)
    inp=torch.stack([torch.log(Di[:,:n-1])/3,vi[:,:n-1]/2,Fq[:,:n-1]/Fq.amax(1,keepdim=True),(lt/3)[:,None].expand(-1,n-1),x[None,:n-1].expand(b,-1),torch.log(E/E.amax(1,keepdim=True))/5],1).float()
    a=time.perf_counter(); pr=E*torch.exp(net(inp).double()); torch.cuda.synchronize(); c=(time.perf_counter()-a)/b*1e3
    from gen import Forward
    o=fw.sample(pr.clamp(min=1e-300),q0[None].expand(b,-1))+lnu; d=o-ref; print(f'FNO: rms dln psi {d.pow(2).mean().sqrt():.4f}, after removing the mean {(d-d.mean(0)).pow(2).mean().sqrt():.4f}, {c:.4f} ms; D range {Di.min():.2e} {Di.max():.2e}, t {ti.min():.2f} {ti.max():.2f}',flush=True)
    be=fw.sample(E,q0[None].expand(b,-1))+lnu; d=be-ref; print(f'envelope alone: rms {d.pow(2).mean().sqrt():.4f}, after removing the mean {(d-d.mean(0)).pow(2).mean().sqrt():.4f}')
