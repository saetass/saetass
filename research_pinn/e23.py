"""Batched solver on the test set of the surrogate, with its metrics: criteria of issue #73."""
import sys, json, time, numpy as np, torch
from saetass.pinn.radial_solver import WindBubbleSolver
from saetass.pinn.wind_bubble import BatchedScales, DELTA, torch_log_steady_state
dev='cuda'; path=sys.argv[1]
d=dict(np.load(path)); keep=d['log_times']>=np.log(.01)-1e-6; lt=d['log_times'][keep]; w=d['w'][:,keep]; zeta=d['zeta']; par=d['parameters'] if 'parameters' in d else d['theta']
print({k:v.shape for k,v in d.items()})
T=lambda a: torch.as_tensor(a,dtype=torch.float64,device=dev)
th=T(par); Pe,lam,a,b=th.exp().T; n,nt,nz=w.shape; outer=a*b; times=T(lt).exp(); wt=T(w)
z=T(zeta)[None]; A=a[:,None]; O=outer[:,None]
x=torch.where(z<1,z,torch.where(z<2,1+(z-1)*(A-1),A+(z-2)*(O-A)))
ls=torch_log_steady_state(x,Pe[:,None],lam[:,None],A,O,DELTA['kolmogorov'])
print('n',n,'nt',nt,'nz',nz,flush=True)
res={}
for nodes,N in ((128,48),(256,48),(512,48),(1024,48),(2048,48)):
    S=WindBubbleSolver(nodes,N,device=dev); pred=torch.empty(n,nt,nz,device=dev,dtype=torch.float64); t0=time.perf_counter()
    for i in range(0,n,8):
        ii=torch.arange(i,min(i+8,n),device=dev); m=ii.numel()
        B=lambda v: v[ii][:,None].expand(m,nt).reshape(-1)
        one=torch.ones(m*nt,device=dev,dtype=torch.float64)
        sc=BatchedScales(B(Pe),B(lam),B(a),B(b),one,one,one)
        pred[ii]=S.log_ratio(x[ii][:,None].expand(m,nt,nz).reshape(m*nt,nz).contiguous(),times.repeat(m),sc).reshape(m,nt,nz)
    torch.cuda.synchronize(); el=time.perf_counter()-t0
    lf=(wt+ls[:,None]); lf=torch.where(torch.isfinite(lf),lf,lf.new_tensor(-1e30)); lp=pred+ls[:,None]; top=lf.amax(2,keepdim=True); pf=(lf-top).exp(); pp=(lp-top).exp()
    l2=(pp-pf).norm(dim=2)/pf.norm(dim=2); md=((lp-lf).abs()*(pf>1e-6)).amax(2)
    q=lambda v: [float(v.median()),float(v.quantile(.95)),float(v.max())]
    print(f'nodes={nodes} N={N} L2 med/p95/worst',['%.2e'%v for v in q(l2)],'dln',['%.2e'%v for v in q(md)],f'{el/(n*nt)*1e6:.0f} us/profile',flush=True)
    wi=md.argmax(); print('   worst dln at bubble',(wi//nt).item(),'time',times[wi%nt].item(),'Pe',Pe[wi//nt].item(),flush=True)
