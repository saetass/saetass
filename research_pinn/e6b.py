import sys,re,time,torch,math,numpy as np
s=open('e6.py').read(); head=s[:s.index('def prepare(fam,b,full=True):')]
sys.argv=['e6.py','x','ss','4']; exec(head)
torch.manual_seed(1)
b=1024; D,v,Q=random_family(b); lt=math.log(.1)+math.log(1e3)*torch.rand(b,device=dev,dtype=DD); t=torch.exp(lt); h=hist(b)
ref=stepper(D,v,Q,t,h,h,M=1920).clamp(min=0)
for M in (15,30,60,120,240,480):
    torch.cuda.synchronize(); a=time.perf_counter(); p=stepper(D,v,Q,t,h,h,M=M); torch.cuda.synchronize(); c=(time.perf_counter()-a)/b*1e3
    e=torch.linalg.norm(p-ref,dim=1)/torch.linalg.norm(ref,dim=1); e=e.cpu().numpy(); print(f'M={M}: {c:.3f} ms/profile, L2 vs M=1920: median {np.median(e):.2e} p95 {np.percentile(e,95):.2e} max {e.max():.2e}',flush=True)
