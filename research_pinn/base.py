import sys, numpy as np, torch
for env in ('ss','be'):
    src=open('e4.py').read(); head=src[:src.index("class Net(FNO):")]
    sys.argv=['e4.py','x',env,'data','random','4']; exec(head)
    torch.manual_seed(0)
    def sc(f):
        L=[];G=[]
        with torch.no_grad():
            for _ in range(24):
                P=prepare(f,64); tr=truth(P,28); pr=P['E']; L.append(torch.linalg.norm(pr-tr,dim=1)/torch.linalg.norm(tr,dim=1))
        L=torch.cat(L).cpu().numpy(); return f'med {np.median(L):.4f} p95 {np.percentile(L,95):.3f}'
    print('envelope alone,',env,'| random',sc(random_family),'| bubble',sc(bubble_family),flush=True)
