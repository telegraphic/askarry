import numpy as np, json
rng=np.random.default_rng(3)
xy=np.load('layout.npy'); N=len(xy); R=19.0; rad=np.hypot(*xy.T); c=299792458.0
f=110e6; k=2*np.pi*f/c
dirs=[(50,40),(62,200),(35,300)]
def a(z,p): s=np.sin(np.radians(z)); return np.exp(1j*k*s*(np.cos(np.radians(p))*xy[:,0]+np.sin(np.radians(p))*xy[:,1]))
At=np.array([a(*d) for d in dirs])
w0=np.exp(-0.5*(rad/R)**2).astype(complex)
w1=(np.eye(N)-np.linalg.pinv(At)@At)@w0
res={}
for eps in (0.01,0.03,0.05,0.10):
    sup=[]
    for _ in range(200):
        g=(1+eps*rng.standard_normal(N))*np.exp(1j*eps*rng.standard_normal(N))  # amp & phase (rad) errors
        for d in dirs:
            b=abs(np.sum(g*w0*a(*d)))**2; af=abs(np.sum(g*w1*a(*d)))**2
            sup.append(10*np.log10(af/b))
    res[eps]=(float(np.median(sup)),float(np.percentile(sup,10)),float(np.percentile(sup,90)))
    print(eps,res[eps])
json.dump({str(k):v for k,v in res.items()},open('nullmc.json','w'))
