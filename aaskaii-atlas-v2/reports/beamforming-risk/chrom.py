import numpy as np, json
xy=np.load('layout.npy'); N=len(xy); R=19.0; rad=np.hypot(*xy.T); c=299792458.0
fr=np.linspace(120e6,180e6,241); out={'freq':(fr/1e6).tolist()}
def w(kind,f):
    if kind=='uniform': return np.ones(N)
    s=(R/2)*(120e6/f); return np.exp(-0.5*(rad/s)**2)
for kind in ('uniform','gauss_const'):
    for ang in (2.0,4.0,20.0):
        phis=np.radians(np.arange(0,360,1)); th=np.radians(ang)
        l=np.sin(th)*np.cos(phis); m=np.sin(th)*np.sin(phis); vals=[]
        for f in fr:
            k=2*np.pi*f/c; ww=w(kind,f)
            B=np.abs(np.exp(1j*k*(np.outer(l,xy[:,0])+np.outer(m,xy[:,1])))@ww/ww.sum())**2
            vals.append([B.mean(), B[0]])
        v=np.array(vals); out[f'{kind}_{ang}_mean']=v[:,0].tolist(); out[f'{kind}_{ang}_phi0']=v[:,1].tolist()
        print(kind,ang,'azmean span %.2f dB'%(10*np.log10(v[:,0].max()/v[:,0].min())),'phi0 span %.2f dB'%(10*np.log10(v[:,1].max()/v[:,1].min())))
    eta=[w(kind,f).sum()**2/(N*(w(kind,f)**2).sum()) for f in (120e6,150e6,180e6)]; print(kind,'eta',np.round(eta,3))
json.dump(out,open('chrom.json','w'))
