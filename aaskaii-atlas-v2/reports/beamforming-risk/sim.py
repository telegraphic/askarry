"""Toy SKA-Low station beam model: 256 elements, 38 m, pseudo-random layout.
Array factor only (isotropic elements, no mutual coupling)."""
import numpy as np, json
from scipy.signal.windows import taylor
rng=np.random.default_rng(7)
N,D,dmin=256,38.0,1.5
R=D/2
pts=[]
while len(pts)<N:
    r=R*np.sqrt(rng.random()); a=2*np.pi*rng.random()
    p=np.array([r*np.cos(a),r*np.sin(a)])
    if all(np.hypot(*(p-q))>=dmin for q in pts): pts.append(p)
xy=np.array(pts); rad=np.hypot(xy[:,0],xy[:,1])
np.save('layout.npy',xy)
c=299792458.0
def weights(kind,f=None,f0=None):
    if kind=='uniform': return np.ones(N)
    if kind=='gauss_D': s=R; return np.exp(-0.5*(rad/s)**2)        # sigma = station radius
    if kind=='gauss_halfD': s=R/2; return np.exp(-0.5*(rad/s)**2)  # sigma = half radius
    if kind=='taylor28':
        t=taylor(2001,nbar=4,sll=28,norm=True); x=0.5+rad/(2*R)
        return np.interp(x,np.linspace(0,1,2001),t)
    if kind=='gauss_const':  # frequency-scaled: sigma ∝ lambda so FWHM fixed at f0
        s=R*(f0/f); return np.exp(-0.5*(rad/s)**2)
def af(w,f,l,m):
    k=2*np.pi*f/c
    ph=np.exp(1j*k*(np.outer(l.ravel(),xy[:,0])+np.outer(m.ravel(),xy[:,1])))
    return (ph@w).reshape(l.shape)/w.sum()
def eta(w): return w.sum()**2/(N*(w**2).sum())
def metrics(w,f):
    th=np.radians(np.linspace(0,89.5,1800)); phis=np.radians(np.arange(0,360,3))
    T,P=np.meshgrid(th,phis); l=np.sin(T)*np.cos(P); m=np.sin(T)*np.sin(P)
    B=np.abs(af(w,f,l,m))**2
    az=B.mean(0); deg=np.degrees(th)
    i=np.argmax(az<0.5); fwhm=2*np.interp(0.5,[az[i],az[i-1]],[deg[i],deg[i-1]])
    # first null of azimuthal mean
    j=i+np.argmax(np.diff(az[i:])>0); 
    k=j+np.argmax(np.diff(az[j:])<0); fsl=az[k]
    far=deg>3*deg[j]
    wsa=np.sin(T)[:,far]
    farmean=(B[:,far]*wsa).sum()/wsa.sum()
    return dict(fwhm=fwhm,first_null=deg[j],first_sl_db=10*np.log10(fsl),
                far_mean_db=10*np.log10(farmean),eta=eta(w),pred_far_db=10*np.log10(1/(N*eta(w)))), deg, az
out={}; cuts={}
for f in (70e6,160e6):
    for kind in ('uniform','gauss_D','taylor28','gauss_halfD'):
        w=weights(kind); mtr,deg,az=metrics(w,f)
        out[f'{int(f/1e6)}_{kind}']=mtr; cuts[f'{int(f/1e6)}_{kind}']=az.tolist(); cuts['deg']=deg.tolist()
        print(int(f/1e6),kind,{k:round(v,3) for k,v in mtr.items()})
json.dump(out,open('metrics.json','w'),indent=1); json.dump(cuts,open('cuts.json','w'))
# chromaticity: response vs frequency at fixed angles
fr=np.linspace(120e6,180e6,121); angs=[3.0,6.0,25.0]; chrom={}
for kind in ('uniform','gauss_const'):
    for a in angs:
        th=np.radians(a); phis=np.radians(np.arange(0,360,2))
        l=np.sin(th)*np.cos(phis); m=np.sin(th)*np.sin(phis)
        vals=[]
        for f in fr:
            w=weights(kind,f,150e6) if kind=='gauss_const' else weights(kind)
            vals.append(np.abs(af(w,f,l,m))[0]**2)   # single azimuth phi=0
        chrom[f'{kind}_{a}']=np.array(vals).tolist()
chrom['freq']=(fr/1e6).tolist(); json.dump(chrom,open('chrom.json','w'))
# null steering at 110 MHz: 3 RFI directions
f=110e6; k=2*np.pi*f/c
dirs=[(50,40),(62,200),(35,300)]  # zenith angle, azimuth deg
A=np.array([np.exp(1j*k*(np.sin(np.radians(z))*np.cos(np.radians(p))*xy[:,0]+np.sin(np.radians(z))*np.sin(np.radians(p))*xy[:,1])) for z,p in dirs]).T
w0=weights('gauss_D').astype(complex)
Pn=np.eye(N)-A@np.linalg.pinv(A)
# conjugate convention: AF = sum w * exp(+ik r.s); null requires sum w a = 0 -> project w onto null space of A^T
At=A.T; Pn=np.eye(N)-At.conj().T@np.linalg.solve(At@At.conj().T,At)
w1=Pn.conj()@w0 if False else (np.eye(N)-np.linalg.pinv(At)@At)@w0
def resp(w,z,p):
    l=np.array([[np.sin(np.radians(z))*np.cos(np.radians(p))]]); m=np.array([[np.sin(np.radians(z))*np.sin(np.radians(p))]])
    return np.abs((np.exp(1j*k*(l.ravel()[:,None]*xy[:,0]+m.ravel()[:,None]*xy[:,1]))@w))[0]**2
g0=abs(w0.sum())**2; g1=abs(w1.sum())**2
ns={'dirs':dirs,'before_db':[10*np.log10(resp(w0,z,p)/g0) for z,p in dirs],
    'after_db':[10*np.log10(max(resp(w1,z,p),1e-30)/g1) for z,p in dirs],
    'boresight_loss_db':10*np.log10((abs(w1.sum())**2/(np.abs(w1)**2).sum())/(abs(w0.sum())**2/(np.abs(w0)**2).sum())),
    }
# cut along azimuth of first RFI dir
zz=np.linspace(-89,89,3561); pp=np.radians(dirs[0][1])
l=np.sin(np.radians(zz))*np.cos(pp); m=np.sin(np.radians(zz))*np.sin(pp)
ph=np.exp(1j*k*(np.outer(l,xy[:,0])+np.outer(m,xy[:,1])))
ns['cut_z']=zz.tolist(); ns['cut_before']=(10*np.log10(np.abs(ph@w0)**2/g0)).tolist(); ns['cut_after']=(10*np.log10(np.abs(ph@w1)**2/g1+1e-12)).tolist()
print({k:v for k,v in ns.items() if not k.startswith('cut')})
json.dump(ns,open('nulls.json','w'))
