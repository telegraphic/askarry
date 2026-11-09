"""Figures for the beam-apodization report. Run after sim.py, nullmc.py, chrom.py.
risks.json is the AASKAII Atlas v2 systematic risk register data file."""
import json, sys, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import glob, os
for f in glob.glob(os.path.expanduser('~/.fonts/Libertinus*.otf')): fm.fontManager.addfont(f)
plt.rcParams.update({'font.family':'Libertinus Sans','font.size':9,'axes.linewidth':0.6,
  'axes.edgecolor':'#555','xtick.major.width':0.6,'ytick.major.width':0.6,'axes.grid':True,
  'grid.color':'#e4e4e4','grid.linewidth':0.5,'axes.spines.top':False,'axes.spines.right':False,
  'legend.frameon':False,'pdf.fonttype':42,'mathtext.fontset':'custom','mathtext.rm':'Libertinus Sans'})
BLUE,ORANGE,GREEN,GREY='#2A5FA8','#C8562A','#1B8F68','#9A9893'
risks=json.load(open(sys.argv[1]))['items']
lever={'Beam sidelobes':'d','Primary-beam model':'d','Survey-speed limitation':'d',
 'Timing / synchronisation':'n','ML sim-to-real gap':'n'}
# fig 1 register
r=sorted(risks,key=lambda x:x['n_papers'])
fig,ax=plt.subplots(figsize=(5.6,3.9))
col={'d':BLUE,'p':GREEN,'n':GREY}
lab={'Polarization leakage':'Polarisation leakage','Data rate / volume':'Data rate or volume','Bandpass / gain stability':'Bandpass or gain stability','Timing / synchronisation':'Timing or synchronization'}
for i,x in enumerate(r):
    c=col[lever.get(x['label'],'p')]
    ax.barh(i,x['n_papers'],color=c,height=0.62,lw=0)
    ax.text(x['n_papers']+0.6,i,f"{x['n_papers']}",va='center',fontsize=7.5,color='#444')
ax.set_yticks(range(len(r))); ax.set_yticklabels([lab.get(x['label'],x['label']) for x in r],fontsize=8)
ax.set_xlabel('AASKAII chapters naming the risk at least twice'); ax.grid(axis='y',visible=False)
ax.set_xlim(0,45)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=BLUE,label='direct lever'),Patch(color=GREEN,label='partial lever'),Patch(color=GREY,label='little or none')],
          loc='upper left',bbox_to_anchor=(1.0,0.25),fontsize=8,title='Beamforming as a remedy',title_fontsize=8)
fig.tight_layout(); fig.savefig('fig_register.pdf'); fig.savefig('fig_register.png',dpi=110)
# fig 2 beam cuts at 160 MHz and 70 MHz
cuts=json.load(open('cuts.json')); met=json.load(open('metrics.json')); deg=np.array(cuts['deg'])
fig,axs=plt.subplots(1,2,figsize=(6.4,2.7),sharey=True)
sty=[('uniform','uniform',GREY,'-'),('gauss_D','Gaussian, σ = R',GREEN,'-'),('taylor28','Taylor, 28 dB',BLUE,'-'),('gauss_halfD','Gaussian, σ = R/2',ORANGE,'-')]
for ax,f in zip(axs,(70,160)):
    for k,l,c,ls in sty:
        y=10*np.log10(np.maximum(cuts[f'{f}_{k}'],1e-6)); ax.plot(deg,y,color=c,lw=0.9,ls=ls,label=l)
    ax.axhline(-24.08,color='#333',lw=0.6,ls=':'); 
    ax.text((88 if f==70 else 44),-23.3,'1/N',ha='right',fontsize=7.5,color='#333')
    ax.set_xlim(0,90 if f==70 else 45); ax.set_ylim(-45,1); ax.set_xlabel('angle from beam centre (deg)')
    ax.set_title(f'{f} MHz',fontsize=9,loc='left')
axs[0].set_ylabel('azimuthal mean power (dB)')
axs[1].legend(loc='upper left',bbox_to_anchor=(1.0,1.0),fontsize=7.5)
fig.tight_layout(); fig.savefig('fig_beams.pdf'); fig.savefig('fig_beams.png',dpi=110)
# fig 3 chromaticity
ch=json.load(open('chrom.json')); fq=np.array(ch['freq'])
fig,axs=plt.subplots(1,2,figsize=(6.4,2.6))
for ax,ang in zip(axs,(2.0,4.0)):
    for k,l,c in (('uniform','uniform weights',GREY),('gauss_const','frequency-scaled Gaussian',BLUE)):
        ax.plot(fq,10*np.log10(ch[f'{k}_{ang}_mean']),color=c,lw=1,label=l)
    ax.set_title(f'{ang:.0f}° from beam centre',fontsize=9,loc='left'); ax.set_xlabel('frequency (MHz)')
axs[0].set_ylabel('beam power (dB)')
axs[1].legend(loc='upper left',bbox_to_anchor=(1.0,1.0),fontsize=7.5)
fig.tight_layout(); fig.savefig('fig_chrom.pdf'); fig.savefig('fig_chrom.png',dpi=110)
# fig 4 nulls
ns=json.load(open('nulls.json')); mc=json.load(open('nullmc.json'))
fig,axs=plt.subplots(1,2,figsize=(6.4,2.6),gridspec_kw={'width_ratios':[1.6,1]})
z=np.array(ns['cut_z']); ax=axs[0]
ax.plot(z,ns['cut_before'],color=GREY,lw=0.8,label='Gaussian taper')
ax.plot(z,np.maximum(ns['cut_after'],-80),color=BLUE,lw=0.8,label='with three nulls')
ax.axvline(50,color=ORANGE,lw=0.8,ls='--'); ax.text(51,-6,'interferer',color=ORANGE,fontsize=7.5)
ax.set_xlim(0,89); ax.set_ylim(-80,2); ax.set_xlabel('zenith angle along interferer azimuth (deg)'); ax.set_ylabel('power (dB)')
ax.legend(loc='lower left',fontsize=7.5)
ax=axs[1]; e=np.array([float(k) for k in mc]); v=np.array(list(mc.values()))
ax.fill_between(100*e,v[:,1],v[:,2],color=BLUE,alpha=0.18,lw=0); ax.plot(100*e,v[:,0],'o-',color=BLUE,ms=3,lw=1)
ax.set_xlabel('element gain and phase error, rms (%)'); ax.set_ylabel('suppression (dB)'); ax.set_ylim(-45,0); ax.set_xlim(0,11)
fig.tight_layout(); fig.savefig('fig_nulls.pdf'); fig.savefig('fig_nulls.png',dpi=110)
