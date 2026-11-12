"""Figures of the two logit routes for manuscript/binary_outcome_sketch.tex (2026-11-12).

Reads the saved runs under experiments/runs/ (binary_outcome_routes, binary_outcome_routes_vlbw/cost_*,
binary_outcome_routes_vlbw_pooled_c*) and writes binary_outcome_routes_logit.png,
binary_outcome_vlbw_logit.png and binary_outcome_vlbw_pooled_logit.png into the folder given:

    python experiments/scripts/binary_outcome_logit_figures.py ../manuscript-worktree/figures
"""
import json, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R='/Users/zoudj/Documents/Codex/PerfRDD/code/experiments/runs/'
import sys
OUT = (sys.argv[1] if len(sys.argv) > 1 else '.').rstrip('/') + '/'
BG='#fcfcfb'; BL='#2a78d6'; OR='#eb6834'
plt.rcParams.update({'figure.facecolor':BG,'axes.facecolor':BG,'axes.edgecolor':'#888','axes.grid':True,'grid.color':'#e6e6e3','grid.linewidth':0.6,'axes.spines.top':False,'axes.spines.right':False,'font.size':9,'axes.labelcolor':'#444','xtick.color':'#555','ytick.color':'#555','lines.linewidth':1.2,'savefig.facecolor':BG})
def ld(p): return json.load(open(R+p+'/summary.json'))
def get(rows,route): return [r for r in rows if r['route']==route][0]
def styl(ax): ax.set_axisbelow(True)

# Figure 1
s=ld('binary_outcome_routes')
order=[('chile','any_2025','Chile\nany 2025'),('chile','univ_2024','Chile\nuniv. 2024'),('chile','univ_2025','Chile\nuniv. 2025'),('gpa','grad5','GPA\ngrad. 5 yr'),('gpa','grad6','GPA\ngrad. 6 yr'),('gpa','return2','GPA\nreturn yr 2'),('gpa','stay','GPA\nstay')]
fig,axs=plt.subplots(2,7,figsize=(10,5.2))
for j,(st,oc,t) in enumerate(order):
    rows=[r for r in s if r['setting']==st and r['outcome']==oc]
    sp=get(rows,'logit_sp'); em=get(rows,'logit_emp')
    x=np.array(sp['curves']); p0=sp['phi0']
    i0=np.argmin(abs(x-p0))
    for k,key in enumerate(['U','tau']):
        ax=axs[k,j]; styl(ax)
        f=100 if key=='U' else 1
        ys=np.array(sp[key])*f; ye=np.array(em[key])*f
        if key=='U': ys=ys-ys[i0]; ye=ye-ye[i0]
        ax.plot(x,ys,color=BL,label='spline tails'); ax.plot(x,ye,color=OR,ls='--',label='empirical tails')
        ax.axvline(p0,color='#999',lw=0.7,ls=':')
        if key=='tau': ax.axhline(0,color='#aaa',lw=0.7)
        if k==0: ax.set_title(t,fontsize=9)
        ax.tick_params(labelsize=8)
        ax.set_xlabel('threshold' if k==1 else '',fontsize=9)
        if j==0: ax.set_ylabel('utility minus deployed, points' if k==0 else 'net effect at the threshold',fontsize=9)
axs[0,0].legend(fontsize=7,frameon=False,loc='lower left')
fig.tight_layout(); fig.savefig(OUT+'binary_outcome_routes_logit.png',dpi=130); plt.close(fig)

# Figure 2: 2002
costs=['0','0.01','0.02','0.03']; cols=['#2a78d6','#eb6834','#2c9a5f','#b03a48']
fig,axs=plt.subplots(1,2,figsize=(10,3.9))
r0=ld('binary_outcome_routes_vlbw/cost_0')
sp=get(r0,'logit_sp'); em=get(r0,'logit_emp'); g=-np.array(sp['curves'])
ax=axs[0]; styl(ax)
ax.plot(g,100*np.array(sp['tau']),color=BL,label='spline tails'); ax.plot(g,100*np.array(em['tau']),color=OR,ls='--',label='empirical tails')
ax.axhline(0,color='#aaa',lw=0.7); ax.axvline(1500,color='#999',lw=0.7,ls=':')
ax.set_xlabel('threshold, grams (treated below)'); ax.set_ylabel('survival gain of the infant at the threshold, points'); ax.legend(frameon=False,fontsize=8)
ax=axs[1]; styl(ax)
for c,col in zip(costs,cols):
    rr=ld('binary_outcome_routes_vlbw/cost_'+c)
    for rt,ls in [('logit_sp','-'),('logit_emp','--')]:
        d=get(rr,rt); U=100*np.array(d['U']); 
        ax.plot(g,U-U.max(),color=col,ls=ls,lw=1.2,label=(f'cost {float(c):.2f}: optimum {-d["phi_pooled"]:.0f} g, spline tails' if rt=='logit_sp' else None))
ax.axvline(1500,color='#999',lw=0.7,ls=':'); ax.set_xlabel('threshold, grams (treated below)'); ax.set_ylabel('utility minus its maximum, points of survival')
ax.legend(frameon=False,fontsize=7,loc='lower center')
for a in axs: a.invert_xaxis()
fig.tight_layout(); fig.savefig(OUT+'binary_outcome_vlbw_logit.png',dpi=130); plt.close(fig)

# Figure 3: pooled
fig,axs=plt.subplots(1,2,figsize=(10,3.9))
pc={'0':'0','0.01':'0.01','0.02':'0.02','0.03':'0.03'}
pe=get(ld('binary_outcome_routes_vlbw_pooled_c0'),'logit_emp'); se=get(r0,'logit_emp')
ax=axs[0]; styl(ax)
ax.plot(g,100*np.array(pe['tau']),color=BL,label='1995 to 2002 pooled'); ax.plot(g,100*np.array(se['tau']),color=OR,label='2002 only')
for c in [0.01,0.02,0.03]:
    ax.axhline(100*c,color='#777',lw=0.7,ls=':'); ax.text(1795,100*c+0.12,f'cost {c:.2f}',fontsize=7,color='#555')
ax.axvline(1500,color='#999',lw=0.7); ax.set_xlabel('threshold, grams (treated below)'); ax.set_ylabel('survival gain at the threshold, points'); ax.legend(frameon=False,fontsize=8,loc='upper left')
ax=axs[1]; styl(ax)
pe2=get(ld('binary_outcome_routes_vlbw_pooled_c0.02'),'logit_emp'); se2=get(ld('binary_outcome_routes_vlbw/cost_0.02'),'logit_emp')
i0=int(np.argmin(abs(g-1500)))
off={'1995 to 2002 pooled':(-8,10),'2002 only':(26,-2)}
for d,col,lab in [(pe2,BL,'1995 to 2002 pooled'),(se2,OR,'2002 only')]:
    U=100*np.array(d['U']); U=U-U[i0]; ax.plot(g,U,color=col,label=lab)
    k=int(np.argmax(U)); ax.plot(g[k],U[k],'o',color=col,ms=5); ax.annotate(f'{-d["phi_pooled"]:.0f} g',(g[k],U[k]),textcoords='offset points',xytext=off[lab],ha='center',fontsize=8,color=col)
ax.axvline(1500,color='#999',lw=0.7); ax.set_xlabel('threshold, grams (treated below)'); ax.set_ylabel('utility gain over the 1500 g rule, points of survival'); ax.set_ylim(-0.11,0.02); ax.legend(frameon=False,fontsize=8,loc='lower center')
for a in axs: a.invert_xaxis()
fig.tight_layout(); fig.savefig(OUT+'binary_outcome_vlbw_pooled_logit.png',dpi=130); plt.close(fig)
