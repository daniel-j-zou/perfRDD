"""Stylized simulation: does the ridge of fit_logit make the GLM arm fit inconsistent? (2026-11-17)

Written by a Claude review subagent for the proof review of Section 2 of
manuscript/binary_outcome_vlbw.tex; moved here by Claude. Design: X in R^3 standardized,
eta ~ N(0,1) independent of X, Q = gamma'Xtilde + eta, 17% treated, Y ~ Bernoulli(F(a_D)) with
c0 falling 4 to 2 and c1 3.8 to 2.5 across J = [-1.5, 1.5], beta2 = -0.3; true eta (no
generated regressor). Fits the logit on both arms as fit_logit does (cubic B-splines on equally
spaced knots, K = round(n_min^{11/60}), Newton with step halving) under four ridges:
1e-3 n/p (the code), 0, sqrt(n)/10 and sqrt(n)/1000.

    python experiments/scripts/binary_ridge_sim.py 8 20000 80000 320000 1280000

Output of that call (8 replications) is recorded in the research log of 2026-11-17.
"""
import numpy as np, sys, time
from scipy.interpolate import BSpline
from scipy.special import expit
J=(-1.5,1.5)
gam=np.array([1.5,1.0,0.6]); b1=np.array([-0.2,0.1,0.1]); b2=-0.3
s=lambda e: np.sin(np.pi*e/3)
c0=lambda e: 3-1.0*s(e); c1=lambda e: 3.15-0.65*s(e)
def draw(n,rng):
    X=rng.standard_normal((n,3)); X=(X-X.mean(0))/X.std(0)
    eta=rng.standard_normal(n); return X,eta
rng0=np.random.default_rng(0)
Xp,ep=draw(4_000_000,rng0); phi0=np.quantile(Xp@gam+ep,0.83)
def arm_val(d,e,X): return np.where(d==1,c1(e)+X@b1+X[:,0]*b2,c0(e)+X@b1)
def basis(e,K):
    interior=np.linspace(J[0],J[1],K+2)[1:-1]
    t=np.concatenate([np.repeat(J[0],4),interior,np.repeat(J[1],4)])
    return BSpline.design_matrix(np.clip(e,*J),t,3).toarray(),t
def fit(X,eta,D,Y,ridge=None,lam_abs=None):
    k=(eta>=J[0])&(eta<=J[1]); X,eta,D,Y=X[k],eta[k],D[k],Y[k]
    n1=int(D.sum()); K=max(3,int(round(min(n1,len(D)-n1)**(11/60))))
    B,t=basis(eta,K); nb=B.shape[1]
    A=np.column_stack([(1-D)[:,None]*B,D[:,None]*B,X,D[:,None]*X[:,:1]])
    n,p=A.shape; lam=ridge*n/p if ridge is not None else lam_abs(n)
    th=np.zeros(p)
    for d,sl in ((0,slice(0,nb)),(1,slice(nb,2*nb))):
        m=np.clip(Y[D==d].mean(),1e-3,1-1e-3); th[sl]=np.log(m/(1-m))
    f=lambda th:float(np.sum(Y*(A@th)-np.logaddexp(0,A@th)))-0.5*lam*th@th
    ll=f(th)
    for _ in range(60):
        m=expit(A@th); g=A.T@(Y-m)-lam*th
        H=(A*(m*(1-m))[:,None]).T@A+lam*np.eye(p); st=np.linalg.lstsq(H,g,rcond=1e-12)[0]
        tt=1.
        while tt>1e-4:
            new=th+tt*st; l2=f(new)
            if l2>=ll-1e-10: break
            tt/=2
        th=new; done=abs(l2-ll)<1e-9*(1+abs(l2)); ll=l2
        if done: break
    q=3
    return dict(w0=th[:nb],w1=th[nb:2*nb],beta1=th[2*nb:2*nb+q],beta2=th[2*nb+q:],t=t,lam=lam,pen=lam*th@th/(2*n),p=p,K=K,n=n,pen_obs=1e-3*th@th/(2*p))
def chat(fi,e,d):
    return BSpline(fi['t'],fi['w1'] if d else fi['w0'],3,extrapolate=False)(np.clip(e,*J))
# evaluation units near threshold
rng1=np.random.default_rng(1); Xv,ev=draw(3_000_000,rng1); Qv=Xv@gam+ev
nr=(np.abs(Qv-phi0)<0.1)&(ev>=J[0])&(ev<=J[1]); Xn,en=Xv[nr],ev[nr]
tru_eff=np.mean(expit(c1(en)+Xn@b1+Xn[:,0]*b2)-expit(c0(en)+Xn@b1))
grid=np.linspace(-1.2,1.2,121)
def err(fi):
    e2=fi['beta2'][0]-b2
    sup0=np.max(np.abs(chat(fi,grid,0)-c0(grid))); sup1=np.max(np.abs(chat(fi,grid,1)-c1(grid)))
    eff=np.mean(expit(chat(fi,en,1)+Xn@fi['beta1']+Xn[:,0]*(fi['beta2'][0]+fi['beta1'][0]*0+0))-expit(chat(fi,en,0)+Xn@fi['beta1']))
    return e2,sup0,sup1,eff-tru_eff
if __name__=="__main__":
    ns=[int(a) for a in sys.argv[2:]]; R=int(sys.argv[1])
    print("phi0",phi0,"true near-thr effect",tru_eff,"treated share(pop)",np.mean(Xp@gam+ep>phi0))
    specs={"i  1e-3n/p":dict(ridge=1e-3),"ii lam=0":dict(ridge=0.0),"iii sqrt(n)/10":dict(lam_abs=lambda n:np.sqrt(n)/10),"iv sqrt(n)/1000":dict(lam_abs=lambda n:np.sqrt(n)/1000)}
    for n in ns:
        res={k:[] for k in specs}; pens=[]
        t0=time.time()
        for r in range(R):
            rng=np.random.default_rng(1000*r+n%997); X,eta=draw(n,rng)
            D=(X@gam+eta>phi0).astype(float)
            a=np.where(D==1,c1(eta)+X@b1+X[:,0]*b2,c0(eta)+X@b1)
            Y=(rng.random(n)<expit(a)).astype(float)
            for k,sp in specs.items():
                fi=fit(X,eta,D,Y,**sp); res[k].append(err(fi))
                if k.startswith("i "): pens.append(fi['pen_obs']); Kn=fi['K']; pp=fi['p']; nfit=fi['n']
        print(f"n={n} nfit={nfit} K={Kn} p={pp} treated={D.mean():.3f} time={time.time()-t0:.0f}s")
        for k,v in res.items():
            v=np.array(v); print(f"  {k:15s} "+"  ".join(f"{nm}: {v[:,j].mean():+.4f}({v[:,j].std(ddof=1):.4f})" for j,nm in enumerate(["b2err","sup_c0","sup_c1","effErr"])))
        print(f"  per-obs penalty (i): mean {np.mean(pens):.5f}")
