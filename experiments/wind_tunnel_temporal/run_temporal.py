import os,glob,json,numpy as np,pandas as pd,matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
ROOT='/Users/dhivyasreedhar/Documents/Codex/2026-09-06/th/work/data'; OUT=os.path.dirname(__file__); REP=OUT+'/report';os.makedirs(REP,exist_ok=True)
C=['load_in','rpm_in','current_in','pressure_downwind','pressure_upwind']; L=['LIn','ω̃in','C̃in','P̃dw','P̃up']; lag=20
def load(f): return pd.read_csv(f,usecols=C).replace([np.inf,-np.inf],np.nan).interpolate().ffill().bfill().to_numpy(float)
def mk(a):
 z=(a-a.mean(0))/(a.std(0)+1e-8);return np.array([z[i-lag:i].ravel() for i in range(lag,len(z))]),z[lag:]
def run(a):
 X,y=mk(a); k=int(.7*len(X));m=Ridge(alpha=10).fit(X[:k],y[:k]);p=m.predict(X[k:]);return m,np.array([r2_score(y[k:,j],p[:,j]) for j in range(5)])
def edge(m):
 q=np.abs(m.coef_).reshape(5,lag,5).sum(1);return q/(q.max(1,keepdims=True)+1e-9)
obs=load(ROOT+'/wt_walks_v1/actuators_random_walk_1.csv'); im=np.vstack([load(f)[::20] for f in glob.glob(ROOT+'/wt_intake_impulse_v1/*.csv')]); mo,ro=run(obs);mi,ri=run(im); Xo,yo=mk(obs);Xi,yi=mk(im);ko=int(.7*len(Xo));ki=int(.7*len(Xi));mm=Ridge(alpha=10).fit(np.vstack([Xo[:ko],Xi[:ki]]),np.vstack([yo[:ko],yi[:ki]]));po=mm.predict(Xo[ko:]);pi=mm.predict(Xi[ki:]);rm=np.array([r2_score(yo[ko:,j],po[:,j]) for j in range(5)]);rmi=np.array([r2_score(yi[ki:,j],pi[:,j]) for j in range(5)])
eo,ei,em=edge(mo),edge(mi),edge(mm)
fig,ax=plt.subplots(2,2,figsize=(12,10));
for a,z,t in zip(ax.flat[:3],[eo,ei,em],['Observational','Impulse interventions','Mixed training']):a.imshow(z,cmap='magma',vmin=0,vmax=1);a.set_xticks(range(5),L,rotation=45);a.set_yticks(range(5),L);a.set_title(t+' lagged edge scores');
x=np.arange(5);w=.25;ax[1,1].bar(x-w,ro,w,label='obs');ax[1,1].bar(x,ri,w,label='impulse');ax[1,1].bar(x+w,rmi,w,label='mixed→impulse');ax[1,1].set_xticks(x,L);ax[1,1].set_ylabel('chronological R²');ax[1,1].legend();fig.tight_layout();fig.savefig(REP+'/temporal_edge_heatmaps.png',dpi=200);plt.close(fig)
a=pd.read_csv(glob.glob(ROOT+'/wt_intake_impulse_v1/*.csv')[0],usecols=C).iloc[:2500];fig,ax=plt.subplots(5,1,figsize=(12,9),sharex=True)
for j,(q,l) in enumerate(zip(ax,L)):q.plot(a.iloc[:,j].to_numpy(),lw=.5);q.set_ylabel(l);q.grid(alpha=.2)
fig.tight_layout();fig.savefig(REP+'/impulse_response_timeseries.png',dpi=180);plt.close(fig)
s={'variables':L,'observational_rows':len(obs),'impulse_rows_used':len(im),'lag':lag,'split':'chronological 70/30','r2_observational':dict(zip(L,ro.tolist())),'r2_impulse':dict(zip(L,ri.tolist())),'r2_mixed_on_impulse':dict(zip(L,rmi.tolist())),'edge_scores_observational':eo.tolist(),'edge_scores_impulse':ei.tolist(),'edge_scores_mixed':em.tolist(),'ground_truth_chain':['LIn -> ω̃in','ω̃in -> P̃dw','ω̃in -> P̃up'],'method':'stable lagged Ridge predictor; absolute coefficient mass over 20 past steps'};json.dump(s,open(REP+'/summary.json','w'),indent=2);print(json.dumps({k:s[k] for k in ['observational_rows','impulse_rows_used','r2_observational','r2_impulse','r2_mixed_on_impulse']},indent=2))
