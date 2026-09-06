"""Observational vs. hard-intervention light-tunnel experiment.

Uses only standard-config columns R,G,B,theta_1,theta_2,I1,I2,I3. The model is
the continuous CatFormer adapter from ../light_tunnel/run.py. It is trained as
masked regression; intervention labels are never used as input features.
"""
import json, shutil, hashlib, argparse
from pathlib import Path
import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
import optax
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
import sys
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'light_tunnel'))
from run import ContinuousCatFormer, batch_predict

COLS=['red','green','blue','pol_1','pol_2','ir_1','ir_2','ir_3']
LABELS=['R','G','B',r'$\theta_1$',r'$\theta_2$',r'$\tilde I_1$',r'$\tilde I_2$',r'$\tilde I_3$']
TARGETS=[5,6,7]
PARENTS=[3,4]

class ContinuousCatFormer8(ContinuousCatFormer):
    def __init__(self, length, key):
        self.vocab_size=8; d=18+length; k1,k2=jax.random.split(key)
        self.A=[1e-3*jax.random.normal(k1,(3,d,d)),1e-3*jax.random.normal(k2,(1,4*d,4*d))]; self.W=jnp.zeros((8*d,1))
    def embed(self,tokens):
        ids,values,observed=tokens[...,0].astype(int),tokens[...,1],tokens[...,2]; identity=jax.nn.one_hot(ids,8); pos=jnp.broadcast_to(jnp.eye(tokens.shape[-2]),(*tokens.shape[:-1],tokens.shape[-2])); return jnp.concatenate([identity,identity*(values*observed)[...,None],observed[...,None],jnp.ones_like(observed[...,None]),pos],-1)

def load(root):
    files={n:pd.read_csv(root/(n+'.csv')) for n in ['uniform_reference','uniform_pol_1_strong','uniform_pol_2_strong']}
    for n,d in files.items():
        assert set(d.config)=={'standard'} and set(COLS)<=set(d.columns)
    return {n:d[COLS].to_numpy('float32') for n,d in files.items()}

def make_prompts(query,context,targets,rng,mode='all',support=2):
    b=len(query); n=query.shape[1]; rows=context[rng.integers(len(context),size=(b,support))]; order=np.argsort(rng.random((b,support,n)),axis=-1); st=np.stack([order,np.take_along_axis(rows,order,-1),np.ones_like(rows)],-1); scores=rng.random((b,n));scores[np.arange(b),targets]=2; qo=np.argsort(scores,-1); qv=np.take_along_axis(query,qo,-1); obs=(qo!=targets[:,None]); qt=np.stack([qo,qv*obs,obs],-1); return np.concatenate([st.reshape(b,support*n,3),qt],1).astype('float32'), query[np.arange(b),targets].astype('float32')

def train_model(train_rows, val, seed, steps=1800, mode='obs'):
    # Training examples are sampled only from the declared regime(s).
    rng=np.random.default_rng(seed); model=ContinuousCatFormer8(24,jax.random.PRNGKey(seed))
    opt=optax.chain(optax.clip_by_global_norm(1.),optax.adamw(optax.cosine_decay_schedule(.0003,steps,alpha=.1),weight_decay=1e-4)); state=opt.init(model)
    tasks=np.array(TARGETS);batch=64
    @jax.jit
    def step(m,st,x,y):
        loss,g=jax.value_and_grad(lambda mm:jnp.mean((mm(x)-y)**2))(m);u,st=opt.update(g,st,m);return optax.apply_updates(m,u),st
    vx=[]
    for t in tasks:
        x,y=make_prompts(val,train_rows,np.full(len(val),t),np.random.default_rng(seed+100+t),'all');vx.append((x,y))
    best=1e99;bestm=model
    for i in range(steps+1):
        if i%300==0 or i==steps:
            loss=np.mean([np.mean((batch_predict(model,x)-y)**2) for x,y in vx])
            if loss<best:best,bestm=loss,model
        if i==steps:break
        q=train_rows[rng.integers(len(train_rows),size=batch)];t=rng.choice(tasks,size=batch)
        x,y=make_prompts(q,train_rows,t,rng,'all');model,state=step(model,state,jnp.asarray(x),jnp.asarray(y))
    return bestm,best

def predict(model, rows, context, seed):
    p=np.zeros((len(rows),len(TARGETS)))
    for k,t in enumerate(TARGETS):
        x,y=make_prompts(rows,context,np.full(len(rows),t),np.random.default_rng(seed+t),'all')
        p[:,k]=batch_predict(model,x)
    return p

def ridge(train, test):
    # Includes cos^2(theta1-theta2) and linear RGB terms; transparent physical control.
    def feat(x):
        d=np.deg2rad(x[:,3]-x[:,4]);return np.c_[x[:,:5],np.cos(d)**2,x[:,:3]*np.cos(d)[:,None]**2]
    out=[]
    for t in TARGETS:
        m=Ridge(alpha=1.).fit(feat(train),train[:,t]);out.append(m.predict(feat(test)))
    return np.array(out).T

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--steps',type=int,default=1800);ap.add_argument('--seeds',type=int,nargs='+',default=[0,1,2]);args=ap.parse_args()
    droot=ROOT/'data/lt_interventions_standard_v1';data=load(droot);out=ROOT/'results';out.mkdir(exist_ok=True)
    obs=data['uniform_reference']; theta1=data['uniform_pol_1_strong'];theta2=data['uniform_pol_2_strong']
    # Normalize from observational rows only. Keep raw copies for physical diagnostics.
    mu,s=obs.mean(0),obs.std(0); z={k:(v-mu)/s for k,v in data.items()}
    # 80/20 splits within every regime; intervention rows are never mixed into obs-only training.
    rng=np.random.default_rng(123); splits={}
    for k,v in z.items():
        ix=rng.permutation(len(v));splits[k]=(v[ix[:int(.8*len(v))]],v[ix[int(.8*len(v)):]])
    summary=dict(dataset='lt_interventions_standard_v1',configuration='light tunnel / standard',columns=COLS,labels=LABELS,
      regimes={k:len(v) for k,v in data.items()},normalization={'mean':mu.tolist(),'std':s.tolist()},
      data_sha256={k:hashlib.sha256((droot/(k+'.csv')).read_bytes()).hexdigest() for k in data},
      zip_md5=hashlib.md5((ROOT/'data/lt_interventions_standard_v1.zip').read_bytes()).hexdigest(),
      ground_truth={'critical_edges':[['pol_1','ir_3'],['pol_2','ir_3']]},runs=[],method='continuous CatFormer masked regression; no intervention labels or camera variables')
    for setting in ['observational_only','observational_plus_interventions']:
      train=np.concatenate([splits['uniform_reference'][0]] + ([splits['uniform_pol_1_strong'][0],splits['uniform_pol_2_strong'][0]] if setting.endswith('interventions') else []))
      evals={'observational':splits['uniform_reference'][1],'do_theta1':splits['uniform_pol_1_strong'][1],'do_theta2':splits['uniform_pol_2_strong'][1]}
      lin={k:ridge(train,v) for k,v in evals.items()}
      linmet={k:{LABELS[t]:float(r2_score(v[:,t],lin[k][:,j])) for j,t in enumerate(TARGETS)} for k,v in evals.items()}
      for seed in args.seeds:
        model,val=train_model(train,splits['uniform_reference'][1],seed,args.steps)
        for regime,v in evals.items():
          pred=predict(model,v,train,9000+seed)
          metrics={LABELS[t]:float(r2_score(v[:,t],pred[:,j])) for j,t in enumerate(TARGETS)}
          summary['runs'].append(dict(setting=setting,seed=seed,regime=regime,validation_mse=val,metrics=metrics,linear_metrics=linmet[regime]))
        np.savez_compressed(out/f'{setting}_seed{seed}.npz',model_W=np.asarray(model.W),model_A0=np.asarray(model.A[0]),model_A1=np.asarray(model.A[1]))
    # A direct Malus-law diagnostic: does I3 become predictable when theta pair is jointly supplied?
    d=np.deg2rad(obs[:,3]-obs[:,4]);malus=np.cos(d)**2
    summary['malus_diagnostic']={'corr_I3_cos2_theta_difference':float(np.asarray(np.corrcoef(obs[:,7],malus)[0,1])),'corr_I3_theta1':float(np.corrcoef(obs[:,7],obs[:,3])[0,1]),'corr_I3_theta2':float(np.corrcoef(obs[:,7],obs[:,4])[0,1])}
    (out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False,default=float));print(json.dumps(summary['malus_diagnostic']))

if __name__=='__main__':main()
