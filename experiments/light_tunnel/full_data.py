"""Exhaustive row/target training, five blocked folds, full-data fit and actual weight plots.

Uses the existing continuous CatFormer adapter. No subsampled validation/test metrics.
"""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/light-tunnel-mpl')
import argparse,json,time,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
import optax
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from run import COLS,TRUTH,ContinuousCatFormer,batch_predict,episodes,ranking,binned_chi_square

ROOT=Path(__file__).resolve().parent

@jax.jit
def first_attention(m,tokens):
    x=m.embed(tokens)
    # Only the final query is needed; avoid forming T x T logits.
    logits=jnp.einsum('bd,hde,bse->bhs',x[:,-1,:],m.A[0],x)
    return jax.nn.softmax(logits,axis=-1)

def prompts(query,context,targets,rng,mode,query_indices=None):
    x,y=episodes(query,context,targets,rng,2,mode)
    # Explicitly prevent a training query row appearing as its own support example.
    if query_indices is not None:
        support_indices=rng.integers(len(context)-1,size=(len(query),2))
        support_indices += support_indices >= query_indices[:,None]
        ids=x[:,:14,0].astype(int).reshape(-1,2,7)
        values=np.take_along_axis(context[support_indices],ids,-1)
        x[:,:14,1]=values.reshape(-1,14)
        assert np.all(support_indices!=query_indices[:,None])
    return x,y

def evaluate(model,rows,train,mode,seed):
    tasks=np.arange(3,7) if mode=='rgb' else np.arange(7)
    pred=np.full((len(rows),7),np.nan)
    attention=np.zeros((3,7,7))
    # All rows for every eligible target, never subsampled.
    for t in tasks:
        rng=np.random.default_rng(seed+int(t))
        x,y=prompts(rows,train,np.full(len(rows),t),rng,mode)
        pred[:,t]=batch_predict(model,x)
        count=0
        for i in range(0,len(x),128):
            xx=x[i:i+128];n=len(xx)
            aa=np.asarray(first_attention(model,jnp.asarray(np.pad(xx,((0,128-n),(0,0),(0,0))))))[:n]
            for source in range(7):
                attention[:,source,t]+=np.sum(aa[:,:,-7:]*(xx[:,None,-7:,0]==source),axis=(0,2))
            count+=n
        attention[:,:,t]/=count
    return pred,attention

def fit(train,val,mode,seed,epochs,batch,out):
    tasks=np.arange(3,7) if mode=='rgb' else np.arange(7)
    n_examples=len(train)*len(tasks)
    nsteps=int(np.ceil(n_examples/batch))*epochs
    model=ContinuousCatFormer(21,jax.random.PRNGKey(seed))
    opt=optax.chain(optax.clip_by_global_norm(1.),optax.adamw(optax.cosine_decay_schedule(.003,nsteps,alpha=.1),weight_decay=1e-4))
    state=opt.init(model)
    @jax.jit
    def step(m,state,x,y,w):
        loss,g=jax.value_and_grad(lambda mm:jnp.sum((mm(x)-y)**2*w)/jnp.sum(w))(m)
        updates,state=opt.update(g,state,m)
        return optax.apply_updates(m,updates),state,loss
    rng=np.random.default_rng(seed)
    best=float('inf');best_model=model;best_epoch=0;trace=[];begin=time.monotonic()
    for epoch in range(1,epochs+1):
        order=rng.permutation(n_examples)
        seen=np.zeros(n_examples,dtype=bool)
        losses=[]
        for start in range(0,n_examples,batch):
            chunk=order[start:start+batch];n=len(chunk);seen[chunk]=True
            chunk=np.resize(chunk,batch)
            qi=chunk//len(tasks);target=tasks[chunk%len(tasks)]
            x,y=prompts(train[qi],train,target,rng,mode,qi)
            w=np.arange(batch)<n
            model,state,loss=step(model,state,jnp.asarray(x),jnp.asarray(y),jnp.asarray(w,dtype=jnp.float32))
            losses.append(float(loss))
        assert seen.all(), 'An eligible training row-target pair was skipped'
        row=dict(epoch=epoch,training_mse=float(np.mean(losses)),training_pairs=n_examples,elapsed_seconds=time.monotonic()-begin)
        if val is not None:
            pp,_=evaluate(model,val,train,mode,1800)
            mse=float(np.mean((pp[:,tasks]-val[:,tasks])**2));row['validation_mse']=mse
            if mse<best:best=mse;best_model=model;best_epoch=epoch
        else:best_model=model;best_epoch=epoch
        trace.append(row)
        print(json.dumps(dict(run=out.name,mode=mode,seed=seed,**row)),flush=True)
    out.mkdir(parents=True,exist_ok=True)
    model=best_model
    np.savez_compressed(out/'weights.npz',W=np.asarray(model.W),A0=np.asarray(model.A[0]),A1=np.asarray(model.A[1]))
    info=dict(seed=seed,mode=mode,epochs=epochs,best_epoch=best_epoch,train_rows=len(train),
              examples_per_epoch=n_examples,all_training_pairs_seen_each_epoch=True,trace=trace)
    (out/'training.json').write_text(json.dumps(info,indent=2))
    return model,info

def linear(train,val,test,mode):
    tasks=range(3,7) if mode=='rgb' else range(7)
    preds=np.full_like(test,np.nan)
    for target in tasks:
        feat=list(range(3)) if mode=='rgb' else [i for i in range(7) if i!=target]
        models=[Ridge(alpha=a).fit(train[:,feat],train[:,target]) for a in [0.,.01,1.,10.]]
        errors=[np.mean((m.predict(val[:,feat])-val[:,target])**2) for m in models]
        preds[:,target]=models[int(np.argmin(errors))].predict(test[:,feat])
    return preds

def metrics(y,p,tasks):
    return {COLS[t]:dict(r2=float(r2_score(y[:,t],p[:,t])),rmse=float(np.sqrt(np.mean((y[:,t]-p[:,t])**2)))) for t in tasks}

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--epochs',type=int,default=5)
    parser.add_argument('--seeds',type=int,nargs='+',default=[0,1,2])
    args=parser.parse_args()
    out=ROOT/'full_data_results';out.mkdir(exist_ok=True)
    df=pd.read_csv(ROOT/'data/color_mix.csv');raw=df[COLS].to_numpy(dtype=np.float32)
    assert len(raw)==10000 and np.isfinite(raw).all() and set(df.config)=={'standard'}
    blocks=np.arange(len(raw))//2000
    truth=json.loads((ROOT/'data/ground_truth.json').read_text())
    assert set(map(tuple,truth['edges']))=={(COLS[i],COLS[j]) for i,j in zip(*np.where(TRUTH))}
    summary=dict(dataset='lt_walks_v1/color_mix',n_rows=len(raw),columns=COLS,seeds=args.seeds,epochs=args.epochs,
                 protocol='5 contiguous test blocks; next block validation; remaining rows training, with 100-row training embargo around validation/test boundaries',
                 caveat='Blocked cross-validation uses other time blocks, including future blocks; it is not prospective forecasting. All test rows are scored exactly once per seed/mode.',
                 ground_truth=TRUTH.tolist(),runs=[],aggregate={},full_fits=[],devices=[str(x) for x in jax.devices()],
                 data_sha256=hashlib.sha256((ROOT/'data/color_mix.csv').read_bytes()).hexdigest())
    for mode in ['rgb','all']:
        tasks=np.arange(3,7) if mode=='rgb' else np.arange(7)
        for seed in args.seeds:
            oof=np.full_like(raw,np.nan);lin=np.full_like(raw,np.nan);coverage=np.zeros(len(raw),int);a_sum=np.zeros((3,7,7))
            for fold in range(5):
                ti=np.flatnonzero(blocks==fold);vi=np.flatnonzero(blocks==(fold+1)%5)
                held=np.isin(blocks,[fold,(fold+1)%5]);expanded=held.copy()
                for shift in range(1,101):
                    expanded[shift:] |= held[:-shift]
                    expanded[:-shift] |= held[shift:]
                tr=np.flatnonzero(~expanded)
                assert not np.intersect1d(tr,ti).size and not np.intersect1d(tr,vi).size
                mu=raw[tr].mean(0);sd=raw[tr].std(0);assert (sd>0).all()
                z=(raw-mu)/sd
                runout=out/f'{mode}_seed{seed}_fold{fold}'
                m,info=fit(z[tr],z[vi],mode,seed,args.epochs,64,runout)
                p,a=evaluate(m,z[ti],z[tr],mode,2800+fold)
                lp=linear(z[tr],z[vi],z[ti],mode)
                oof[ti]=p*sd+mu;lin[ti]=lp*sd+mu;coverage[ti]+=1;a_sum+=a/5
                np.savez_compressed(runout/'fold.npz',train_indices=tr,validation_indices=vi,test_indices=ti,
                                    mean=mu,std=sd,predictions=oof[ti],linear_predictions=lin[ti],attention=a)
                info.update(fold=fold,validation_rows=len(vi),test_rows=len(ti),metrics=metrics(raw[ti],oof[ti],tasks))
                summary['runs'].append(info)
                (out/'summary.json').write_text(json.dumps(summary,indent=2))
            assert np.all(coverage==1) and np.isfinite(oof[:,tasks]).all()
            np.savez_compressed(out/f'{mode}_seed{seed}_oof.npz',predictions=oof,linear_predictions=lin,actual=raw,coverage=coverage,attention=a_sum)
            summary['aggregate'][f'{mode}_seed{seed}']=dict(metrics=metrics(raw,oof,tasks),linear_metrics=metrics(raw,lin,tasks),
                scored_rows=10000,scored_row_target_pairs=10000*len(tasks),first_layer_query_attention=a_sum.tolist(),
                **({'attention_ranking':ranking(a_sum.mean(0))} if mode=='all' else {}))
            (out/'summary.json').write_text(json.dumps(summary,indent=2))
    # Separate final fit: every row-target pair trains the visualization model.
    # No in-sample metric from this model is reported as held-out performance.
    mu=raw.mean(0);sd=raw.std(0);z=(raw-mu)/sd
    for mode in ['rgb','all']:
        runout=out/f'{mode}_full_fit_seed0'
        m,info=fit(z,None,mode,0,args.epochs,64,runout)
        info.update(normalization=dict(mean=mu.tolist(),std=sd.tolist()),purpose='Full-data weight visualization, not held-out evaluation')
        summary['full_fits'].append(info)
        (runout/'normalization.json').write_text(json.dumps(info,indent=2))
    summary['all_rows_chi_square_descriptive']=binned_chi_square(raw).tolist()
    summary['complete']=True
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    print('COMPLETE: every row tested once per seed/mode; final full-data models trained exhaustively',flush=True)

if __name__=='__main__':main()
