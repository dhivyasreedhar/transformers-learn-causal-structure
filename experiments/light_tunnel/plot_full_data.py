"""Plot actual signed model parameters, with block layout inspired by the supplied figure."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/light-tunnel-mpl')
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from run import COLS,TRUTH
ROOT=Path(__file__).resolve().parent
SRC=ROOT/'full_data_results';OUT=ROOT/'full_data_report';OUT.mkdir(exist_ok=True)
labels=['R','G','B',r'$\tilde I_1$',r'$\tilde I_2$',r'$\tilde I_3$',r'$\tilde C$']
plt.rcParams.update({'font.family':'DejaVu Serif','font.size':10,'axes.spines.top':False,'axes.spines.right':False})

def cuts(ax,nblocks,d=37,features=16):
    for i in range(nblocks):
        for cut in [i*d+features,(i+1)*d]:
            if cut<nblocks*d:
                ax.axhline(cut-.5,color='white',lw=1.2 if cut%d else 2)
                ax.axvline(cut-.5,color='white',lw=1.2 if cut%d else 2)

def weights(mode):
    w=np.load(SRC/f'{mode}_full_fit_seed0/weights.npz')
    a0,a1,wo=w['A0'],w['A1'][0],w['W'].T
    assert a0.shape==(3,37,37) and a1.shape==(148,148) and wo.shape==(1,296)
    fig=plt.figure(figsize=(15,9),layout='constrained')
    grid=fig.add_gridspec(2,3,height_ratios=[3.7,1.4],width_ratios=[1,1,1.65])
    ax=fig.add_subplot(grid[0,0]);im=ax.imshow(TRUTH.T,cmap='viridis',vmin=0,vmax=1)
    ax.set_title(r'Causal graph $\mathcal{G}$',fontsize=19,pad=18)
    ax.set_xticks(range(7),labels,rotation=45,ha='right');ax.set_yticks(range(7),labels)
    ax.set_xlabel('Source variable');ax.set_ylabel('Target variable')
    cb=fig.colorbar(im,ax=ax,orientation='horizontal',shrink=.8,pad=.04,ticks=[0,1]);cb.set_label('Ground-truth edge (0/1)')
    ax=fig.add_subplot(grid[0,1]);lim=float(np.max(np.abs(a0)))
    im=ax.imshow(a0[0],cmap='viridis',vmin=-lim,vmax=lim);cuts(ax,1)
    ax.set_title(r'Weights $\widetilde A^{(1)}_1$',fontsize=19,pad=18)
    ax.set_xticks([7.5,26],['Features\n16','Position\n21']);ax.xaxis.tick_top()
    ax.set_yticks([7.5,26],['Features 16','Position 21'],rotation=90,va='center')
    ax.add_patch(Rectangle((15.5,15.5),21,21,fill=False,ec='red',lw=2))
    ax.set_xlabel('Key features (columns)');ax.set_ylabel('Query features (rows)')
    cb=fig.colorbar(im,ax=ax,orientation='horizontal',shrink=.8,pad=.04);cb.set_label('Signed parameter value (shared scale for 3 heads)')
    ax=fig.add_subplot(grid[0,2]);lim=float(np.max(np.abs(a1)))
    im=ax.imshow(a1,cmap='viridis',vmin=-lim,vmax=lim);cuts(ax,4)
    ax.set_title(r'Weights $\widetilde A^{(2)}$',fontsize=19,pad=18)
    centers=[18+i*37 for i in range(4)]
    groups=['Input','Head 1','Head 2','Head 3']
    ax.set_xticks(centers,groups);ax.xaxis.tick_top();ax.set_yticks(centers,groups,rotation=90,va='center')
    for i in range(1,4):ax.add_patch(Rectangle((i*37-.5,-.5),16,16,fill=False,ec='red',lw=1.5))
    ax.set_xlabel('Key residual-stream blocks');ax.set_ylabel('Query residual-stream blocks')
    cb=fig.colorbar(im,ax=ax,orientation='horizontal',shrink=.8,pad=.04);cb.set_label('Signed parameter value')
    ax=fig.add_subplot(grid[1,:]);lim=float(np.max(np.abs(wo)))
    im=ax.imshow(wo,cmap='viridis',vmin=-lim,vmax=lim,aspect='auto',extent=(-.5,295.5,.5,-.5))
    ax.set_title(r'Output weights $\widetilde W_O$ — scalar regression readout',fontsize=19,pad=48)
    names=['Input','Layer 1: H1','Layer 1: H2','Layer 1: H3','Layer 2: Input','Layer 2: H1','Layer 2: H2','Layer 2: H3']
    for i,name in enumerate(names):
        base=37*i
        if i:ax.axvline(base-.5,color='white',lw=3)
        ax.axvline(base+15.5,color='white',lw=1)
        ax.text(base+18,1.18,name,transform=ax.get_xaxis_transform(),ha='center',fontsize=10)
        # Actual variable-value features, not a categorical identity output block.
        ax.add_patch(Rectangle((base+6.5,-.5),7,1,fill=False,ec='red',lw=1.3))
    ax.set_xticks([i*37+c for i in range(8) for c in [7.5,26]],['Features','Position']*8,fontsize=8)
    ax.xaxis.tick_top();ax.set_yticks([0],['Scalar output']);ax.set_xlabel('Residual-stream coordinates (296 total)')
    cb=fig.colorbar(im,ax=ax,orientation='horizontal',shrink=.5,pad=.12);cb.set_label('Signed parameter value')
    fig.suptitle(f'Full-data CatFormer weights · {mode.upper()} setting · 10,000 color_mix rows · seed 0',fontsize=20)
    fig.supxlabel('Actual unnormalized weights, not attention probabilities. Red outlines mark inspected blocks, not recovered edges.\nFeatures = variable identity (7) + value (7) + observed flag + constant. Ground truth is a 7-variable graph; prompts have 21 positions.',fontsize=10)
    fig.savefig(OUT/f'{mode}_weights_paper_style.png',dpi=190)
    fig.savefig(OUT/f'{mode}_weights_paper_style.svg')
    plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(12,4.5),layout='constrained');lim=float(np.max(np.abs(a0)))
    for h,ax in enumerate(axes):
        im=ax.imshow(a0[h],cmap='viridis',vmin=-lim,vmax=lim);cuts(ax,1)
        ax.add_patch(Rectangle((15.5,15.5),21,21,fill=False,ec='red',lw=2))
        ax.set_title(f'Layer 1, head {h+1}');ax.set_xticks([7.5,26],['Features','Position']);ax.set_yticks([7.5,26],['Features','Position'])
        ax.set_xlabel('Key features');ax.set_ylabel('Query features')
    fig.colorbar(im,ax=axes,shrink=.75,label='Signed parameter value')
    fig.suptitle(f'All first-layer heads · full-data {mode.upper()} fit · seed 0',fontsize=16)
    fig.supxlabel('Source: saved full-data checkpoint · all 10,000 rows trained for 5 exhaustive epochs · position blocks outlined',fontsize=9)
    fig.savefig(OUT/f'{mode}_first_layer_heads.png',dpi=180);plt.close(fig)

s=json.loads((SRC/'summary.json').read_text());assert s.get('complete')
for mode in ['rgb','all']:weights(mode)
rows=[]
for mode in ['rgb','all']:
 for t in (range(3,7) if mode=='rgb' else range(7)):
    runs=[s['aggregate'][f'{mode}_seed{seed}'] for seed in s['seeds']]
    vals=np.array([r['metrics'][COLS[t]]['r2'] for r in runs])
    rows.append(dict(mode=mode,target=COLS[t],r2_mean=vals.mean(),r2_seed_sd=vals.std(ddof=1),linear_r2=runs[0]['linear_metrics'][COLS[t]]['r2'],rows_scored_per_seed=10000))
df=pd.DataFrame(rows);df.to_csv(OUT/'all_rows_metrics.csv',index=False)
fig,axes=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
for ax,mode in zip(axes,['rgb','all']):
 d=df[df['mode']==mode];x=np.arange(len(d))
 ax.bar(x-.18,d.linear_r2,.36,color='#aab2bc',label='Linear regression')
 ax.bar(x+.18,d.r2_mean,.36,color='#245875',yerr=d.r2_seed_sd,capsize=3,label='CatFormer (3 seeds)')
 ax.set_xticks(x,[labels[COLS.index(v)] for v in d.target]);ax.set_xlabel('Target variable');ax.set_ylabel('Out-of-fold R² (unitless)')
 ax.set_title('RGB-only prediction' if mode=='rgb' else 'Masked-variable prediction')
 ax.set_ylim(min(0,d.r2_mean.min()-.1),1.04);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
axes[0].legend(loc='lower left')
fig.suptitle('Every one of the 10,000 rows evaluated out of fold',fontsize=16)
fig.supxlabel('Source: lt_walks_v1/color_mix · 5 blocked folds × 3 seeds · all rows, no evaluation subsampling · error bars: seed SD',fontsize=9)
fig.savefig(OUT/'full_data_prediction.png',dpi=180);plt.close(fig)

allattention=np.mean([s['aggregate'][f'all_seed{seed}']['first_layer_query_attention'] for seed in s['seeds']],axis=(0,1))
fig,axes=plt.subplots(1,2,figsize=(9,4.8),layout='constrained')
for ax,mat,title,bar in zip(axes,[TRUTH.T,allattention.T],['Ground truth','Mean first-layer query attention'],['Edge (0/1)','Probability']):
 im=ax.imshow(mat,cmap='viridis',vmin=0,vmax=1 if title=='Ground truth' else None)
 ax.set_xticks(range(7),labels,rotation=45,ha='right');ax.set_yticks(range(7),labels)
 ax.set_xlabel('Source variable');ax.set_ylabel('Target variable');ax.set_title(title);fig.colorbar(im,ax=ax,shrink=.75,label=bar)
fig.supxlabel('Source: all 10,000 out-of-fold queries per target · mean over 5 folds, 3 seeds and 3 heads · attention is not a causal estimate',fontsize=9)
fig.savefig(OUT/'full_data_graph_attention.png',dpi=180);plt.close(fig)

cv_rgb=df[df['mode']=='rgb'];means=', '.join(f'{v:.3f}' for v in cv_rgb.r2_mean)
linear=', '.join(f'{v:.3f}' for v in cv_rgb.linear_r2)
rank=[s['aggregate'][f'all_seed{seed}']['attention_ranking'] for seed in s['seeds']]
report=f'''# Full color_mix experiment: every row used

All **10,000 measurements** and all seven requested variables were used. This run replaces subsampled evaluation with complete five-fold blocked evaluation, and separately trains final visualization models on all 10,000 rows.

## Complete data coverage

- Five contiguous test blocks of 2,000 rows. Each row is predicted exactly once per seed in each setting, by a model that did not train on that row.
- The next block cyclically supplies validation data. All 2,000 validation rows are evaluated after every epoch; the best validation epoch selects the checkpoint.
- The remaining blocks train the model, with 100-row training exclusions adjacent to validation/test blocks. No row is permanently dropped: every row is tested and is eligible for training in other folds.
- Three seeds, two settings: **30 cross-validation models**. Each training epoch explicitly visits every eligible row-target pair once. There is no random-batch coverage uncertainty.
- Evaluation totals: **120,000 RGB-output predictions** (10,000 × 4 targets × 3 seeds), and **210,000 masked-variable predictions** (10,000 × 7 targets × 3 seeds).
- Two additional models, RGB-only and all-variable, are fitted using **all 10,000 rows for five exhaustive epochs**. Their actual weights produce the reference-style figures; they are not used to claim held-out performance.

## Results

Pooled out-of-fold R², averaged across three seeds, for I1, I2, I3, and current: **{means}**. Linear regression on the same out-of-fold queries: **{linear}**. Values are pooled from physical-unit predictions using the normalization of each training fold.

![Full-data prediction](full_data_prediction.png)

First-layer attention's directed-edge average precision in the all-variable task is **{np.mean([r['average_precision'] for r in rank]):.3f}**, averaged across seeds (12 positives among 42 candidate directed pairs). Attention remains a descriptive diagnostic, not an identified causal graph.

![Graph and attention](full_data_graph_attention.png)

## Weight plots matching the reference layout

The figures show the true 7-variable graph, raw first-layer head-1 weights, raw second-layer weights, and the output matrix. Companion figures show all three first-layer heads. Signed values are displayed on symmetric scales; no softmax or threshold is applied to the raw weights.

![RGB full-data weights](rgb_weights_paper_style.png)

![All-variable full-data weights](all_weights_paper_style.png)

These are the actual parameters of the **continuous-value adapter**, not invented copies of the categorical paper figure. The first layer has three 37×37 matrices; layer two is 148×148; the scalar output readout is 1×296. A base representation contains 16 data features and 21 positional coordinates. Because query variable order is randomized, the 7-variable graph is not directly a 21-position adjacency matrix. Red boxes only identify inspected blocks; they do not assert that those blocks recover a graph or an identity matrix.

The adapter retains the supplied `CatFormer.attn` implementation and concatenated attention outputs. Continuous value/identity embeddings, scalar MSE regression, random variable ordering, and a fixed physical mechanism differ from the paper's theoretical categorical task. Therefore this is an empirical adaptation, not an exact reproduction or a test of its theorem.

## Reproduce

From `experiments/light_tunnel`, using the environment described in the main README:

```sh
python full_data.py --epochs 5 --seeds 0 1 2
python plot_full_data.py
```

`full_data_results/summary.json` records all folds and metrics. Each fold saves train/validation/test row indices, normalization, predictions and model weights. The `*_oof.npz` files contain 10,000-row prediction arrays and explicit coverage counters. Final full-data weights are under `rgb_full_fit_seed0` and `all_full_fit_seed0`. The plotting script produces PNG and vector SVG reference-style figures.

## Interpretation limits

This is blocked cross-validation, not future-only forecasting: training may include time blocks later than the test block. Exclusion gaps reduce immediate temporal leakage but do not make the deterministic driving signals independent. Seed error bars quantify optimization variation on this one physical run. Support rows come exclusively from the training fold; a training query cannot be its own support row. No causal labels or intervention indicators are fed into training. No score from the final all-row fit is labeled held-out.

Data: *Causal chambers as a real-world physical testbed for AI methodology*, Gamella, Peters, Bühlmann (2025), https://doi.org/10.1038/s42256-024-00964-x. Dataset `lt_walks_v1/color_mix`, standard configuration, CC BY 4.0. Source: https://github.com/juangamella/causal-chamber/tree/main/datasets/lt_walks_v1. Selected raw columns: {', '.join(COLS)}.

Transformer: *How Transformers Learn Causal Structure with Gradient Descent*, https://arxiv.org/abs/2402.14735. Original supplied model file retained unmodified in `vendor/catformer.py`.
'''
(OUT/'REPORT.md').write_text(report)
print(report[:1800])
