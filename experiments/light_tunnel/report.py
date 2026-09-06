"""Regenerate scientific plots, score CSVs and a readable report from saved runs."""
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/light-tunnel-mpl')
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parent
s = json.loads((ROOT/'results/summary.json').read_text())
sy = json.loads((ROOT/'synthetic_results/summary.json').read_text())
cols = s['columns']; labels = ['R','G','B','I1','I2','I3','Current']
rgb = [r for r in s['runs'] if r['mode']=='rgb']
all_runs = [r for r in s['runs'] if r['mode']=='all']
out = ROOT/'report'; out.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})

records=[]
for mode in ['rgb','all']:
 for c in (cols[3:] if mode=='rgb' else cols):
  runs=[r for r in s['runs'] if r['mode']==mode]
  vals=np.array([r['metrics'][c]['r2'] for r in runs])
  records.append(dict(setting=mode,target=c,transformer_r2_mean=vals.mean(),transformer_r2_seed_sd=vals.std(ddof=1),
                      ridge_r2=s['linear'][mode]['metrics'][c]['r2'],
                      transformer_no_context_r2_mean=np.mean([r['no_context_metrics'][c]['r2'] for r in runs])))
pred_table=pd.DataFrame(records); pred_table.to_csv(out/'prediction_metrics.csv',index=False)

fig,axes=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
for ax,mode,title in zip(axes,['rgb','all'],['RGB-only prediction control','Masked-variable prediction from other six']):
 d=pred_table[pred_table.setting==mode]; x=np.arange(len(d))
 ax.bar(x-.18,d.ridge_r2,.36,label='Validation-selected linear regression',color='#aab2bc')
 ax.bar(x+.18,d.transformer_r2_mean,.36,yerr=d.transformer_r2_seed_sd,capsize=3,label='CatFormer adaptation (3 seeds)',color='#245875')
 ax.set_xticks(x,[labels[cols.index(c)] for c in d.target]);ax.set_ylim(0,1.04)
 ax.set_title(title);ax.set_xlabel('Predicted variable');ax.set_ylabel('Held-out R² (unitless)')
 ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
axes[0].legend(loc='lower left',fontsize=8)
fig.suptitle('Light tunnel: good prediction does not establish causal recovery',fontsize=14)
fig.supxlabel('Source: lt_walks_v1/color_mix · 256 fixed query rows sampled across rows 8100–9999 · error bars: seed SD',fontsize=9)
fig.savefig(out/'prediction_comparison.png',dpi=180);plt.close(fig)

score=np.mean([r['query_permutation_delta_mse'] for r in all_runs],axis=0)
attn=np.mean([r['first_layer_query_attention'] for r in all_runs],axis=0)
truth=np.asarray(s['ground_truth'])
pd.DataFrame(score,index=cols,columns=cols).to_csv(out/'permutation_reliance.csv')
pd.DataFrame(attn,index=cols,columns=cols).to_csv(out/'first_layer_attention.csv')
pd.DataFrame(s['binned_chi_square'],index=cols,columns=cols).to_csv(out/'binned_chi_square_train.csv')

fig,axes=plt.subplots(1,3,figsize=(13.8,5),layout='constrained')
for ax,mat,title,bar in zip(axes,[truth,score,attn],['Ground truth: 12 directed edges','Query permutation reliance','First-layer query attention'],['Edge present (0/1)','Increase in standardized MSE','Attention probability']):
 mat=np.array(mat,dtype=float);np.fill_diagonal(mat,np.nan)
 image=ax.imshow(mat.T,cmap='Blues',vmin=0)
 ax.set_xticks(range(7),labels,rotation=45,ha='right');ax.set_yticks(range(7),labels)
 ax.set_xlabel('Source variable');ax.set_ylabel('Target variable');ax.set_title(title)
 fig.colorbar(image,ax=ax,shrink=.68,label=bar)
fig.suptitle('Predictive dependence scores do not reproduce the directed graph',fontsize=14)
fig.supxlabel('Source: color_mix · rows 8100–9999, 256 fixed queries per target · mean of 3 seeds; attention also averaged over 3 heads',fontsize=9)
fig.savefig(out/'dependency_comparison.png',dpi=180);plt.close(fig)

fig,axes=plt.subplots(1,2,figsize=(10,4.8),layout='constrained')
for ax,key,title in zip(axes,['pearson_correlation','binned_chi_square'],['Training Pearson correlation','Training binned χ² divergence']):
 a=np.asarray(s[key],dtype=float).copy();np.fill_diagonal(a,np.nan)
 im=ax.imshow(a,cmap='coolwarm' if key=='pearson_correlation' else 'Blues',vmin=-1 if key=='pearson_correlation' else 0,vmax=1 if key=='pearson_correlation' else None)
 ax.set_xticks(range(7),labels,rotation=45,ha='right');ax.set_yticks(range(7),labels)
 ax.set_xlabel('Variable');ax.set_ylabel('Variable');ax.set_title(title);fig.colorbar(im,ax=ax,label='Unitless')
fig.supxlabel('Source: color_mix training rows 0–5899 · χ² uses ≤8 quantile bins, midpoint bins for discrete variables; not conditional MI',fontsize=9)
fig.savefig(out/'signal_diagnostics.png',dpi=180);plt.close(fig)

fig,ax=plt.subplots(figsize=(8,4.5),layout='constrained')
for mode,color in [('rgb','#245875'),('all','#9a6631')]:
 for r in [r for r in s['runs'] if r['mode']==mode]:
  ax.plot([t['step'] for t in r['trace']],[t['validation_mse'] for t in r['trace']],color=color,alpha=.7,label=f'{mode}, seed {r["seed"]}')
ax.set_yscale('log');ax.set_xlabel('Optimizer steps');ax.set_ylabel('Validation standardized MSE (log scale)');ax.set_title('Validation loss during training')
ax.legend(fontsize=8,ncol=2);ax.grid(alpha=.2)
fig.supxlabel('Source: color_mix validation rows 6100–7899 · 128 fixed queries per target',fontsize=9)
fig.savefig(out/'learning_curves.png',dpi=180);plt.close(fig)

edges=[]
for r in all_runs:
 for kind in ['edge_ranking','attention_edge_ranking']:
  edges.append(dict(seed=r['seed'],score=kind,**r[kind]))
pd.DataFrame(edges).to_csv(out/'edge_metrics.csv',index=False)
# Also assess only the four output targets: 24 candidates, 12 positives.
output_metrics=[]
for r in all_runs:
 for field in ['query_permutation_delta_mse','first_layer_query_attention']:
  mask=~np.eye(7,dtype=bool);mask[:,:3]=False
  yy=truth[mask];ss=np.asarray(r[field])[mask]
  output_metrics.append(dict(seed=r['seed'],score=field,average_precision=float(average_precision_score(yy,ss)),
                             auroc=float(roc_auc_score(yy,ss)),precision_at_12=float(yy[np.argsort(-ss)[:12]].mean())))
pd.DataFrame(output_metrics).to_csv(out/'output_target_edge_metrics.csv',index=False)

means=pred_table[pred_table.setting=='rgb'].transformer_r2_mean.to_numpy()
lin=pred_table[pred_table.setting=='rgb'].ridge_r2.to_numpy()
ap=np.array([r['edge_ranking']['average_precision'] for r in all_runs])
aap=np.array([r['attention_edge_ranking']['average_precision'] for r in all_runs])
chi=np.asarray(s['binned_chi_square'])
text=f'''# Light tunnel experiment results

The continuous CatFormer adaptation learns the easy RGB-to-sensor prediction task. It does not recover the directed ground-truth graph using the tested attention or permutation scores.

## Prediction control

Mean held-out R² over three seeds (I1, I2, I3, current): **{', '.join(f'{x:.3f}' for x in means)}**. Linear-regression R² on exactly the same query rows: **{', '.join(f'{x:.3f}' for x in lin)}**. Thus prediction is broadly comparable to the linear baseline; I3 is slightly worse with this adapter.

The synthetic independent-input linear control reaches R² **{', '.join(f'{sy['runs'][0]['metrics'][c]['r2']:.3f}' for c in cols[3:])}** (one seed). This is evidence that the continuous adapter can learn a strong linear signal.

![Prediction comparison](prediction_comparison.png)

## Exploratory graph recovery

The all-variable task predicts each variable from the other six without a supplied causal ordering. There are 42 directed candidate edges, of which 12 are ground-truth positives. Query-permutation importance gives average precision **{ap.mean():.3f} ± {ap.std(ddof=1):.3f}** across three seeds. None of the top 12 ranked pairs is a true directed edge in any of these runs. First-layer query attention gives average precision **{aap.mean():.3f} ± {aap.std(ddof=1):.3f}**, with 1, 3, and 2 correct edges among the top 12 for seeds 0, 1, and 2. The positive fraction is 12/42 = 0.286; this is a prevalence reference, not a finite-sample significance test.

Scores favor sensor proxies and reverse predictive relationships. For example, I1↔I2 is prominent in permutation reliance. `output_target_edge_metrics.csv` additionally restricts evaluation to the four output targets (24 candidates), to separate reverse-direction errors from choosing sensor proxies.

These are predictive scoring heuristics, not a validated causal-discovery algorithm. Marginal permutations may create off-distribution inputs; attention weights need not measure feature importance. The reported error bars describe variation across training seeds on one fixed split, not uncertainty across independent experiments.

![Dependency comparison](dependency_comparison.png)

## The original mutual-information hunch

The hypothesized dominance of R–I1 over sensor–sensor dependence is not supported by the training diagnostic. The binned pairwise χ² divergence is **{chi[0,3]:.2f} for R–I1**, versus **{chi[3,4]:.2f} for I1–I2**. These empirical, bin-dependent values are not the conditional χ² mutual information in the transformer paper. They do explain why prediction can use sensors as proxies.

![Signal diagnostics](signal_diagnostics.png)

## What was actually run

- Dataset: `lt_walks_v1/color_mix`, 10,000 standard-configuration measurements. Selected variables: red, green, blue, ir_1, ir_2, ir_3, current.
- Split: train 0–5899, validation 6100–7899, test 8100–9999 (zero-based). Each boundary has a 200-row gap. Training-only normalization; no timestamp or intervention labels.
- Model: original `CatFormer.attn`, two concatenating attention layers, 3 then 1 heads; continuous values, variable identities and missing flags; scalar MSE output; AdamW, 1,500 steps, batch 64. Best checkpoint chosen on validation loss.
- Prompts: two complete training support rows plus one masked query row (21 tokens). Variable order randomized; hidden target last. RGB control hides all query sensor/current values; all-variable task observes the other six.
- Evaluation: the same 256 evenly spaced test rows per target for the neural and linear models. Three real-data seeds (0,1,2); synthetic control seed 0.
- Checks passed: query target never appears in its input, non-RGB values hidden in RGB mode, original attention method reused, causal masking preserved, gradients finite. Dataset ZIP checksum and ground-truth edges verified.
- All data and code are in `experiments/light_tunnel/` in the supplied repository. Source files outside this subfolder were not edited.

## Scope of the conclusion

This is an empirical continuous-data adaptation, **not a reproduction of either paper's original experiment or the transformer theorem**. The theorem assumes a different categorical task, a mechanism that varies across prompts, a restricted model, and staged training. Here the mechanism is fixed and the objective is supervised regression. Strong performance persists after support values are removed (see `prediction_metrics.csv`), so these results do not establish mechanism inference in context.

The supported conclusion is: **the adapted transformer is a functioning predictor on this linear physical dataset, while the tested dependency scores fail to identify its causal graph.** A claim that transformers are fundamentally bounded would require additional task-matched controls, identification assumptions, and datasets.

![Learning curves](learning_curves.png)

## Reproduce and inspect

See `../README.md` for commands, methodology, data license, and limitations. `../results/summary.json` and `../synthetic_results/summary.json` contain all run metrics, normalization, diagnostics and configuration. NPZ checkpoints and predictions sit beside those files. Run `python report.py` to regenerate this report and figures.

Data attribution: Juan L. Gamella, Jonas Peters, Peter Bühlmann, *Causal chambers as a real-world physical testbed for AI methodology* (2025), https://doi.org/10.1038/s42256-024-00964-x. Dataset: https://github.com/juangamella/causal-chamber/tree/main/datasets/lt_walks_v1 (CC BY 4.0; raw CSV unchanged).

Model source: Eshaan Nichani, Alex Damian, Jason D. Lee, *How Transformers Learn Causal Structure with Gradient Descent*, https://arxiv.org/abs/2402.14735. Supplied repository commit: `7d5e4569f77485e7bbeb7ee7834fb56453439aa4`.
'''
(out/'REPORT.md').write_text(text)
print(text[:1200])
