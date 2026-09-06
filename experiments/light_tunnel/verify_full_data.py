"""Audit complete coverage, split separation, scores and actual saved plot matrices."""
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import r2_score
ROOT=Path(__file__).resolve().parent
s=json.loads((ROOT/'full_data_results/summary.json').read_text())
assert s.get('complete') and len(s['runs'])==10*len(s['seeds']) and len(s['full_fits'])==2
report=[]
for mode in ['rgb','all']:
 tasks=range(3,7) if mode=='rgb' else range(7)
 for seed in s['seeds']:
  o=np.load(ROOT/f'full_data_results/{mode}_seed{seed}_oof.npz')
  assert np.all(o['coverage']==1)
  assert o['predictions'].shape==(10000,7)
  assert np.isfinite(o['predictions'][:,list(tasks)]).all()
  train_coverage=np.zeros(10000,int);test_coverage=np.zeros(10000,int)
  for fold in range(5):
   p=ROOT/f'full_data_results/{mode}_seed{seed}_fold{fold}'
   d=np.load(p/'fold.npz');tr=d['train_indices'];va=d['validation_indices'];te=d['test_indices']
   assert len(va)==len(te)==2000
   assert not np.intersect1d(tr,va).size and not np.intersect1d(tr,te).size and not np.intersect1d(va,te).size
   for i in range(1,101):
    assert not np.isin(tr-i,np.r_[va,te]).any()
    assert not np.isin(tr+i,np.r_[va,te]).any()
   train_coverage[tr]+=1;test_coverage[te]+=1
   np.testing.assert_allclose(o['predictions'][te],d['predictions'],equal_nan=True)
   info=json.loads((p/'training.json').read_text())
   assert info['all_training_pairs_seen_each_epoch']
   assert all(t['training_pairs']==len(tr)*len(list(tasks)) for t in info['trace'])
  assert np.all(test_coverage==1) and np.all(train_coverage>0)
  for target in tasks:
   np.testing.assert_allclose(r2_score(o['actual'][:,target],o['predictions'][:,target]),
                             s['aggregate'][f'{mode}_seed{seed}']['metrics'][s['columns'][target]]['r2'],rtol=1e-6)
  report.append(f'{mode}, seed {seed}: 10,000/10,000 test rows covered exactly once; every row trained in other folds; no split overlap; 100-row training embargo verified; scores reproduced.')
 for kind in ['weights.npz']:
  w=np.load(ROOT/f'full_data_results/{mode}_full_fit_seed0/{kind}')
  assert w['A0'].shape==(3,37,37) and w['A1'].shape==(1,148,148) and w['W'].shape==(296,1)
  assert all(np.isfinite(w[k]).all() for k in w.files)
 info=json.loads((ROOT/f'full_data_results/{mode}_full_fit_seed0/training.json').read_text())
 assert info['train_rows']==10000 and info['epochs']==s['epochs'] and info['all_training_pairs_seen_each_epoch']
 report.append(f'{mode} final model: all 10,000 rows used in every one of 5 complete training epochs; plot matrices finite and correctly shaped.')
(ROOT/'full_data_results/verification.txt').write_text('\n'.join(report)+'\n')
print('\n'.join(report))
