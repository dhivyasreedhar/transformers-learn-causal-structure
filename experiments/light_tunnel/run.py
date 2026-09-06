"""Continuous-value adaptation of the supplied CatFormer; no causal labels used in training.

Run from any directory: python run.py --data /path/to/color_mix.csv --out results
The original attention implementation is imported unchanged from vendor/catformer.py.
"""
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/light-tunnel-mpl')
import argparse, hashlib, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
import optax
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, average_precision_score, roc_auc_score
from vendor.catformer import CatFormer

COLS = ['red', 'green', 'blue', 'ir_1', 'ir_2', 'ir_3', 'current']
TRUTH = np.zeros((7, 7), dtype=int)  # source, target
TRUTH[:3, 3:] = 1

class ContinuousCatFormer(CatFormer):
    def __init__(self, length, key):
        # id, variable-specific standardized scalar, observed flag, constant, position
        self.vocab_size = 7
        d = 16 + length
        k1, k2 = jax.random.split(key)
        self.A = [1e-3 * jax.random.normal(k1, (3, d, d)),
                  1e-3 * jax.random.normal(k2, (1, 4*d, 4*d))]
        self.W = jnp.zeros((8*d, 1))

    def embed(self, tokens):
        ids, values, observed = tokens[..., 0].astype(int), tokens[..., 1], tokens[..., 2]
        identity = jax.nn.one_hot(ids, 7)
        pos = jnp.broadcast_to(jnp.eye(tokens.shape[-2]), (*tokens.shape[:-1], tokens.shape[-2]))
        return jnp.concatenate([identity, identity * (values*observed)[..., None],
                                observed[..., None], jnp.ones_like(observed[..., None]), pos], -1)

    def __call__(self, tokens):
        x = self.embed(tokens)
        for a in self.A:
            attended = jax.vmap(self.attn, (None, 0), -2)(x, a)
            x = jnp.concatenate([x, attended.reshape(*attended.shape[:-2], -1)], -1)
        return (x[..., -1, :] @ self.W)[..., 0]

def episodes(query, context, targets, rng, support=2, mode='all'):
    """Complete support rows from TRAIN only; randomized variable order; masked query last.

    No timestamp, intervention indicator, graph, or held-out support labels are supplied.
    """
    b = len(query)
    rows = context[rng.integers(len(context), size=(b, support))]
    order = np.argsort(rng.random((b, support, 7)), axis=-1)
    support_tokens = np.stack([order, np.take_along_axis(rows, order, -1), np.ones_like(rows)], -1)
    scores = rng.random((b, 7))
    scores[np.arange(b), targets] = 2
    order = np.argsort(scores, axis=-1)
    values = np.take_along_axis(query, order, -1)
    observed = (order != targets[:, None])
    if mode == 'rgb':
        observed &= order < 3
    query_tokens = np.stack([order, values*observed, observed], -1)
    tokens = np.concatenate([support_tokens.reshape(b, 7*support, 3), query_tokens], axis=1)
    labels = query[np.arange(b), targets]
    assert np.all(tokens[:, -1, 2] == 0) and np.all(tokens[:, -1, 1] == 0)
    return tokens.astype('float32'), labels.astype('float32')

predict = jax.jit(lambda model, x: model(x))

def batch_predict(model, tokens):
    out = []
    for start in range(0, len(tokens), 128):
        x = tokens[start:start+128]
        n = len(x)
        x = np.pad(x, ((0, 128-n), (0, 0), (0, 0)))
        out.append(np.asarray(predict(model, jnp.asarray(x)))[:n])
    return np.concatenate(out)

def make_eval(rows, train, mode, seed, support, count=256):
    rng = np.random.default_rng(seed)
    idx = np.linspace(0, len(rows)-1, min(count, len(rows)), dtype=int)
    tasks = np.arange(3, 7) if mode == 'rgb' else np.arange(7)
    targets = np.repeat(tasks, len(idx))
    query = np.tile(rows[idx], (len(tasks), 1))
    tokens, labels = episodes(query, train, targets, rng, support, mode)
    return tokens, labels, targets, query

def ranking(scores):
    mask = ~np.eye(7, dtype=bool)
    y, s = TRUTH[mask], np.asarray(scores)[mask]
    top = np.argsort(-s, kind='stable')[:12]
    return dict(average_precision=float(average_precision_score(y, s)),
                auroc=float(roc_auc_score(y, s)), precision_at_12=float(y[top].mean()),
                positive_fraction=float(y.mean()))

def scores_by_target(labels, pred, targets):
    return {COLS[t]: dict(r2=float(r2_score(labels[targets == t], pred[targets == t])),
                          standardized_mse=float(np.mean((labels[targets == t]-pred[targets == t])**2)))
            for t in np.unique(targets)}

def train_one(train, val, test, mode, seed, args, out):
    start = time.monotonic()
    rng = np.random.default_rng(seed)
    model = ContinuousCatFormer(7*(args.support+1), jax.random.PRNGKey(seed))
    optimizer = optax.chain(optax.clip_by_global_norm(1.),
                           optax.adamw(optax.cosine_decay_schedule(0.003, args.steps, alpha=0.1), weight_decay=1e-4))
    state = optimizer.init(model)
    @jax.jit
    def step(model, state, x, y):
        loss, grad = jax.value_and_grad(lambda m: jnp.mean((m(x)-y)**2))(model)
        updates, state = optimizer.update(grad, state, model)
        return optax.apply_updates(model, updates), state, loss
    vx, vy, vt, _ = make_eval(val, train, mode, 991, args.support, count=128)
    tx, ty, tt, tq = make_eval(test, train, mode, 992, args.support, count=256)
    trace, best, best_step, best_model = [], float('inf'), 0, model
    tasks = np.arange(3, 7) if mode == 'rgb' else np.arange(7)
    for i in range(args.steps+1):
        if i % args.eval_every == 0 or i == args.steps:
            vp = batch_predict(model, vx)
            vl = float(np.mean((vp-vy)**2))
            if not np.isfinite(vl): raise RuntimeError('Nonfinite validation loss')
            trace.append(dict(step=i, validation_mse=vl, elapsed_seconds=time.monotonic()-start))
            if vl < best:
                best, best_step, best_model = vl, i, model
            print(json.dumps(dict(mode=mode, seed=seed, **trace[-1])), flush=True)
        if i == args.steps: break
        q = train[rng.integers(len(train), size=args.batch)]
        targets = rng.choice(tasks, size=args.batch)
        x, y = episodes(q, train, targets, rng, args.support, mode)
        model, state, loss = step(model, state, jnp.asarray(x), jnp.asarray(y))
    model = best_model
    pred = batch_predict(model, tx)
    result = dict(mode=mode, seed=seed, best_step=best_step, validation_mse=best,
                  elapsed_seconds=time.monotonic()-start, metrics=scores_by_target(ty, pred, tt), trace=trace)
    # Context ablation asks whether support rows are used; it is not an ICL guarantee.
    no_context = tx.copy()
    no_context[:, :7*args.support, 1:] = 0
    result['no_context_metrics'] = scores_by_target(ty, batch_predict(model, no_context), tt)
    # Shuffle QUERY source values between held-out rows, leaving support unchanged.
    # This estimates predictive reliance, not a causal intervention or retrained ablation.
    delta = np.zeros((7, 7))
    permutation_rng = np.random.default_rng(407)
    for target in tasks:
        ix = np.flatnonzero(tt == target)
        base_mse = np.mean((pred[ix]-ty[ix])**2)
        for source in range(7):
            if source == target or (mode == 'rgb' and source >= 3): continue
            losses = []
            for repeat in range(3):
                perturbed = tx[ix].copy()
                qtokens = perturbed[:, -7:]
                rows, cols = np.where(qtokens[..., 0] == source)
                values = qtokens[rows, cols, 1].copy()
                qtokens[rows, cols, 1] = permutation_rng.permutation(values)
                losses.append(np.mean((batch_predict(model, perturbed)-ty[ix])**2)-base_mse)
            delta[source, target] = np.mean(losses)
    result['query_permutation_delta_mse'] = delta.tolist()
    if mode == 'all': result['edge_ranking'] = ranking(delta)
    # First-layer attention at the final masked query token; map positions back to variable IDs.
    def first_attention(m, tokens):
        x = m.embed(tokens)
        logits = jnp.einsum('btd,hde,bse->bhts', x, m.A[0], x)
        return jax.nn.softmax(logits[:, :, -1, :], axis=-1)
    attn_fn = jax.jit(first_attention)
    attn = np.concatenate([np.asarray(attn_fn(model, jnp.asarray(tx[i:i+128]))) for i in range(0, len(tx), 128)])
    a = np.zeros((7, 7))
    for t in tasks:
        mask = tt == t
        for s in range(7):
            a[s, t] = np.mean(np.sum(attn[mask, :, -7:] * (tx[mask, None, -7:, 0] == s), axis=-1))
    result['first_layer_query_attention'] = a.tolist()
    if mode == 'all': result['attention_edge_ranking'] = ranking(a)
    np.savez_compressed(out / f'{mode}_seed{seed}_weights.npz', W=np.asarray(model.W),
                        A0=np.asarray(model.A[0]), A1=np.asarray(model.A[1]))
    np.savez_compressed(out / f'{mode}_seed{seed}_predictions.npz', labels=ty, predictions=pred, targets=tt)
    (out/f'{mode}_seed{seed}.json').write_text(json.dumps(result, indent=2))
    return result

def baseline(train, val, test, mode):
    test = test[np.linspace(0, len(test)-1, min(256, len(test)), dtype=int)]
    result, coef = {}, np.zeros((7, 7))
    for target in (range(3, 7) if mode == 'rgb' else range(7)):
        features = list(range(3)) if mode == 'rgb' else [j for j in range(7) if j != target]
        models = [Ridge(alpha=a).fit(train[:, features], train[:, target]) for a in [0., 0.01, 1., 10.]]
        errors = [np.mean((m.predict(val[:, features])-val[:, target])**2) for m in models]
        model = models[int(np.argmin(errors))]
        pred = model.predict(test[:, features])
        result[COLS[target]] = dict(r2=float(r2_score(test[:, target], pred)),
                                    standardized_mse=float(np.mean((test[:, target]-pred)**2)), alpha=float(model.alpha))
        coef[features, target] = np.abs(model.coef_)
    return dict(metrics=result, absolute_coefficients=coef.tolist(),
                **({'edge_ranking': ranking(coef)} if mode == 'all' else {}))

def binned_chi_square(train):
    """Empirical pairwise divergence; preserve low-cardinality inputs such as binary green."""
    bins = []
    for j in range(train.shape[1]):
        unique = np.unique(train[:, j])
        bins.append((unique[:-1]+unique[1:])/2 if len(unique) <= 8 else
                    np.unique(np.quantile(train[:, j], np.linspace(0, 1, 9)))[1:-1])
    discrete = [np.digitize(train[:, j], bins[j]) for j in range(train.shape[1])]
    chi = np.zeros((train.shape[1], train.shape[1]))
    for i in range(train.shape[1]):
        for j in range(train.shape[1]):
            counts = np.zeros((len(bins[i])+1, len(bins[j])+1))
            np.add.at(counts, (discrete[i], discrete[j]), 1)
            joint = counts/counts.sum()
            independent = joint.sum(1)[:, None]*joint.sum(0)[None, :]
            valid = independent > 0
            chi[i, j] = np.sum((joint[valid]-independent[valid])**2/independent[valid])
    return chi

def load_checkpoint(path, support=2):
    weights = np.load(path)
    model = ContinuousCatFormer(7*(support+1), jax.random.PRNGKey(0))
    return model.replace(W=jnp.asarray(weights['W']),
                         A=[jnp.asarray(weights['A0']), jnp.asarray(weights['A1'])])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--out', type=Path, default=Path('results'))
    p.add_argument('--steps', type=int, default=1500)
    p.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    p.add_argument('--modes', nargs='+', default=['rgb', 'all'])
    p.add_argument('--support', type=int, default=2)
    p.add_argument('--batch', type=int, default=64)
    p.add_argument('--eval-every', type=int, default=250)
    p.add_argument('--synthetic', action='store_true')
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.data)
    assert set(df.config) == {'standard'}
    raw = df[COLS].to_numpy(dtype='float32')
    assert np.isfinite(raw).all() and len(raw) == 10000
    if args.synthetic:
        rng = np.random.default_rng(12345)
        rgb = rng.uniform(-1, 1, (10000, 3))
        weights = np.array([[1., .4, .8, .7], [.5, 1., .3, .9], [.3, .6, 1., .6]])
        raw = np.c_[rgb, rgb@weights + rng.normal(0, .03, (10000, 4))].astype('float32')
    # Chronological 60/20/20 with 100-row embargo on each side of both boundaries.
    indices = dict(train=np.arange(0, 5900), validation=np.arange(6100, 7900), test=np.arange(8100, 10000))
    mu, sd = raw[indices['train']].mean(0), raw[indices['train']].std(0)
    assert (sd > 0).all()
    z = (raw-mu)/sd
    train, val, test = (z[indices[k]] for k in ['train', 'validation', 'test'])
    chi = binned_chi_square(train)
    summary = dict(dataset='synthetic_linear_control' if args.synthetic else 'lt_walks_v1/color_mix',
                   columns=COLS, data_sha256=hashlib.sha256(args.data.read_bytes()).hexdigest(),
                   source_repo_commit='7d5e4569f77485e7bbeb7ee7834fb56453439aa4',
                   devices=[str(d) for d in jax.devices()], jax_version=jax.__version__,
                   config={k:str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
                   split={k:dict(start=int(v[0]), end_inclusive=int(v[-1]), n=len(v)) for k,v in indices.items()},
                   normalization=dict(mean=mu.tolist(), std=sd.tolist()),
                   ground_truth=TRUTH.tolist(), pearson_correlation=np.corrcoef(train.T).tolist(),
                   binned_chi_square=chi.tolist(), linear={}, runs=[])
    pd.DataFrame(raw, columns=COLS).describe().to_csv(args.out/'data_summary.csv')
    print(json.dumps(dict(devices=summary['devices'], split=summary['split'])), flush=True)
    for mode in args.modes:
        summary['linear'][mode] = baseline(train, val, test, mode)
        print('LINEAR', mode, json.dumps(summary['linear'][mode]['metrics']), flush=True)
        for seed in args.seeds:
            summary['runs'].append(train_one(train, val, test, mode, seed, args, args.out))
            (args.out/'summary.json').write_text(json.dumps(summary, indent=2))
    print('COMPLETE', str(args.out/'summary.json'), flush=True)

if __name__ == '__main__': main()
