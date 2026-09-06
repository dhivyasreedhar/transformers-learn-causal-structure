"""Checks for target leakage, causal masking, and the reused attention implementation."""
import numpy as np
import jax
import jax.numpy as jnp
from run import episodes, ContinuousCatFormer, binned_chi_square
from vendor.catformer import CatFormer

def main():
    rng = np.random.default_rng(4)
    query = rng.normal(size=(7, 7)).astype('float32')
    context = rng.normal(size=(20, 7)).astype('float32')
    targets = np.arange(7)
    x, y = episodes(query, context, targets, np.random.default_rng(55))
    changed = query.copy()
    changed[np.arange(7), targets] += 10000
    x2, y2 = episodes(changed, context, targets, np.random.default_rng(55))
    assert np.array_equal(x, x2), 'Target leaked into tokens'
    assert not np.array_equal(y, y2)
    rx, _ = episodes(query[3:], context, targets[3:], np.random.default_rng(1), mode='rgb')
    q = rx[:, -7:]
    assert np.all(q[..., 2][q[..., 0] >= 3] == 0)
    model = ContinuousCatFormer(21, jax.random.PRNGKey(0))
    assert ContinuousCatFormer.attn is CatFormer.attn
    assert model(x).shape == (7,)
    embedded = model.embed(jnp.asarray(x))
    a = model.A[0][0]
    first = model.attn(embedded, a)
    altered = embedded.at[:, -1, :].set(100)
    second = model.attn(altered, a)
    np.testing.assert_allclose(first[:, :-1], second[:, :-1], atol=1e-6)
    assert np.isfinite(np.asarray(jax.grad(lambda m: jnp.mean((m(x)-y)**2))(model).W)).all()
    binary = np.tile([[0, 0], [1, 1]], (100, 1))
    np.testing.assert_allclose(binned_chi_square(binary), np.ones((2, 2)))
    print('PASS: target hidden, non-RGB values hidden in RGB mode, original attention reused, causal mask, gradients')

if __name__ == '__main__': main()
