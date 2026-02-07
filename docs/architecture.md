# Architecture

This repo implements a small GPT-style causal language model where self-attention can be:

1) **Dense causal attention** (baseline)
2) **Hamiltonian Cycle Sparse Attention (HCSA)**

## GPT skeleton

- Token embedding + learned positional embedding
- N Transformer blocks:
  - LayerNorm
  - Attention (dense or HCSA)
  - Residual
  - LayerNorm
  - MLP (GeLU)
  - Residual
- Final LayerNorm + linear vocab head

## Hamiltonian Cycle Sparse Attention (HCSA)

We define a sparse attention neighborhood per token position `i` as the union of:

- **Cycle neighbors**: two adjacent nodes in a Hamiltonian cycle over positions
- **Local causal window**: `{max(0,i-w), ..., i-1}` (always included)
- **Landmarks** (optional): `{j | j % stride == 0 and j < i}`

The cycle is built from positions `{0..T-1}`. We treat cycle edges as **undirected**.

### Enforcing causality

Sparse neighbors may include future indices (because the cycle ignores causality). We enforce causality by masking: token `i` can only attend to neighbors `j <= i` (self-attention to `i` is allowed; it guarantees a non-empty neighborhood at `i=0`).

### Efficient gather/scatter

We build a padded index tensor `neigh_idx: [T, D]` (with `-1` padding), then:

- Gather K,V to `[B, T, D, dh]` with `torch.gather`
- Compute scores only for those gathered entries
- Mask padding and causal violations
- Softmax over the neighbor dimension only

This is implemented in `hcsa/attention_hcsa.py`.

## Cycle strategies

See `hcsa/cycles.py`.

### Routing similarity

A learned routing projection produces `r_i = W_r x_i` and similarity:

`s(i,j) = (r_i · r_j) / sqrt(d_r)`

### random_cycle

Random permutation per head (optionally multiple cycles to create a union graph). Complexity: O(T).

### greedy_cycle

Nearest-neighbor greedy TSP-like heuristic: start from a node, repeatedly pick the most similar unvisited node. Complexity: O(T^2).

### online_insertion_cycle

Maintain a cycle while adding a new node `u` by choosing the best edge `(a,b)` to split:

`delta = s(a,u) + s(u,b) - s(a,b)`

Complexity: O(T) per insertion.
