# PRAXIS: Fast Rashomon Sets for Sparse Decision Trees

## Installation

```bash
pip install tree-praxis
```

## What PRAXIS does

This code creates **Rashomon sets of decision trees**. The Rashomon set is the set of all almost-optimal models.

PRAXIS is designed to enumerate the Rashomon set for **sparse decision trees**. In other words, instead of returning a single optimal decision tree, it returns a set of decision trees whose objective values are all within a small factor of the best tree found. The objective is:

```text
misclassifications + lambda_reg * n_samples * number_of_leaves.
```

This is useful when there are many decision trees with nearly the same accuracy. Rather than pretending there is one uniquely best tree, PRAXIS lets you inspect, count, compare, and make predictions with all good trees.

See [`examples/example.ipynb`](https://github.com/zakk-h/PRAXIS/blob/praxis/examples/example.ipynb) for a complete walkthrough of using the code.

To learn more about the algorithmic ideas behind PRAXIS, please see our ICML 2026 paper. At a high level, PRAXIS uses a proxy algorithm to estimate the best achievable objective within each subproblem, and then refines these estimates as enumeration proceeds. When the proxy algorithm is exact, PRAXIS performs exact Rashomon set enumeration. When the proxy is approximate, PRAXIS can trade a small amount of empirical approximation quality for substantially faster runtime.

PRAXIS also includes tools for computing feature importance over the Rashomon set through the **Rashomon Importance Distribution (RID)** and to directly use the proxy algorithms to return one tree, if desired.

## Data requirements

PRAXIS expects the input matrix `X` to already be binary.

```python
X[i, j] in {0, 1}
```

If your data has continuous, ordinal, or categorical features, binarize it first. The package includes `ThresholdGuessBinarizer` for this purpose.

```python
from praxis import ThresholdGuessBinarizer

binarizer = ThresholdGuessBinarizer()
X_binary = binarizer.fit_transform(X, y)
```

The label vector `y` must contain integer class labels numbered consecutively:

```python
0, 1, ..., num_classes - 1
```

For binary classification, this means labels must be:

```python
0, 1
```

## Basic example

```python
from praxis import PRAXIS

model = PRAXIS()

model.fit(
    X_binary,
    y,
    lambda_reg=0.01,
    depth_budget=5,
    rashomon_mult=0.03,
    lookahead_k=1,
)

print("Number of trees:", model.count_trees())
print("Minimum objective:", model.get_min_objective())
```

See [`examples/example.ipynb`](https://github.com/zakk-h/PRAXIS/blob/praxis/examples/example.ipynb) for a complete walkthrough on binarization, fitting PRAXIS, and accessing information about the Rashomon set. We provide a comprehensive list of options and included methods below.

## `PRAXIS.fit(...)`

```python
model.fit(
    X,
    y,
    lambda_reg=0.01,
    depth_budget=5,
    rashomon_mult=0.01,
    multiplicative_slack=0.0,
    key_mode="hash",
    lookahead_k=1,
    proxy_style=0,
    root_budget=None,
    use_budget_refinement=True,
    guarantee_rule_list_recovery=False,
    majority_leaf_only=False,
    cache_early_exits=False,
    heuristic_for_greedy=1,
    proxy_caching=True,
    num_proxy_features=0,
    proxy_only=False,
)
```

### Parameters

- **`X`**  
  Binary feature matrix of shape `(n_samples, n_features)`. All entries must be `0` or `1`. If your dataset is not binary, use `ThresholdGuessBinarizer` first.

- **`y`**  
  Class-label vector of shape `(n_samples,)`. Labels must be integers numbered `0, 1, ..., num_classes - 1`.

- **`lambda_reg=0.01`**  
  Regularization strength for tree complexity. This controls the penalty on leaves in the objective. Larger values favor smaller trees.

- **`depth_budget=5`**  
  Maximum depth of the decision trees.

- **`rashomon_mult=0.01`**  
  Multiplicative slack for the Rashomon set. A tree is included if its objective is within this factor of the proxy algorithm. For example, `rashomon_mult=0.03` means to search for trees within 3% of the proxy's objective. After the algorithm returns, one may want to only consider trees within 3% of the minimum objective tree, which can be done easily as the trees are returned in sorted order of best-to-worst objective.

- **`multiplicative_slack=0.0`**  
  An extra amount of slack applied to the objective bound. Most users can leave this at `0.0`.

- **`key_mode="hash"`**  
  Cache-key representation used internally.

  Options:
  - `"hash"`: use a 64-bit hash of the subproblem bitvector. Fast and memory efficient.
  - `"exact"` or `"bitvector"`: use exact bitvector keys. More memory, no hash collisions.
  - `"literal"`, `"lits"`, `"lits_exact"`, or `"itemset"`: slower, but avoids hash collisions without needing more memory.

  Most users should use `"hash"`.

- **`lookahead_k=1`**  
  Lookahead used by the proxy algorithm. Larger values usually make the proxy stronger but slower. We recommend `lookahead_k = 1` for most users. If `lookahead_k = 0`, PRAXIS uses the a greedy tree algorithm as the proxy. If `lookahead_k = 1`, PRAXIS uses a modified version of the LicketySPLIT algorithm. `lookahead_k = depth_budget-1`, then the proxy is optimal and the Rashomon set returned will be exact. `lookahead_k` should always be set in `0,1,2,..depth_budget-1`.

- **`proxy_style=0`**  
  Which proxy/oracle style to use.

  Nearly all users should use the default `0`.

- **`root_budget=None`**  
  Optional manual objective bound. If `None`, PRAXIS computes a reference objective and sets the Rashomon bound automatically using `rashomon_mult`. If you pass an integer, PRAXIS uses that objective bound directly. The loss optimized is number of misclassifications + lambda_reg * n_samples * number_of_leaves.

- **`use_budget_refinement=True`**  
  Enables the iterative budget refinement procedure. This is usually helpful and should nearly always stay `True`.

- **`guarantee_rule_list_recovery=False`**  
  Enables a special rule-list recovery mode. Most users should leave this `False`.

- **`majority_leaf_only=False`**  
  If `True`, only keeps the majority-class leaf prediction when constructing trees. If `False`, PRAXIS may keep multiple valid leaf predictions when they fit within the objective budget.

- **`cache_early_exits=False`**  
  Caches cheap subproblems and early exits. This can speed up some runs at the expense of more memory consumption.

- **`heuristic_for_greedy=1`**  
  Split heuristic used by the greedy routine.

  Options:
  - `0`, `"entropy"`, `"info_gain"`, `"information_gain"`, `"ig"`: entropy / information-gain style splitting.
  - `1`, `"entropy_depth1_exact"`, `"depth1_exact"`, `"default"`: entropy-style splitting with a depth-1 exact evaluation. This is the default.
  - `2`, `"best_split_for_leaves"`, `"misclassification_minimizing"`: choose splits based on minimizing child leaf objectives / misclassification.

- **`proxy_caching=True`**  
  Enables caching for proxy subproblems. This should almost always be left on as it will speed up PRAXIS by orders of magnitude.

- **`num_proxy_features=0`**  
  Restricts the proxy algorithm to the first `num_proxy_features` features. If `0` or negative, all features are used. If the first `num_proxy_features` features are chosen well (such as via a feature selection procedure), this can speed up runtime. We recommend not changing this for the vast majority of users.

- **`proxy_only=False`**  
  Controls whether PRAXIS builds the full Rashomon set or only returns the proxy tree.

  Options:
  - `False`: build the Rashomon set.
  - `True`: run only the proxy/single-tree algorithm and skip Rashomon enumeration. This tree is stored at index 0 of the data structures.

## Main methods

### `count_trees()`

```python
model.count_trees()
```

Returns the number of trees in the Rashomon set.

### `get_min_objective()`

```python
model.get_min_objective()
```

Returns the minimum objective value among the enumerated trees.

### `get_root_histogram()`

```python
model.get_root_histogram()
```

Returns a histogram of objective values at the root:

```python
[(objective_value, number_of_trees), ...]
```

This is useful for seeing how many trees exist at each objective value.

### `get_tree_objective(tree_index)`

```python
obj, obj_norm = model.get_tree_objective(tree_index)
```

Returns the unnormalized and normalized objective value of a specific tree.

- `obj`: integer objective value.
- `obj_norm`: objective divided by the number of samples.

### `count_trees_within_mult(mult)`

```python
model.count_trees_within_mult(0.03)
```

Counts how many enumerated trees have objective at most:

```text
round((1 + mult) * minimum_objective)
```

This is a post-hoc way to ask how many trees are within a different multiplicative slack of the best enumerated tree.

### `get_tree_paths(tree_index)`

```python
paths, predictions = model.get_tree_paths(tree_index)
```

Returns the selected tree as paths and leaf predictions.

The path representation uses signed, 1-indexed feature IDs:

- `+f` means go left / feature `f - 1` is true.
- `-f` means go right / feature `f - 1` is false.

The feature IDs are 1-indexed in this raw method, so subtract 1 to recover normal Python feature indices.

### `get_tree_paths_str(tree_index)`

```python
paths, predictions = model.get_tree_paths_str(tree_index)
```

Returns a more readable 0-indexed string representation of the paths.

Example output:

```python
["[+0, -3]", "[-0, +2]"]
```

This is usually easier to inspect than `get_tree_paths(...)`.

### `get_predictions(tree_index, X)`

```python
preds = model.get_predictions(tree_index, X_binary)
```

Returns predictions from one tree.

If the model was fit with `proxy_only=True`, only `tree_index=0` is supported.

### `get_all_predictions(X, stack=False)`

```python
preds = model.get_all_predictions(X_binary)
```

Returns predictions from all trees in the Rashomon set.

If `stack=False`, returns a list of prediction arrays.

If `stack=True`, returns a 2D array:

```python
(n_trees, n_samples)
```

### `plot_tree(tree_index, feature_names=None, figsize=(8, 6), ax=None, title=None, show=True)`

```python
fig, ax = model.plot_tree(0, feature_names=feature_names)
```

Plots one tree using matplotlib.

Parameters:
- `tree_index`: which tree to plot.
- `feature_names`: optional names for the binary features.
- `figsize`: matplotlib figure size.
- `ax`: optional existing matplotlib axis.
- `title`: optional plot title.
- `show`: whether to call `plt.show()`.

Returns:

```python
(fig, ax)
```

## Rashomon disagreement methods

These methods summarize disagreement across the Rashomon set. They assume binary predictions in `{0, 1}`.

### `get_p_per_sample(X, tree_indices=None)`

```python
p = model.get_p_per_sample(X_binary)
```

Returns one value per sample:

```text
proportion of selected trees predicting class 1
```

If `tree_indices=None`, all trees are used. Otherwise, only the specified tree indices are used.

### `get_variance_per_sample(X, tree_indices=None)`

```python
v = model.get_variance_per_sample(X_binary)
```

Returns the variance of hard predictions across trees for each sample.

For binary predictions, this is equivalent to:

```text
p_i * (1 - p_i)
```

where `p_i` is the proportion of trees predicting class `1` for sample `i`.

### `get_avg_variance_across_samples(X, tree_indices=None)`

```python
avg_v = model.get_avg_variance_across_samples(X_binary)
```

Returns the average disagreement variance across samples.

### `plot_disagreement_cdf(...)`

```python
fig, ax = model.plot_disagreement_cdf(X_binary)
```

Plots the empirical CDF of per-sample prediction variances.

Parameters:
- `X`: binary feature matrix.
- `tree_indices`: optional subset of trees.
- `ax`: optional matplotlib axis.
- `figsize`: figure size.
- `title`: plot title.
- `show`: whether to call `plt.show()`.
- `label`: optional plot label.

Returns:

```python
(fig, ax)
```

## Rashomon Importance Distribution methods

PRAXIS includes tools for estimating feature importance over the Rashomon set using subtractive model reliance.

### `compute_rid(...)`

```python
rid_out = model.compute_rid(
    X_binary,
    y,
    n_boot=10,
    lambda_reg=0.01,
    depth_budget=5,
    rashomon_mult=0.03,
    lookahead_k=1,
    seed=0,
    memory_efficient=False,
    binning_map=None,
)
```

Computes Rashomon Importance Distribution output.

Parameters:
- `X`: binary feature matrix.
- `y`: integer labels.
- `n_boot`: number of bootstrap samples.
- `lambda_reg`: tree complexity regularization.
- `depth_budget`: maximum tree depth.
- `rashomon_mult`: Rashomon slack.
- `lookahead_k`: proxy lookahead.
- `seed`: random seed.
- `memory_efficient`: whether to use the lower-memory implementation path.
- `binning_map`: optional map from original features to binarized columns. This is useful when multiple binary threshold features came from the same original variable.

Returns and stores the RID output dictionary.

### `rid_plot_mean(feature_names=None, **kwargs)`

```python
model.rid_plot_mean(feature_names=feature_names)
```

Plots the mean reliance score for each feature.

### `rid_plot_violin(feature_names=None, **kwargs)`

```python
model.rid_plot_violin(feature_names=feature_names)
```

Plots the distribution of reliance scores for each feature.

### `rid_plot_cdfs(feature_names=None, **kwargs)`

```python
model.rid_plot_cdfs(feature_names=feature_names)
```

Plots the CDF of reliance scores for each feature.

## Recommended starting configuration

For most users, start with:

```python
model = PRAXIS()

model.fit(
    X_binary,
    y,
    lambda_reg=0.01,
    depth_budget=5,
    rashomon_mult=0.03,
    lookahead_k=1,
    key_mode="hash",
)
```

If your original data is not binary, use `ThresholdGuessBinarizer` before fitting.

See [`examples/example.ipynb`](https://github.com/zakk-h/PRAXIS/blob/praxis/examples/example.ipynb) for a complete walkthrough on binarization, fitting PRAXIS, and accessing information about the Rashomon set.
