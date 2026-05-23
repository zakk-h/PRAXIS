import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle
from matplotlib.lines import Line2D
from matplotlib.cm import get_cmap
from ._core import PRAXIS as _PRAXISCore, rid_subtractive_model_reliance as _rid_subtractive_core
from ._threshold_guessing import ThresholdGuessBinarizer

__all__ = ["PRAXIS", "ThresholdGuessBinarizer"]

def _normalize_key(s: str) -> str:
    # lower, trim, and make separators uniform
    s = str(s).strip().lower()
    s = s.replace("-", "_").replace(" ", "_")
    # collapse repeats
    while "__" in s:
        s = s.replace("__", "_")
    return s

_PROXY_STYLE_MAP = {
    # explicit canonical names
    "recursively_choose_best_split": 0,
    "block_split": 1,
    "split_without_postprocessing": 3,

    # synonyms
    "licketysplit": 0,
    "lickety_split": 0,
    "lickety": 0,
    "greedy": 0,
    "default": 0,
    "recursive": 0,
    "recursively_choose": 0,
    "choose_best_split": 0,

    "block": 1,
    "cyclic": 1,
    "cyclic_k": 1,

    "no_postprocessing": 3,
    "no_postprocess": 3,
    "without_postprocessing": 3,
    "split_no_postprocessing": 3,
    "split": 3,
}


def parse_proxy_style(proxy_style):
    # accepts int or string
    if isinstance(proxy_style, (int, np.integer)):
        v = int(proxy_style)
        if v in (0, 1, 3):
            return v
        raise ValueError(
            f"proxy_style={v} is not supported. "
            f"Supported oracle styles are 0, 1, 3."
        )

    key = _normalize_key(proxy_style)
    if key in _PROXY_STYLE_MAP:
        return _PROXY_STYLE_MAP[key]

    allowed = sorted(set(_PROXY_STYLE_MAP.keys()))
    raise ValueError(
        f"Unknown proxy_style='{proxy_style}'. "
        f"Supported: oracle_style 0/1/3, or one of: {allowed}"
    )

_GREEDY_HEURISTIC_MAP = {
    "entropy": 0,
    "info_gain": 0,
    "information_gain": 0,
    "ig": 0,

    "entropy_depth1_exact": 1,
    "entropy_with_depth1_exact": 1,
    "depth1_exact": 1,
    "default": 1,

    "best_split_for_leaves": 2,
    "min_child_leaf_objective": 2,
    "min_child_objective": 2,
    "always_misclassification_minimizing": 2,
    "misclassification_minimizing": 2,
    "misclassification_based": 2,
}


def parse_heuristic_for_greedy(heuristic_for_greedy):
    # accepts int or string
    if isinstance(heuristic_for_greedy, (int, np.integer)):
        v = int(heuristic_for_greedy)
        if v in (0, 1, 2):
            return v
        raise ValueError(
            f"heuristic_for_greedy={v} is invalid. Supported: 0,1,2."
        )

    key = _normalize_key(heuristic_for_greedy)
    if key in _GREEDY_HEURISTIC_MAP:
        return _GREEDY_HEURISTIC_MAP[key]

    allowed = sorted(set(_GREEDY_HEURISTIC_MAP.keys()))
    raise ValueError(
        f"Unknown heuristic_for_greedy='{heuristic_for_greedy}'. "
        f"Supported: 0/1/2, or one of: {allowed}"
    )

def _validate_binary_X(X):
    X_arr = np.asarray(X)

    if X_arr.ndim != 2:
        raise ValueError(
            f"X must be a 2D array of binary features with shape (n_samples, n_features). "
            f"Got shape {X_arr.shape}."
        )

    if X_arr.size == 0:
        raise ValueError("X must be non-empty.")

    bad_mask = ~((X_arr == 0) | (X_arr == 1))
    if np.any(bad_mask):
        bad_vals = np.unique(X_arr[bad_mask])
        shown = bad_vals[:10]
        raise ValueError(
            "PRAXIS expects X to already be binary, with entries only in {0, 1}. "
            f"Found non-binary values such as {shown}. "
            "For continuous, ordinal, or categorical features, use ThresholdGuessBinarizer "
            "first to convert X into binary threshold features."
        )

    return np.asarray(X_arr, dtype=np.uint8)


def _validate_class_labels(y):
    y_arr = np.asarray(y)

    if y_arr.ndim != 1:
        raise ValueError(
            f"y must be a 1D array of class labels. Got shape {y_arr.shape}."
        )

    if y_arr.size == 0:
        raise ValueError("y must be non-empty.")

    if not np.issubdtype(y_arr.dtype, np.integer):
        # allow floats only if they are exactly integer-valued
        if np.issubdtype(y_arr.dtype, np.floating) and np.all(np.isfinite(y_arr)) and np.all(y_arr == np.floor(y_arr)):
            y_arr = y_arr.astype(int)
        else:
            raise ValueError(
                "PRAXIS expects y to contain integer class labels numbered "
                "0, 1, ..., num_classes - 1. "
                f"Got dtype {y_arr.dtype}. Please renumber your classes before fitting."
            )

    y_arr = np.asarray(y_arr, dtype=int)
    classes = np.unique(y_arr)

    if classes[0] < 0:
        raise ValueError(
            "PRAXIS expects y labels to be numbered 0, 1, ..., num_classes - 1. "
            f"Found negative label(s): {classes[classes < 0]}. "
            "Please renumber your classes before fitting."
        )

    expected = np.arange(classes.size)
    if not np.array_equal(classes, expected):
        raise ValueError(
            "PRAXIS expects y labels to be consecutive integers numbered "
            "0, 1, ..., num_classes - 1. "
            f"Found labels {classes}, but expected {expected}. "
            "Please renumber your classes before fitting."
        )

    return y_arr

class PRAXIS:
    def __init__(self):
        self._model = _PRAXISCore()
        self._rid_out = None

    def fit(
        self,
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
    ):
        X = _validate_binary_X(X)
        y = _validate_class_labels(y)

        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X and y must have the same number of samples. "
                f"Got X.shape[0]={X.shape[0]} and y.shape[0]={y.shape[0]}."
            )

        proxy_style_int = parse_proxy_style(proxy_style)
        greedy_heur_int = parse_heuristic_for_greedy(heuristic_for_greedy)
        
        rashomon_mode = not bool(proxy_only)
        
        if root_budget is None:
            root_budget_int = -1
        else:
            root_budget_int = int(root_budget)
        self._model.fit(
            X,
            y,
            lambda_reg,
            depth_budget,
            rashomon_mult,
            multiplicative_slack,
            key_mode,
            False,
            lookahead_k,
            root_budget_int,
            bool(use_budget_refinement), 
            bool(guarantee_rule_list_recovery), 
            int(proxy_style_int), 
            bool(majority_leaf_only),
            bool(cache_early_exits),
            int(greedy_heur_int),
            bool(proxy_caching),
            int(num_proxy_features),
            bool(rashomon_mode),
        )

    def count_trees(self):
        return self._model.count_trees()

    def get_min_objective(self):
        return self._model.get_min_objective()

    def get_root_histogram(self):
        return self._model.get_root_histogram()
    
    def get_tree_objective(self, tree_index: int):
        obj, obj_norm = self._model.get_tree_objective(int(tree_index))
        return obj, obj_norm
    
    def count_trees_within_mult(self, mult: float) -> int:
        hist = self.get_root_histogram()
        min_obj = self.get_min_objective()
        thresh = round((1.0 + mult) * min_obj)
        return sum(cnt for obj, cnt in hist if obj <= thresh)


    # WARNING: 1-indexed unlike features
    def get_tree_paths(self, tree_index: int):
        """
        returns (paths, predictions):
        - paths: list of lists of signed feature indices. these are 1-indexed but features are 0-indexed so must subtract 1.
          +f means "go left / True on feature f-1"
          -f means "go right / False on feature f-1".
        - predictions: list of 0/1 labels for each leaf.
        """
        return self._model.get_tree_paths(int(tree_index))
    
    def get_tree_paths_str(self, tree_index: int):
        """
        returns (paths_str, predictions) where:
        - paths_str is a list of strings like "[+0, -1, +2]"
        - indices are shifted by -1 so features are 0-indexed as one would expect
        """
        paths, preds = self.get_tree_paths(tree_index)

        out = []
        for p in paths:
            converted = []
            for v in p:
                if v >= 0:
                    converted.append(f"+{v - 1}")
                else:
                    converted.append(f"-{abs(v) - 1}")
            path_str = "[" + ", ".join(converted) + "]"
            out.append(path_str)

        return out, preds
    
    def get_predictions(self, tree_index: int, X):
        X = np.asarray(X, dtype=np.uint8)
        return self._model.get_predictions(int(tree_index), X)

    def get_all_predictions(self, X, stack: bool = False):
        X = np.asarray(X, dtype=np.uint8)
        return self._model.get_all_predictions(X, bool(stack))
    
    def plot_tree(self, tree_index: int, feature_names=None, figsize=(8, 6), ax=None, title=None, show=True):
        paths, preds = self.get_tree_paths(tree_index)

        # feature names if not given
        if feature_names is None:
            encodings = [abs(v) for path in paths for v in path]
            if encodings:
                max_f = max(encodings) - 1  # convert back to 0-based now that we don't need sign
            else:
                max_f = -1
            feature_names = [f"f{j}" for j in range(max_f + 1)]

        # convert path representation into an explicit tree structure
        class Node:
            __slots__ = ("feature", "left", "right", "prediction")
            def __init__(self):
                self.feature = None
                self.left = None
                self.right = None
                self.prediction = None

        root = Node()

        # build tree
        for path, pred in zip(paths, preds):
            cur = root
            for signed_f in path:
                f = abs(signed_f) - 1  # 0-based
                go_left = signed_f > 0  # + => left / True, - => right / False

                if cur.feature is None:
                    cur.feature = f
                    cur.left = Node()
                    cur.right = Node()

                cur = cur.left if go_left else cur.right

            cur.prediction = pred

        def collect_leaves_in_order(node, leaves):
            if node is None:
                return
            if node.prediction is not None or (node.left is None and node.right is None):
                leaves.append(node)
                return
            collect_leaves_in_order(node.left, leaves)
            collect_leaves_in_order(node.right, leaves)

        def tree_depth(node):
            if node is None:
                return 0
            if node.prediction is not None:
                return 1
            return 1 + max(tree_depth(node.left), tree_depth(node.right))

        def assign_positions_tree(root, positions):
            leaves = []
            collect_leaves_in_order(root, leaves)
            if not leaves:
                leaves = [root]

            leaf_x = {leaf: i for i, leaf in enumerate(leaves)}

            def dfs(node, depth):
                if node is None:
                    return
                if node.prediction is not None or (node.left is None and node.right is None):
                    x = leaf_x[node]
                    positions[node] = (x, -depth)
                    return
                dfs(node.left, depth + 1)
                dfs(node.right, depth + 1)
                x_left, _ = positions[node.left]
                x_right, _ = positions[node.right]
                positions[node] = (0.5 * (x_left + x_right), -depth)

            dfs(root, 0)
            return len(leaves)

        positions = {}
        n_leaves = assign_positions_tree(root, positions)
        depth = tree_depth(root)

        x_scale = 3.2
        y_scale = 2.2

        for node, (x, y) in list(positions.items()):
            positions[node] = (x * x_scale, y * y_scale)

        if ax is None:
            width = max(figsize[0], 1.6 * n_leaves)
            height = max(figsize[1], 1.4 * depth)
            fig, ax = plt.subplots(figsize=(width, height))
        else:
            fig = ax.figure

        ax.set_axis_off()

        def _edge_label_pos(x1, y1, x2, y2, frac=0.52, base_offset=0.28, side_sign=+1.0):
            mx = x1 + frac * (x2 - x1)
            my = y1 + frac * (y2 - y1)
            dx = x2 - x1
            dy = y2 - y1
            dist = (dx * dx + dy * dy) ** 0.5
            if dist == 0:
                return mx, my
            nx = -dy / dist
            ny =  dx / dist

            # more offset on short edges, less on long edges
            scale = min(1.8, max(0.9, 1.2 / (dist ** 0.5)))
            offset = base_offset * scale

            return mx + side_sign * offset * nx, my + side_sign * offset * ny


        def _edge_label(parent_feature_idx, is_left_branch):
            name = feature_names[parent_feature_idx]
            # left branch: feature is True => no prefix
            # right branch: feature is False => prefix "!" # no ! prefix anymore 
            return name if is_left_branch else f"{name}"
        
        def _shrink_segment(x1, y1, x2, y2, r1, r2):
            dx = x2 - x1
            dy = y2 - y1
            dist = (dx * dx + dy * dy) ** 0.5
            if dist == 0:
                return x1, y1, x2, y2
            ux = dx / dist
            uy = dy / dist
            return (
                x1 + ux * r1,
                y1 + uy * r1,
                x2 - ux * r2,
                y2 - uy * r2,
            )


        def draw_node(node):
            x, y = positions[node]
            internal_r = 0.34
            leaf_r = 0.40

            # draw edges + labels + recurse
            if node.left is not None:
                x2, y2 = positions[node.left]
                # ax.add_line(Line2D([x, x2], [y, y2], color="black", linewidth=2.2))
                r_parent = internal_r
                r_child = leaf_r if node.left.prediction is not None else internal_r

                sx, sy, ex, ey = _shrink_segment(x, y, x2, y2, r_parent, r_child)
                ax.add_line(Line2D([sx, ex], [sy, ey], color="#4D4D4D", linewidth=2.2))
                
                # if node.feature is not None and node.prediction is None:
                #     tx, ty = _edge_label_pos(x, y, x2, y2, frac=0.52, base_offset=0.28, side_sign=+1.0)
                #     ax.text(
                #         tx, ty,
                #         _edge_label(node.feature, True),
                #         ha="center", va="center", fontsize=11,
                #         bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="none", alpha=0.9),
                #     )
                draw_node(node.left)

            if node.right is not None:
                x2, y2 = positions[node.right]
                #ax.add_line(Line2D([x, x2], [y, y2], color="black", linewidth=2.2))
                r_parent = internal_r
                r_child = leaf_r if node.right.prediction is not None else internal_r

                sx, sy, ex, ey = _shrink_segment(x, y, x2, y2, r_parent, r_child)
                ax.add_line(Line2D([sx, ex], [sy, ey], color="#4D4D4D", linewidth=2.2))

                # if node.feature is not None and node.prediction is None:
                #     tx, ty = _edge_label_pos(x, y, x2, y2, frac=0.52, base_offset=0.28, side_sign=-1.0)
                #     ax.text(
                #         tx, ty,
                #         _edge_label(node.feature, False),
                #         ha="center", va="center", fontsize=11,
                #         bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="none", alpha=0.9),
                #     )
                draw_node(node.right)

            # draw node
            if node.prediction is None:
                face = "#DCEAF4"
                edge = "#4D4D4D"
                text_color = "#222222"
                radius = internal_r
                label = None
            else:
                if int(node.prediction) == 0:
                    face = "#E69F00"
                else:
                    face = "#009E73"
                edge = "#4D4D4D"
                text_color = "#111111"
                radius = leaf_r
                label = str(node.prediction)

                edge = "#4D4D4D"
                text_color = "#111111"
                radius = leaf_r
                label = str(node.prediction)

            circ = Circle(
                (x, y),
                radius,
                facecolor=face,
                edgecolor=edge,
                linewidth=1.6
            )
            ax.add_patch(circ)

            if node.prediction is None and node.feature is not None:
                # feature name above internal node
                ax.text(
                    x, y + radius + 0.22,
                    feature_names[node.feature],
                    ha="center", va="bottom",
                    fontsize=11,
                    color="#222222",
                    bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.95),
                    zorder=10,
                )

            if label is not None:
                ax.text(
                    x, y,
                    label,
                    ha="center", va="center",
                    fontsize=12,
                    fontweight="bold",
                    color="white",
                    zorder=10,
                )
        draw_node(root)

        xs, ys = zip(*positions.values())
        pad_x = 1.6
        pad_y = 1.6
        ax.set_xlim(min(xs) - pad_x, max(xs) + pad_x)
        ax.set_ylim(min(ys) - pad_y, max(ys) + pad_y)
        ax.set_aspect("equal", adjustable="box")
        ax.set_axis_off()
        ax.set_title(f"PRAXIS Tree {tree_index}" if title is None else str(title))
        if show:
            plt.show()
        return fig, ax


        
    def get_tree_frontier_scores(self, tree_index: int, depth_budget: int):
        # returns a list of (depth_from_root, frontier_score) for each internal node of the specified tree. Root has depth 0.
        return self._model.get_tree_frontier_scores(int(tree_index), int(depth_budget))

    def root_lickety_objective_lookahead1(self, depth_budget: int):
        return int(self._model.root_lickety_objective_lookahead1(int(depth_budget)))

    def compute_rid(
        self,
        X,
        y,
        n_boot=10,
        lambda_reg=0.01,
        depth_budget=5,
        rashomon_mult=0.03,
        lookahead_k=1,
        seed=0,
        memory_efficient=False,
        binning_map=None,
    ):
        X = np.asarray(X, dtype=np.uint8)
        y = np.asarray(y, dtype=int)

        self._rid_out = _rid_subtractive_core(
            X,
            y,
            int(n_boot),
            float(lambda_reg),
            int(depth_budget),
            float(rashomon_mult),
            int(lookahead_k),
            int(seed),
            bool(memory_efficient),
            binning_map,
        )
        return self._rid_out
    
    def _require_rid(self):
        if self._rid_out is None:
            raise RuntimeError("RID not computed. Call compute_rid(...) first.")
        return self._rid_out
    
    def rid_plot_mean(self, feature_names=None, **kwargs):
        rid_out = self._require_rid()
        return rid_plot_mean(rid_out, feature_names=feature_names, **kwargs)

    def rid_plot_violin(self, feature_names=None, **kwargs):
        rid_out = self._require_rid()
        return rid_plot_violin(rid_out, feature_names=feature_names, **kwargs)

    def rid_plot_cdfs(self, feature_names=None, **kwargs):
        rid_out = self._require_rid()
        return rid_plot_cdfs(rid_out, feature_names=feature_names, **kwargs)
    
    @staticmethod
    def _require_binary_predictions(preds):
        a = np.asarray(preds)
        u = np.unique(a)
        ok = np.all((u == 0) | (u == 1))
        if not ok:
            raise ValueError(f"We require predictions in {{0,1}}.")

    def get_p_per_sample(self, X, tree_indices=None):
        # returns p_i per sample, the proportion of models predicting 1 - require binary predicitons
        # tree_indices : iterable[int] | None. if none, uses all trees, otherwise averages over the tree indices provided.
       
        X = np.asarray(X, dtype=np.uint8)

        if tree_indices is None:
            P = self.get_all_predictions(X, stack=True)
        else:
            idxs = list(tree_indices)
            if len(idxs) == 0:
                raise ValueError("tree_indices is empty.")
            preds_list = [self.get_predictions(int(t), X) for t in idxs]
            P = np.stack(preds_list, axis=0)

        self._require_binary_predictions(P)

        return P.mean(axis=0)

    def get_variance_per_sample(self, X, tree_indices=None):
        # returns per-sample variance of hard predictions across trees: p_i(1-p_i)
        X = np.asarray(X, dtype=np.uint8)

        if tree_indices is None:
            P = self.get_all_predictions(X, stack=True)  # (T, N)
        else:
            idxs = list(tree_indices)
            preds_list = [self.get_predictions(int(t), X) for t in idxs]
            P = np.stack(preds_list, axis=0)

        P = np.asarray(P)
        self._require_binary_predictions(P)

        return P.var(axis=0, ddof=0)

    def get_avg_variance_across_samples(self, X, tree_indices=None):
        v = self.get_variance_per_sample(X, tree_indices=tree_indices)
        return float(np.mean(v))

    def plot_disagreement_cdf(self, X, tree_indices=None, ax=None, figsize=(6.5, 4.0), title="Disagreement across samples", show=True, label=None):
        # plots proportion of points where variance is at most threshold t.
        v = self.get_variance_per_sample(X, tree_indices=tree_indices)
        v = np.asarray(v, float)
        n = v.size
    
        xs = np.sort(v)
        F = (np.arange(1, n + 1, dtype=float) / float(n))

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure

        ax.step(xs, F, where="post", linewidth=2.0, label=label)
        ax.set_ylabel("Proportion with var ≤ t")

        ax.set_xlabel("Variance threshold t")
        ax.set_ylim(-0.02, 1.02)
        ax.set_title(title)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, alpha=0.25)

        if label is not None:
            ax.legend(frameon=False)

        fig.tight_layout()
        if show:
            plt.show()
        return fig, ax

def _rid_style_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.25)


def _rid_feature_names(feature_names, V):
    if feature_names is None:
        return [f"f{j}" for j in range(V)]
    if len(feature_names) != V:
        raise ValueError(f"feature_names must have length {V}, got {len(feature_names)}")
    return list(feature_names)


def _rid_sorted_xy(xs, ps):
    xs = np.asarray(xs, float)
    ps = np.asarray(ps, float)
    if xs.size == 0:
        return xs, ps
    idx = np.argsort(xs)
    xs, ps = xs[idx], ps[idx]
    ps = np.clip(ps, 0.0, 1.0)
    ps = np.maximum.accumulate(ps)
    return xs, ps


def _rid_pmf_from_cdf(xs, ps):
    if xs.size == 0:
        return xs, np.asarray([], float)
    pprev = np.concatenate([[0.0], ps[:-1]])
    w = ps - pprev
    w = np.maximum(w, 0.0)
    s = w.sum()
    if s <= 0:
        w = np.ones_like(w) / max(1, w.size)
    else:
        w = w / s
    return xs, w


def rid_plot_mean(
    rid_out,
    feature_names=None,
    figsize=(10, 3),
    ax=None,
    title="RID mean reliance per feature",
    show=True,
):
    mean = np.asarray(rid_out["mean_sub_mr"], dtype=float)
    V = int(mean.size)
    feature_names = _rid_feature_names(feature_names, V)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    x = np.arange(V)
    ax.scatter(x, mean, s=30)
    ax.set_xticks(x)
    ax.set_xticklabels(feature_names, rotation=45, ha="right")
    ax.set_xlabel("feature")
    ax.set_ylabel("mean reliance\n(average accuracy drop when scrambled)")
    ax.set_title(title)
    _rid_style_ax(ax)

    fig.tight_layout()
    if show:
        plt.show()
    return fig, ax


def rid_plot_violin(
    rid_out,
    feature_names=None,
    samples_per_feature=4000,
    seed=123,
    figsize=(10, 6),
    ax=None,
    title="RID distribution per feature",
    show=True,
):
    mean = np.asarray(rid_out["mean_sub_mr"], dtype=float)
    cdf_x = rid_out["cdf_x"]
    cdf_p = rid_out["cdf_p"]

    V = int(mean.size)
    feature_names = _rid_feature_names(feature_names, V)

    order = np.argsort(-mean)
    mean_s = mean[order]
    names_s = [feature_names[j] for j in order]

    cdf_pairs = []
    xmin, xmax = 0.0, 0.0
    for j in range(V):
        xs, ps = _rid_sorted_xy(cdf_x[j], cdf_p[j])
        cdf_pairs.append((xs, ps))
        if xs.size:
            xmin = min(xmin, float(xs[0]))
            xmax = max(xmax, float(xs[-1]))

    rng = np.random.default_rng(seed)
    samples_sorted = []
    for j in order:
        xs, ps = cdf_pairs[int(j)]
        xs, w = _rid_pmf_from_cdf(xs, ps)
        if xs.size == 0:
            samples_sorted.append(np.zeros(int(samples_per_feature), float))
        else:
            samples_sorted.append(
                rng.choice(xs, size=int(samples_per_feature), replace=True, p=w)
            )

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    parts = ax.violinplot(
        samples_sorted,
        positions=np.arange(V),
        vert=False,
        widths=0.85,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )
    for body in parts["bodies"]:
        body.set_alpha(0.65)

    med_sorted = np.array([np.median(s) for s in samples_sorted])
    # ax.scatter(med_sorted, np.arange(V), s=18, zorder=3, label="median")
    # ax.scatter(mean_s, np.arange(V), s=22, zorder=3, marker="x", label="mean")
    # median: white dot with black outline (high contrast on top of violin)
    ax.scatter(
        med_sorted, np.arange(V),
        s=44, zorder=4,
        facecolors="white", edgecolors="black", linewidths=1.2,
        label="median",
    )
    ax.scatter(
        mean_s, np.arange(V),
        s=46, zorder=4,
        marker="x", c="black", linewidths=1.6,
        label="mean",
    )


    ax.set_yticks(np.arange(V))
    ax.set_yticklabels(names_s)
    ax.set_xlabel("Accuracy drop when scrambled")
    ax.set_ylabel("feature")
    ax.set_title(title)
    ax.set_xlim(xmin, xmax)
    _rid_style_ax(ax)
    ax.legend(frameon=False, fontsize=9, loc="lower right")

    fig.tight_layout()
    if show:
        plt.show()
    return fig, ax


def rid_plot_cdfs(
    rid_out,
    feature_names=None,
    figsize=(10, 4.5),
    ax=None,
    title="RID CDFs (all features overlaid)",
    cmap_name="tab20",
    legend_ncol=5,
    legend_fontsize=9,
    show=True,
):
    cdf_x = rid_out["cdf_x"]
    cdf_p = rid_out["cdf_p"]
    mean = np.asarray(rid_out["mean_sub_mr"], dtype=float)

    V = int(mean.size)
    feature_names = _rid_feature_names(feature_names, V)

    cdf_pairs = []
    xmin, xmax = 0.0, 0.0
    for j in range(V):
        xs, ps = _rid_sorted_xy(cdf_x[j], cdf_p[j])
        cdf_pairs.append((xs, ps))
        if xs.size:
            xmin = min(xmin, float(xs[0]))
            xmax = max(xmax, float(xs[-1]))

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    cmap = get_cmap(cmap_name)
    colors = [cmap(i % 20) for i in range(V)]

    for j in range(V):
        xs, ps = cdf_pairs[j]
        if xs.size == 0:
            continue
        ax.plot(xs, ps, linewidth=1.8, alpha=0.9, color=colors[j], label=feature_names[j])

    ax.set_title(title)
    ax.set_xlabel("Accuracy drop when scrambled")
    ax.set_ylabel("P[Δ accuracy ≤ t]")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(-0.02, 1.02)
    _rid_style_ax(ax)

    ax.legend(
        ncol=int(legend_ncol),
        fontsize=float(legend_fontsize),
        frameon=False,
        handlelength=1.8,
        columnspacing=1.1,
    )

    fig.tight_layout()
    if show:
        plt.show()
    return fig, ax




