"""check_convergence.py — did the probe's lbfgs actually converge?

The LLM-latent rows are 4096-d; the BulkFormer row is 515-d. If lbfgs hits
max_iter on the wide features but converges comfortably on the narrow ones,
the LLM rows are under-fit and their AUROCs are pessimistic — which would
bias the comparison AGAINST the LLM. This fits ONE fold per variant with the
exact pipeline run_pooled_breakdown uses and reports n_iter_ vs max_iter.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

MAX_ITER = 2000
SEED = 20260707
K = 5

VARIANTS = ["LLM-latent-imgtok", "LLM-latent-meanpool", "BulkFormer-93M"]


def main():
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression

    from eval_binary_comparison.per_condition_probe import load_variant

    for tag in VARIANTS:
        p = Path("linear_probe/embeddings") / f"embeddings_{tag}.parquet"
        if not p.exists():
            print(f"[skip] {tag}: missing", flush=True)
            continue
        X, meta = load_variant(p)
        is_pos = meta["is_positive"].astype(bool)
        is_neg = meta["is_neg_hard"].astype(bool)
        sel = np.where(is_pos | is_neg)[0]
        X_s, y = X[sel], is_pos[sel].astype(int)
        groups = meta["series_id"][sel].astype(str)

        skf = StratifiedGroupKFold(n_splits=K, shuffle=True, random_state=SEED)
        tr, _ = next(iter(skf.split(X_s, y, groups)))

        pipe = Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=MAX_ITER, solver="lbfgs",
                                       class_weight="balanced", random_state=SEED)),
        ])
        pipe.fit(X_s[tr], y[tr])
        n_iter = int(pipe.named_steps["clf"].n_iter_[0])
        hit = n_iter >= MAX_ITER
        print(f"{tag:24s} dims={X.shape[1]:5d}  n_iter={n_iter:5d}/{MAX_ITER}  "
              f"{'*** HIT CAP (under-fit) ***' if hit else 'converged'}", flush=True)


if __name__ == "__main__":
    main()
