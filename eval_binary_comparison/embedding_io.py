"""One correct way to read an embedding parquet.

WHY THIS MODULE EXISTS
----------------------
The embedding columns are named `e0000 … e4095`. Selecting them with
`c.startswith("e0")` — the obvious-looking filter, and the one used in
`extract_llm_latents.py::load_population` — matches only `e0000` … `e0999`.
That silently returns **1000 of 4096 dimensions** and every downstream number is
then computed on a quarter of the representation, with no error anywhere.

It is silent because it is not wrong for every file: the BulkFormer-93M parquet
is 515-dimensional, so `e0000`…`e0514` all begin with "e0" and the filter picks
up all of them. It only breaks past 1000 dimensions — i.e. exactly on the LLM
latents, which are the thing under test.

This trap has now been hit twice in this project. The first time left a file
called `llm_latent_probe_TRUNCATED1000.json` in this directory; the second time
produced a three-way probe reporting `dim=1000` for a 4096-d feature set.

`linear_probe/probe.py:122` has always had the correct rule. This module is that
rule, in one place, so a third occurrence is not possible — and it ASSERTS the
recovered width against the parquet's own column count rather than trusting the
filter.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def embedding_columns(names) -> list[str]:
    """Every embedding column, ordered. `e` followed by digits — nothing else."""
    return sorted(c for c in names if c.startswith("e") and c[1:].isdigit())


def load_embeddings(path: Path, id_col: str = "sample_index"):
    """Return (X [n, d] float32, ids). Raises if any dimension went missing."""
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    cols = embedding_columns(table.schema.names)
    if not cols:
        raise ValueError(f"no embedding columns in {path}")

    # TWO guards, because the obvious one is not sufficient.
    #
    # (a) No gaps: names run e0000..e{d-1}, so the count must equal one past the
    #     highest index.
    expected = int(cols[-1][1:]) + 1
    if len(cols) != expected:
        raise ValueError(
            f"{path.name}: recovered {len(cols)} embedding columns but the "
            f"highest index is {cols[-1]} (implying {expected}) — gap in the "
            f"column range."
        )
    # (b) Nothing left behind. Guard (a) alone does NOT catch the bug this
    #     module exists to prevent: the truncated set e0000..e0999 has 1000
    #     columns and a highest index of e0999, so it is perfectly
    #     self-consistent and sails through. The only way to catch a selector
    #     that under-matches is to compare against the file itself. No metadata
    #     column in these parquets begins with "e", so every such name must
    #     have been selected.
    all_e = [c for c in table.schema.names if c.startswith("e")]
    if len(cols) != len(all_e):
        missed = sorted(set(all_e) - set(cols))[:5]
        raise ValueError(
            f"{path.name}: selected {len(cols)} of {len(all_e)} columns "
            f"beginning with 'e' — the selector under-matched, e.g. {missed}. "
            f"Every downstream metric would be computed on a subset."
        )

    d = table.to_pydict()
    n = len(d[id_col])
    X = np.empty((n, len(cols)), dtype=np.float32)
    for j, c in enumerate(cols):
        X[:, j] = np.asarray(d[c], dtype=np.float32)
    return X, np.asarray(d[id_col], dtype=np.int64)
