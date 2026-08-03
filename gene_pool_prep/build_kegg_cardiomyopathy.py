"""
build_kegg_cardiomyopathy.py — KEGG cardiomyopathy pathway gene sets,
intersected against the QC-filtered gene universe.

Source
------
KEGG REST API (https://rest.kegg.jp), fetched once on 2026-08-02:

    /link/hsa/path:hsa05410   -> raw/kegg_hsa05410_link.tsv   (dilated CM)
    /link/hsa/path:hsa05414   -> raw/kegg_hsa05414_link.tsv   (hypertrophic CM)
    /list/hsa                 -> raw/kegg_hsa_gene_list.tsv   (ID -> symbol)

Enrichr's KEGG_2021_Human GMT was the fallback option; it was not needed,
rest.kegg.jp answered directly with HTTP 200 on all three endpoints.

Those three files under `raw/` ARE the acquisition. This script parses the
cached copies — it does NOT hit the network on a normal run. Re-fetching is
an explicit, manual act (`--refetch`), not something a pipeline run does.

Gene universe
-------------
`gene_symbols.npy` (49,231 symbols, already post-QC-mask) is reused exactly
as-is. `kept_gene_mask.npy` is loaded only to assert the two agree; the QC
filter is never recomputed here.

Symbol matching
---------------
The universe uses ARCHS4-style uppercase symbols (`C1ORF112`); KEGG uses
HGNC casing (`C1orf112`). All matching is therefore done on uppercased
symbols, and the universe's own spelling is what gets written out.

A KEGG entry that misses on its primary symbol is retried against its
KEGG-declared aliases. The `matched_via` column records which path each
gene took so an alias-rescued match can be audited or dropped later.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RAW = HERE / "raw"

EXPRESSION_DIR = (
    HERE.parent / "eda" / "dataset" / "cvd_data" / "elasticnet_out" / "expression"
)

PATHWAYS = {
    "hsa05410": "dilated cardiomyopathy",
    "hsa05414": "hypertrophic cardiomyopathy",
}

# A pathway returning far fewer genes than this means a truncated or failed
# response, not a real result. KEGG's cardiomyopathy pathways both sit
# around 90-105 genes.
MIN_EXPECTED_GENES = 60


def refetch() -> None:
    """Re-download the three KEGG endpoints into raw/. Manual use only."""
    import urllib.request

    targets = [
        ("https://rest.kegg.jp/link/hsa/path:hsa05410", "kegg_hsa05410_link.tsv"),
        ("https://rest.kegg.jp/link/hsa/path:hsa05414", "kegg_hsa05414_link.tsv"),
        ("https://rest.kegg.jp/list/hsa", "kegg_hsa_gene_list.tsv"),
    ]
    RAW.mkdir(parents=True, exist_ok=True)
    for url, name in targets:
        with urllib.request.urlopen(url, timeout=120) as resp:
            body = resp.read()
        (RAW / name).write_bytes(body)
        print(f"fetched {url} -> raw/{name} ({len(body):,} bytes)")


def load_symbol_map() -> dict[str, tuple[str, list[str]]]:
    """KEGG gene id -> (primary symbol, aliases).

    Line format: `hsa:7273<TAB>CDS<TAB>2:complement(...)<TAB>TTN, CMD1G, ...; titin`
    The names field is `SYMBOL, ALIAS, ALIAS; description`.

    ~1,400 entries carry a bare description and no `;` at all (predicted
    loci, ncRNAs). Those declare no symbol, so they are left unmapped rather
    than having their description mistaken for one.
    """
    mapping: dict[str, tuple[str, list[str]]] = {}
    with (RAW / "kegg_hsa_gene_list.tsv").open(encoding="utf-8") as fh:
        for line in fh:
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 4:
                continue
            kegg_id, names = fields[0], fields[3]
            if ";" not in names:
                continue
            symbol_part = names.split(";", 1)[0]
            symbols = [s.strip() for s in symbol_part.split(",") if s.strip()]
            if not symbols:
                continue
            mapping[kegg_id] = (symbols[0], symbols[1:])
    return mapping


def load_pathway_ids(pathway: str) -> list[str]:
    """KEGG gene ids belonging to one pathway, from its cached link file."""
    ids: list[str] = []
    with (RAW / f"kegg_{pathway}_link.tsv").open(encoding="utf-8") as fh:
        for line in fh:
            fields = line.rstrip("\n").split("\t")
            if len(fields) == 2 and fields[1].startswith("hsa:"):
                ids.append(fields[1])
    return ids


def load_universe() -> tuple[list[str], dict[str, str]]:
    """The 49,231-gene QC-filtered universe, reused exactly as it exists."""
    symbols = np.load(EXPRESSION_DIR / "gene_symbols.npy", allow_pickle=True)
    mask = np.load(EXPRESSION_DIR / "kept_gene_mask.npy", allow_pickle=True)

    if int(mask.sum()) != symbols.shape[0]:
        raise SystemExit(
            f"gene universe inconsistent: kept_gene_mask sums to {int(mask.sum())} "
            f"but gene_symbols.npy holds {symbols.shape[0]} entries"
        )

    universe = [str(s) for s in symbols]
    # Uppercased lookup -> the universe's own canonical spelling.
    lookup: dict[str, str] = {}
    for sym in universe:
        lookup.setdefault(sym.upper(), sym)
    return universe, lookup


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--refetch",
        action="store_true",
        help="re-download from rest.kegg.jp before building (manual use only)",
    )
    args = parser.parse_args()

    if args.refetch:
        refetch()

    symbol_map = load_symbol_map()
    _, lookup = load_universe()

    # gene symbol -> set of pathways it appears in
    members: dict[str, set[str]] = {}
    unmapped: dict[str, list[str]] = {}
    raw_counts: dict[str, int] = {}

    for pathway in PATHWAYS:
        ids = load_pathway_ids(pathway)
        raw_counts[pathway] = len(ids)
        if len(ids) < MIN_EXPECTED_GENES:
            raise SystemExit(
                f"FAIL: {pathway} returned only {len(ids)} genes "
                f"(expected >= {MIN_EXPECTED_GENES}). Refusing to proceed with a "
                f"possibly-truncated pathway list — re-run with --refetch."
            )
        for kegg_id in ids:
            entry = symbol_map.get(kegg_id)
            if entry is None:
                unmapped.setdefault(pathway, []).append(kegg_id)
                continue
            primary, _ = entry
            members.setdefault(primary, set()).add(pathway)

    kegg_total = len(members)

    rows = []
    n_alias_rescued = 0
    missing: list[str] = []

    for gene in sorted(members):
        pathways = members[gene]
        source = "both" if len(pathways) == 2 else next(iter(pathways))

        canonical = lookup.get(gene.upper())
        matched_via = "primary_symbol"

        if canonical is None:
            # Retry through this gene's KEGG-declared aliases.
            aliases: list[str] = []
            for _, (primary, alist) in symbol_map.items():
                if primary == gene:
                    aliases = alist
                    break
            for alias in aliases:
                hit = lookup.get(alias.upper())
                if hit is not None:
                    canonical = hit
                    matched_via = f"alias:{alias}"
                    n_alias_rescued += 1
                    break

        if canonical is None:
            missing.append(gene)
            continue

        rows.append(
            {
                "gene": canonical,
                "kegg_symbol": gene,
                "source_pathway": source,
                "matched_via": matched_via,
            }
        )

    kept = len(rows)
    dropped = kegg_total - kept
    pct = 100.0 * kept / kegg_total if kegg_total else 0.0

    out_path = HERE / "kegg_cardiomyopathy_genes.csv"
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        fh.write("# KEGG cardiomyopathy pathway genes, QC-universe-intersected\n")
        fh.write("# source: KEGG REST API (rest.kegg.jp), fetched 2026-08-02\n")
        fh.write(f"#   hsa05410 (dilated CM): {raw_counts['hsa05410']} genes\n")
        fh.write(f"#   hsa05414 (hypertrophic CM): {raw_counts['hsa05414']} genes\n")
        fh.write(f"#   union, deduplicated: {kegg_total} genes\n")
        fh.write(
            f"# intersected against QC-filtered universe (49,231 genes, "
            f"gene_symbols.npy): {kept} survive, {dropped} dropped ({pct:.1f}% kept)\n"
        )
        fh.write(f"#   of which alias-rescued (not a primary-symbol match): {n_alias_rescued}\n")
        writer = csv.DictWriter(
            fh, fieldnames=["gene", "kegg_symbol", "source_pathway", "matched_via"]
        )
        writer.writeheader()
        writer.writerows(rows)

    both = sum(1 for r in rows if r["source_pathway"] == "both")
    print(f"hsa05410 (dilated CM):      {raw_counts['hsa05410']} genes from KEGG")
    print(f"hsa05414 (hypertrophic CM): {raw_counts['hsa05414']} genes from KEGG")
    print(f"union, deduplicated:        {kegg_total} genes")
    print(f"after QC-universe intersection: {kept} kept, {dropped} dropped ({pct:.1f}%)")
    print(f"  alias-rescued matches: {n_alias_rescued}")
    print(f"  in both pathways: {both}")
    if unmapped:
        for pathway, ids in unmapped.items():
            print(f"  WARNING {pathway}: {len(ids)} KEGG ids had no symbol: {ids}")
    if missing:
        print(f"  dropped (absent from QC universe): {sorted(missing)}")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
