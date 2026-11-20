"""Command-line interface: nhejscan {pair,screen,candidates}."""
import argparse
import json
import os
import shutil
import sys

import pandas as pd

from . import __version__
from .core import Params, run_mauve, run_pair
from . import pairs as P


def _params(a):
    return Params(min_gap=a.min_gap, max_gap=a.max_gap, flank_mult=a.flank_mult,
                  min_identity=a.min_identity, threads=a.threads, legacy_pairing=a.legacy_pairing,
                  trace_ext_bin=a.trace_ext)


def _common(ap):
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 1, help="worker processes (default: all CPUs)")
    ap.add_argument("--trace-ext", default=None, help="path to trace_ext (default: built-in / PATH)")
    ap.add_argument("--mauve-bin", default="progressiveMauve")
    ap.add_argument("--min-gap", type=int, default=10)
    ap.add_argument("--max-gap", type=int, default=1000)
    ap.add_argument("--flank-mult", type=int, default=5)
    ap.add_argument("--min-identity", type=float, default=80.0)
    ap.add_argument("--legacy-pairing", action="store_true",
                    help="reproduce the original notebook's gap/corresponding-region pairing (see README)")


def _resolve_trace_ext(a):
    if a.trace_ext:
        return a.trace_ext
    from .build import ensure_trace_ext
    return ensure_trace_ext()


def check_tools(mauve_needed=True):
    need = ["mafft", "blastn", "makeblastdb"] + (["progressiveMauve"] if mauve_needed else [])
    missing = [t for t in need if shutil.which(t) is None]
    if missing:
        sys.exit("Missing required tools on PATH: " + ", ".join(missing))


def cmd_pair(a):
    a.trace_ext = _resolve_trace_ext(a)
    p = _params(a)
    os.makedirs(a.outdir, exist_ok=True)
    xmfa = a.xmfa
    if not xmfa:
        check_tools(True)
        xmfa = os.path.join(a.outdir, "alignment.xmfa")
        if not (os.path.exists(xmfa) and os.path.getsize(xmfa) > 1000):
            print("Running progressiveMauve ...")
            run_mauve(a.genome1, a.genome2, xmfa, a.mauve_bin)
    else:
        check_tools(False)
    genomes = {}
    if a.genome1 and a.genome2:
        genomes = {1: a.genome1, 2: a.genome2}
    else:
        print("Genome FASTA files not given: insertion source search (steps 17-21) is skipped.")
    summary, _, _ = run_pair(xmfa, a.outgroup, a.outdir, p, genomes=genomes)
    summary["nhejscan_version"] = __version__
    json.dump(summary, open(os.path.join(a.outdir, "summary.json"), "w"), indent=1, default=str)
    print(json.dumps(summary, indent=1, default=str))


def cmd_candidates(a):
    tree = P.read_tree(a.tree)
    groups = P.groups_from_clade_map(pd.read_csv(a.clade_map))
    rows = P.enumerate_candidates(tree, groups, a.max_distance, P.load_exclude_pairs(a.exclude_pairs))
    acc = P.load_accession_map(a.accession_map)
    for r in rows:
        if a.genome_dir:
            for k, sp in (("genome1", r["species1"]), ("genome2", r["species2"]), ("outgroup_genome", r["outgroup"])):
                r[k + "_found"] = bool(P.find_genome_file(a.genome_dir, sp, acc))
    pd.DataFrame(rows).to_csv(a.out, index=False)
    print(f"{len(rows)} candidate pairs written to {a.out}")


def cmd_screen(a):
    a.trace_ext = _resolve_trace_ext(a)
    check_tools(True)
    p = _params(a)
    tree = P.read_tree(a.tree)
    groups = P.groups_from_clade_map(pd.read_csv(a.clade_map))
    acc = P.load_accession_map(a.accession_map)
    cands = P.enumerate_candidates(tree, groups, a.max_distance, P.load_exclude_pairs(a.exclude_pairs))
    print(f"{len(cands)} candidate pairs")
    rows = []
    for i, c in enumerate(cands, 1):
        g1, g2, g3 = (P.find_genome_file(a.genome_dir, c[k], acc) for k in ("species1", "species2", "outgroup"))
        row = dict(c)
        if not (g1 and g2 and g3):
            row["status"] = "SKIPPED_MISSING_GENOME"
            rows.append(row)
            continue
        d = os.path.join(a.outdir, f"{c['species1']}__{c['species2']}")
        done = os.path.join(d, "summary.json")
        if os.path.exists(done) and not a.force:
            row.update(json.load(open(done)))
            row["status"] = "OK_CACHED"
        else:
            print(f"[{i}/{len(cands)}] {c['species1']} x {c['species2']} (outgroup {c['outgroup']})")
            try:
                os.makedirs(d, exist_ok=True)
                xmfa = os.path.join(d, "alignment.xmfa")
                if not (os.path.exists(xmfa) and os.path.getsize(xmfa) > 1000):
                    run_mauve(g1, g2, xmfa, a.mauve_bin)
                s, _, _ = run_pair(xmfa, g3, d, p, genomes={1: g1, 2: g2})
                json.dump(s, open(done, "w"), indent=1, default=str)
                row.update(s)
                row["status"] = "OK"
            except Exception as e:
                row["status"] = f"ERROR: {type(e).__name__}: {e}"
        rows.append(row)
        pd.DataFrame(rows).to_csv(a.summary, index=False)
    pd.DataFrame(rows).to_csv(a.summary, index=False)
    print(f"Summary: {a.summary}")


def cmd_model(a):
    import subprocess
    env = dict(os.environ, NHEJSCAN_MODEL_INPUT=os.path.abspath(a.counts),
               NHEJSCAN_MODEL_OUTDIR=os.path.abspath(a.outdir), NHEJSCAN_MODEL_SEED=str(a.seed))
    os.makedirs(a.outdir, exist_ok=True)
    subprocess.run([sys.executable, "-m", "nhejscan.model"], check=True, env=env)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="nhejscan", description=__doc__)
    ap.add_argument("--version", action="version", version=__version__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("pair", help="run one genome pair (from genomes, or from an existing XMFA)")
    s.add_argument("--xmfa", help="existing progressiveMauve XMFA (skips the Mauve step)")
    s.add_argument("--genome1")
    s.add_argument("--genome2")
    s.add_argument("--outgroup", required=True, help="outgroup genome FASTA")
    s.add_argument("--outdir", required=True)
    _common(s)
    s.set_defaults(func=cmd_pair)

    c = sub.add_parser("candidates", help="list candidate pairs from a tree")
    c.add_argument("--tree", required=True)
    c.add_argument("--clade-map", required=True)
    c.add_argument("--max-distance", type=float, required=True)
    c.add_argument("--exclude-pairs")
    c.add_argument("--accession-map")
    c.add_argument("--genome-dir")
    c.add_argument("--out", default="candidate_pairs.csv")
    c.set_defaults(func=cmd_candidates)

    r = sub.add_parser("screen", help="run every candidate pair that has all three genomes")
    r.add_argument("--tree", required=True)
    r.add_argument("--clade-map", required=True)
    r.add_argument("--genome-dir", required=True)
    r.add_argument("--max-distance", type=float, required=True)
    r.add_argument("--outdir", default="runs")
    r.add_argument("--summary", default="screen_summary.csv")
    r.add_argument("--exclude-pairs")
    r.add_argument("--accession-map")
    r.add_argument("--force", action="store_true", help="re-run pairs that already have a summary.json")
    _common(r)
    r.set_defaults(func=cmd_screen)

    m = sub.add_parser("model", help="weighted c-NHEJ / MMEJ scoring model on a clade x length-class count table")
    m.add_argument("--counts", required=True, help="CSV: clade,INS_0_1,INS_2_4,INS_5_9,INS_10_plus,DEL_0_1,DEL_2_4,DEL_5_9,DEL_10_plus")
    m.add_argument("--outdir", required=True)
    m.add_argument("--seed", type=int, default=42)
    m.set_defaults(func=cmd_model)

    a = ap.parse_args(argv)
    if a.cmd == "pair" and not a.xmfa and not (a.genome1 and a.genome2):
        ap.error("pair needs --xmfa or both --genome1 and --genome2")
    a.func(a)


if __name__ == "__main__":
    main()
