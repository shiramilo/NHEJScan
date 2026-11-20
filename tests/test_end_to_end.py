"""End-to-end test on a simulated genome triplet with planted deletions / insertions."""
import collections
import json
import subprocess

import pandas as pd
import pytest

from conftest import have
from simulate import simulate

pytestmark = pytest.mark.skipif(not have("progressiveMauve", "mafft", "blastn", "makeblastdb"),
                                reason="external tools not installed")


def test_pipeline_recovers_planted_events(tmp_path, trace_ext):
    from nhejscan.core import Params, run_mauve, run_pair
    simulate(str(tmp_path), seed=7, genome_len=200_000, n_events=24, snp_ab=0.0)
    xmfa = str(tmp_path / "AB.xmfa")
    run_mauve(str(tmp_path / "A.fna"), str(tmp_path / "B.fna"), xmfa)
    summary, _, _ = run_pair(xmfa, str(tmp_path / "O.fna"), str(tmp_path / "out"),
                             Params(trace_ext_bin=trace_ext, threads=2))
    truth = pd.DataFrame(json.load(open(tmp_path / "truth.json")))
    ev = pd.read_csv(tmp_path / "out" / "events.csv")

    td = truth[truth.kind == "DEL"]
    found = collections.Counter(zip(ev[ev.Event == "Deletion"].l.astype(int), ev[ev.Event == "Deletion"].d.astype(int)))
    expect = collections.Counter(zip(td.length + td.mh, td.mh))
    assert found == expect, "confirmed deletions (length, microhomology) differ from the planted ones"

    ti = truth[truth.kind == "INS"]
    assert sorted(ev[ev.Event == "Insertion"].l.astype(int)) == sorted(ti.length)
    assert summary["n_errors"] == 0


def test_insertion_source_search(tmp_path, trace_ext):
    from nhejscan.core import Params, run_mauve, run_pair
    simulate(str(tmp_path), seed=5, genome_len=200_000, n_events=24, snp_ab=0.0, dup_frac=0.5)
    xmfa = str(tmp_path / "AB.xmfa")
    run_mauve(str(tmp_path / "A.fna"), str(tmp_path / "B.fna"), xmfa)
    summary, _, _ = run_pair(xmfa, str(tmp_path / "O.fna"), str(tmp_path / "out"),
                             Params(trace_ext_bin=trace_ext, threads=2),
                             genomes={1: str(tmp_path / "A.fna"), 2: str(tmp_path / "B.fna")})
    truth = pd.DataFrame(json.load(open(tmp_path / "truth.json")))
    ins = pd.read_csv(tmp_path / "out" / "insertions.csv")
    planted = truth[truth.dup]
    assert len(planted) > 0
    flagged = ins[ins.source_copy_found]
    # every templated insertion is traced to a second copy in its own genome, with an alignment written
    for L in planted.length:
        assert (flagged.insertion_length == L).any()
    # long random insertions have no second copy (short ones can match by chance, as in the notebook criterion)
    rnd = truth[(truth.kind == "INS") & (~truth.dup) & (truth.length > 30)]
    for L in rnd.length:
        assert not ((flagged.insertion_length == L).any() and not (planted.length == L).any())
    assert len(list((tmp_path / "out" / "mh_alignments").glob("*_updated.fna"))) == len(flagged)


def test_cli_pair_from_genomes(tmp_path, trace_ext):
    simulate(str(tmp_path), seed=11, genome_len=120_000, n_events=10, snp_ab=0.0)
    r = subprocess.run(["python3", "-m", "nhejscan", "pair", "--genome1", str(tmp_path / "A.fna"),
                        "--genome2", str(tmp_path / "B.fna"), "--outgroup", str(tmp_path / "O.fna"),
                        "--outdir", str(tmp_path / "run"), "--trace-ext", trace_ext, "--threads", "2"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert (tmp_path / "run" / "events.csv").exists() and (tmp_path / "run" / "summary.json").exists()
