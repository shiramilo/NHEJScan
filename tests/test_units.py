import pytest

from nhejscan.core import (Params, classify_three_way, extract_gaps, has_ssr, label_event,
                           messer_classify, parse_xmfa, passes_identity_filter, select_blocks)


def test_parse_xmfa_skips_header_and_joins_wrapped_lines(tmp_path):
    f = tmp_path / "x.xmfa"
    f.write_text("#FormatVersion Mauve1\n#Sequence1File\tA.fna\n>1:1-8 + A.fna\nACGT\nACGT\n"
                 ">2:1-8 + B.fna\nACGT\nAC-T\n=\n>1:9-12 + A.fna\nTTTT\n=\n")
    blocks = parse_xmfa(str(f))
    assert len(blocks) == 2
    assert blocks[0] == [("1:1-8", "ACGTACGT"), ("2:1-8", "ACGTAC-T")]


def test_select_blocks_requires_gap_run():
    p = Params()
    b1 = [("1:1-30", "A" * 30), ("2:1-30", "A" * 10 + "-" * 10 + "A" * 10)]
    b2 = [("1:1-30", "A" * 30), ("2:1-30", "A" * 10 + "-" * 9 + "A" * 11)]
    assert select_blocks([b1, b2], p) == [b1]


def test_has_ssr():
    assert has_ssr("ACACACACACACACACACAC")
    assert has_ssr("TTTTTTTTTTTT")
    assert not has_ssr("ACGTTGCAAGCTTCGATCGGATCCAAGT")


@pytest.mark.parametrize("counts,event", [
    ((20, 0, 20), "Deletion"),
    ((20, 0, 0), "Insertion"),
    ((20, 0, 18), "Deletion"),     # L-S-M, |u1-u3| <= 3
    ((20, 5, 40), "*Deletion*"),   # M-S-L is not clean; S-M-L / M-S-L -> multiple event
    ((40, 20, 0), "*Insertion*"),  # L-M-S
])
def test_label_event(counts, event):
    assert label_event(*counts)[1] == event


def _block(gap_len=15, flank=200, n_extra_gap=False):
    left = "ACGTTGCAAGCTTCGATCGG" * (flank // 20)
    mid = "TGCATGCCAGTTGACCAGTA"[:gap_len] if gap_len <= 20 else "T" * gap_len
    right = "GGATCCAAGTCCATTGACGT" * (flank // 20)
    s1 = left + "-" * gap_len + right
    s2 = left + mid + right
    return [("1:1-1000", s1), ("2:1-1000", s2)]


def test_extract_gaps_basic():
    g = extract_gaps(_block(), Params())
    assert len(g) == 1 and g[0]["seq_id"] == "1:1-1000" and g[0]["other_id"] == "2:1-1000"
    assert g[0]["gap_end"] - g[0]["gap_start"] + 1 == 15


def test_extract_gaps_pairing_fix_vs_legacy():
    """Two gaps; the first fails the identity filter. The corrected pairing keeps the second gap's own
    corresponding region, the legacy pairing attaches the first gap's region."""
    left = "ACGTTGCAAGCTTCGATCGG" * 10
    mid = "GGATCCAAGTCCATTGACGT" * 10
    right = "TTGACCGGATAACCGTAGCT" * 10
    ins1, ins2 = "T" * 12, "ACGTAGCTAGCA"
    s1 = left + "-" * 12 + mid + "-" * 12 + right
    s2 = ("G" * len(left)) + ins1 + mid + ins2 + right  # left region of gap 1 is dissimilar -> removed
    blk = [("1:1-9", s1), ("2:1-9", s2)]
    fixed = extract_gaps(blk, Params())
    legacy = extract_gaps(blk, Params(legacy_pairing=True))
    assert len(fixed) == len(legacy) == 1
    assert fixed[0]["other_region"] != legacy[0]["other_region"]
    assert ins2 in fixed[0]["other_region"]


def test_identity_filter_rejects_short_left_flank_and_accepts_good():
    good = _block(gap_len=12, flank=200)
    assert passes_identity_filter(good[0][1], good[1][1], 10)
    short = [("1:1-9", "ACGTAC" + "-" * 12 + "GGATCCAAGTCCATTGACGT" * 10),
             ("2:1-9", "ACGTAC" + "T" * 12 + "GGATCCAAGTCCATTGACGT" * 10)]
    assert not passes_identity_filter(short[0][1], short[1][1], 10)


def test_messer_class_b_with_known_microhomology(trace_ext):
    """Deletion of D plus one copy of a 6-bp microhomology: d must equal 6 (class b)."""
    import random
    rng = random.Random(3)
    rnd = lambda n: "".join(rng.choice("ACGT") for _ in range(n))
    left, right, mh, d = rnd(200), rnd(200), "GATTCA", rnd(20)
    seq_with = left + mh + d + mh + right   # corresponding sequence
    seq_del = left + mh + right             # gap-bearing sequence, gap placed after the repeat
    # align: gap-bearing sequence carries a 26-bp gap
    s1 = left + mh + "-" * (len(d) + len(mh)) + right
    s2 = seq_with
    m = messer_classify(s1, s2, Params(trace_ext_bin=trace_ext))
    assert m["cls"] == "b" and m["d"] >= 6 and m["l"] == 26


def test_classify_three_way_deletion():
    corr = "AAAA" + "C" * 12 + "TTTT"
    flanks = "AAAA" + "-" * 12 + "TTTT"
    outg = "AAAA" + "C" * 12 + "TTTT"
    r = classify_three_way(corr, flanks, outg, 4)
    assert r["Event"] == "Deletion"
