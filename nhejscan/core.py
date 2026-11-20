"""
Core pipeline: Mauve XMFA -> INDEL candidates -> Messer & Arndt (2007) trace
extension -> outgroup polarisation -> event labels.s
"""
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from multiprocessing import Pool
from typing import Optional

import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq
from io import StringIO

CONVERSION = {'A': '0', 'C': '1', 'G': '2', 'T': '3', 'N': '4',
              'a': '0', 'c': '1', 'g': '2', 't': '3', 'n': '4', '-': '4'}

BLAST_FMT = "6 qseqid sseqid pident length qlen mismatch gapopen qstart qend sstart send evalue bitscore"


@dataclass
class Params:
    min_gap: int = 10            # minimum INDEL length (bp)
    max_gap: int = 1000          # maximum INDEL length (bp)
    flank_mult: int = 5          # flank = flank_mult x gap length
    min_identity: float = 80.0   # % identity of gap region vs. corresponding region
    ssr_threshold: float = 0.8   # fraction of gap-filling sequence covered by one repeat
    outgroup_flank: int = 100    # flank bases (each side) used for outgroup search
    legacy_pairing: bool = False  # reproduce the notebook's gap/corresponding-region pairing (see README)
    threads: int = 1
    trace_ext_bin: str = "trace_ext"
    mafft_bin: str = "mafft"
    blastn_bin: str = "blastn"
    makeblastdb_bin: str = "makeblastdb"


# ---------------------------------------------------------------------------
# Steps 1-4: XMFA parsing and block selection
# ---------------------------------------------------------------------------

def parse_xmfa(path):
    """Return list of blocks; each block is a list of (seq_id, aligned_sequence).

    '#' header lines are skipped. seq_id is the first token of the header
    (e.g. '1:4403-9025'); the genome index is the number before ':'.
    """
    blocks, cur, hdr, parts = [], [], None, []

    def flush():
        nonlocal hdr, parts
        if hdr is not None:
            cur.append((hdr, "".join(parts)))
        hdr, parts = None, []

    with open(path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith("#"):
                continue
            if line.startswith("="):
                flush()
                if cur:
                    blocks.append(cur)
                cur = []
            elif line.startswith(">"):
                flush()
                hdr = line[1:].split()[0]
            elif hdr is not None:
                parts.append(line.strip())
    flush()
    if cur:
        blocks.append(cur)
    return blocks


def select_blocks(blocks, p: Params):
    """Steps 2-4: keep blocks in which any sequence has a gap run >= min_gap.

    (Any run >= min_gap contains a min_gap run, so max_gap is applied later,
    when individual gaps are extracted.) Returned in XMFA order.
    """
    pat = "-" * p.min_gap
    return [b for b in blocks if any(pat in s for _, s in b)]


# ---------------------------------------------------------------------------
# Step 5: gap extraction with flanks + N / identity filters
# ---------------------------------------------------------------------------

def _identity(a, b):
    if len(a) != len(b) or not a:
        return 0.0
    return sum(x == y for x, y in zip(a, b)) / len(a) * 100


def extract_gaps(block, p: Params):
    """Step 5. Return list of dicts for every gap that passes the N and identity filters."""
    found = []
    for sid, seq in block:
        gs = ge = None
        for j, base in enumerate(seq):
            if base == "-":
                if gs is None:
                    gs = j
                ge = j
            elif gs is not None:
                size = ge - gs + 1
                if p.min_gap <= size <= p.max_gap:
                    fl = p.flank_mult * size
                    found.append((sid, gs, ge, seq[max(gs - fl, 0):min(ge + fl + 1, len(seq))]))
                gs = ge = None

    others = []
    for sid, gs, ge, _ in found:
        other = next(((oid, oseq) for oid, oseq in block if oid != sid), None)
        if other is None:
            others.append(None)
            continue
        size = ge - gs + 1
        fl = p.flank_mult * size
        others.append((other[0], other[1][max(gs - fl, 0):min(ge + fl + 1, len(other[1]))]))

    passed = []
    for idx, ((sid, gs, ge, region), oth) in enumerate(zip(found, others)):
        if oth is None:
            continue
        oid, oregion = oth
        size = ge - gs + 1
        if region.count("N") <= 2 and oregion.count("N") <= 2 \
                and region.count("N") <= size * 0.2 \
                and _identity(region, oregion) >= p.min_identity:
            passed.append((idx, sid, gs, ge, region, oid, oregion))

    out = []
    for k, (idx, sid, gs, ge, region, oid, oregion) in enumerate(passed):
        if p.legacy_pairing:
            # Original notebook zipped the *unfiltered* list of corresponding
            # regions against the *filtered* gap list, so after any earlier gap
            # in the same block was removed, later gaps were written with the
            # corresponding region of a different gap.
            oid, oregion = others[k][0], others[k][1]
        out.append(dict(seq_id=sid, other_id=oid, gap_start=gs, gap_end=ge,
                        region=region, other_region=oregion))
    return out


# ---------------------------------------------------------------------------
# Steps 9-11: MAFFT realignment, SSR and identity filters
# ---------------------------------------------------------------------------

def run_mafft(records, p: Params, extra=None, workdir=None):
    """records: list of (name, seq). Returns list of (name, seq) in input order."""
    with tempfile.TemporaryDirectory(dir=workdir) as td:
        inp = os.path.join(td, "in.fna")
        with open(inp, "w") as f:
            for n, s in records:
                f.write(f">{n}\n{s}\n")
        cmd = [p.mafft_bin, "--auto", "--thread", "1"] + (extra or []) + [inp]
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return [(r.id, str(r.seq)) for r in SeqIO.parse(StringIO(res.stdout), "fasta")]


def has_ssr(seq, threshold=0.8):
    """True if one repeat unit (period i, any start) tiles >= threshold of seq."""
    n = len(seq)
    for i in range(1, n // 2 + 1):
        if i * (n // i) / n < threshold:
            continue
        for j in range(n - 2 * i + 1):
            if seq[j:j + i] * (n // i) in seq:
                return True
    return False


def gap_region_in_corresponding(seq, corr, min_len):
    gs = seq.find("-" * min_len)
    if gs == -1:
        return None
    ge = gs + min_len
    while ge < len(seq) and seq[ge] == "-":
        ge += 1
    return corr[gs:ge]


def passes_identity_filter(seq1, seq2, min_gap):
    """Step 11. Returns True if the realigned pair is retained."""
    found = False
    matches = re.findall(r"-+", seq1) or re.findall(r"-+", seq2)
    for gap in matches:
        size = len(gap)
        if size < min_gap:
            continue
        found = True
        gs1 = seq1.find(gap)
        gseq, oseq = (seq1, seq2) if gs1 != -1 else (seq2, seq1)
        gs = gs1 if gs1 != -1 else seq2.find(gap)
        ge = gs + size
        fl_l, fl_r = gseq[gs - 2 * size:gs], gseq[ge:ge + 2 * size]
        if fl_l.upper().count("N") >= 2 or fl_r.upper().count("N") >= 2:
            return False
        region = oseq[gs:ge]
        if region.upper().count("N") / size * 100 > 20:
            return False
        # A left flank shorter than 3 gap lengths cannot be scored and is rejected
        # (the notebook reached the same outcome through a negative slice index).
        if gs - 3 * size < 0:
            return False
        li = _identity(gseq[gs - 3 * size:gs], oseq[gs - 3 * size:gs])
        ri = _identity(gseq[ge:ge + 3 * size], oseq[ge:ge + 3 * size])
        if li < 80 or ri < 80:
            return False
    return found


# ---------------------------------------------------------------------------
# Step 12: Messer & Arndt trace extension
# ---------------------------------------------------------------------------

def _num(s):
    return "".join(CONVERSION[c] for c in s)


def messer_classify(seq1, seq2, p: Params):
    """Return dict(right_rep,left_rep,l,d,l_minus_d,cls) or None if no analysable gap."""
    gs = ge = None
    for i, ch in enumerate(seq1):
        if ch == "-":
            if gs is None:
                gs = i
            ge = i
        elif gs is not None and i - ge > p.min_gap:
            break
    if gs is None:
        return None
    size = ge - gs + 1
    if size <= p.min_gap:
        return None
    left1, right1 = seq1[:gs], seq1[ge + 1:]
    left2, gap2, right2 = seq2[:gs], seq2[gs:ge + 1], seq2[ge + 1:]
    try:
        right = subprocess.check_output(
            [p.trace_ext_bin, _num(right1), _num(gap2 + right2), str(size)], text=True)
        left = subprocess.check_output(
            [p.trace_ext_bin, _num(left1)[::-1], _num(left2 + gap2)[::-1], str(size)], text=True)
    except subprocess.CalledProcessError:
        return None
    d = float(left) + float(right)
    if d > size:
        cls = "a"
    elif size > d > 0:
        cls = "b"
    elif d == size:
        cls = "c"
    elif d == 0:
        cls = "d"
    else:
        cls = "Unknown"
    return dict(right_rep=right.strip(), left_rep=left.strip(), l=size, d=d, l_minus_d=size - d, cls=cls)


# ---------------------------------------------------------------------------
# Steps 14-16: outgroup BLAST, 3-way MAFFT, L/M/S event labels
# ---------------------------------------------------------------------------

def make_blast_db(outgroup_fa, db_prefix, p: Params):
    subprocess.run([p.makeblastdb_bin, "-in", outgroup_fa, "-dbtype", "nucl", "-out", db_prefix],
                   check=True, capture_output=True)


def _blast_first_hit(query_fa_text, db_prefix, p: Params, workdir):
    with tempfile.TemporaryDirectory(dir=workdir) as td:
        q = os.path.join(td, "q.fna")
        with open(q, "w") as f:
            f.write(query_fa_text)
        res = subprocess.run(
            [p.blastn_bin, "-query", q, "-db", db_prefix, "-word_size", "5", "-gapopen", "5",
             "-gapextend", "2", "-reward", "2", "-penalty", "-3", "-num_threads", "1",
             "-outfmt", BLAST_FMT], check=True, capture_output=True, text=True)
    first = res.stdout.split("\n", 1)[0].strip()
    return first.split("\t") if first else None


def outgroup_sequence(hit, genome_dict, flank):
    """Hit sequence +/- flank from the outgroup genome, in query orientation."""
    sid, s, e = hit[1], int(hit[9]), int(hit[10])
    rev = s > e
    lo, hi = (e, s) if rev else (s, e)
    rec = genome_dict.get(sid)
    if rec is None:
        return None
    seq = rec.seq[max(0, lo - flank - 1):hi + flank]
    return seq.reverse_complement() if rev else seq


def label_event(u_gap, u_flank, u_out):
    """Return ([cls_gap, cls_flank, cls_out], event) from unique-base counts."""
    vals = [u_gap, u_flank, u_out]
    mx = max(vals)
    cls = ["L" if v == mx else "S" for v in vals]
    if len(set(vals)) == 3:
        mid = sorted(set(vals))[1]
        cls = ["M" if v == mid else c for v, c in zip(vals, cls)]
    u1, u2, u3 = vals
    ev = None
    if cls == ["L", "S", "L"]:
        ev = "Deletion"
    elif cls == ["L", "S", "S"]:
        ev = "Insertion"
    elif cls == ["L", "S", "M"] and abs(u1 - u3) <= 3:
        ev = "Deletion"
    elif cls == ["L", "S", "M"] and abs(u2 - u3) <= 3:
        ev = "Insertion"
    elif cls == ["M", "S", "L"] and abs(u1 - u3) <= 3:
        ev = "Deletion"
    elif cls == ["S", "M", "L"]:
        ev = "*Deletion*"
    elif cls == ["M", "S", "L"]:
        ev = "*Deletion*"
    elif cls == ["L", "S", "M"] and u1 and abs(u3 / u1) >= 0.9:
        ev = "*Deletion*"
    elif cls == ["L", "M", "S"]:
        ev = "*Insertion*"
    elif cls == ["L", "S", "M"]:
        ev = "*Insertion*"
    return cls, ev


def classify_three_way(corr, flanks, outg, n_left):
    """Step 16 on an aligned (corr, flanks, outgroup) triplet."""
    count, gs = 0, -1
    for i, c in enumerate(flanks):
        if c.isalpha():
            count += 1
        if count == n_left:
            gs = i
            break
    if gs == -1:
        return None
    ge = flanks.find("-", gs)
    while ge != -1 and ge < len(flanks) and flanks[ge] == "-":
        ge += 1
    if ge == -1:
        return None
    if gs > 0:
        gs += 1
    before = after = 0
    i = gs - 1
    while i >= 0 and outg[i] == "-":
        before += 1
        i -= 1
    i = ge
    while i < len(outg) and outg[i] == "-":
        after += 1
        i += 1
    g1, g2, g3 = corr[gs:ge], flanks[gs:ge], outg[gs:ge]
    u1, u2, u3 = (len(x.replace("-", "")) for x in (g1, g2, g3))
    if before > 0:
        u3 -= before
    elif after > 0:
        u3 -= after
    cls, ev = label_event(u1, u2, u3)
    return dict(Gap_Length=len(g1), Flanks_Length=len(g2), Outgroup_Length=len(g3),
                Gap_Unique_Counts=u1, Flanks_Unique_Counts=u2, Outgroup_Unique_Counts=u3,
                Gap=cls[0], Flanks=cls[1], Outgroup=cls[2], Event=ev,
                Gap_Extension_Before=before, Gap_Extension_After=after, _gs=gs, _ge=ge)


# ---------------------------------------------------------------------------
# Steps 17-21: insertions -- search for the source of the inserted sequence in its own genome
# ---------------------------------------------------------------------------

INS_BLAST = dict(word_size="7", reward="1", penalty="-3", gapopen="5", gapextend="2")


def blast_hits(query, db_prefix, p: Params, workdir, **opt):
    """All BLAST hits (outfmt 6 rows as lists of str), dust off, notebook step 17/19 settings."""
    o = dict(INS_BLAST, **opt)
    with tempfile.TemporaryDirectory(dir=workdir) as td:
        q = os.path.join(td, "q.fna")
        with open(q, "w") as f:
            f.write(f">query\n{query}\n")
        res = subprocess.run(
            [p.blastn_bin, "-query", q, "-db", db_prefix, "-dust", "no", "-word_size", o["word_size"],
             "-gapopen", o["gapopen"], "-gapextend", o["gapextend"], "-reward", o["reward"],
             "-penalty", o["penalty"], "-num_threads", "1", "-outfmt", BLAST_FMT],
            check=True, capture_output=True, text=True)
    return [ln.split("\t") for ln in res.stdout.splitlines() if ln.strip()]


def second_valid_hit(hits):
    """Steps 18/20. Hits sorted by e-value; a hit is valid if it covers the query (length >= qlen-5)
    with identity > 90%. Returns the second valid hit (a further copy of the query) or None."""
    rows = sorted(hits, key=lambda h: float(h[11]))  # stable: ties keep BLAST order
    n = 0
    for h in rows:
        if int(h[3]) >= int(h[4]) - 5 and float(h[2]) > 90:
            n += 1
            if n == 2:
                first = rows[0]
                return dict(query_length=int(h[4]), alignment_length=int(h[3]), percent_identity=float(h[2]),
                            contig=h[1], start_position=int(h[9]), end_position=int(h[10]),
                            first_subject_contig=first[1], first_subject_start_position=int(first[9]),
                            first_subject_end_position=int(first[10]),
                            second_hit_on_first_hit_contig=(h[1] == first[1]),
                            total_hits_per_file=max(0, len(rows) - 1))
    return None


def analyse_insertion(corr3, flanks3, gs, ge, genome_num, p: Params, workdir, dbs, genomes, msa_pair, gid):
    """Steps 17-21 for one outgroup-confirmed insertion.

    corr3: the corresponding-genome sequence in the 3-way alignment (it carries the insertion);
    the insertion is searched in the genome it comes from (genome_num). Returns (record, mafft_text).
    """
    def clean(x):
        return x.replace("-", "")

    ins = clean(corr3[gs:ge])
    rec = dict(genome=genome_num, insertion_length=len(ins))
    if len(ins) < 7:
        return rec, None
    db = dbs[genome_num]

    def flanked(fl):
        return clean(corr3[max(0, gs - fl):min(len(corr3), ge + fl)])

    h_gap = blast_hits(ins, db, p, workdir)
    h50 = blast_hits(flanked(50), db, p, workdir)
    h10 = blast_hits(flanked(10), db, p, workdir)
    r_gap, r50, r10 = second_valid_hit(h_gap), second_valid_hit(h50), second_valid_hit(h10)
    rec.update(source_copy_found=r_gap is not None,
               recombination_flank50=r50 is not None,   # step 18: insertion + 50 bp flanks occurs twice
               mh_blast_flank10=r10 is not None)        # step 20: insertion + 10 bp flanks occurs twice
    rec["reports"] = {"gap": r_gap, "flank50": r50, "flank10": r10}
    if r_gap is None:
        return rec, None
    # step 21: add the source copy (+/- 15 bp) to the 2-genome alignment of the gap region
    g = genomes[genome_num]
    if r_gap["contig"] not in g:
        return rec, None
    a, b = r_gap["start_position"], r_gap["end_position"]
    lo, hi = (b, a) if a > b else (a, b)
    seq = g[r_gap["contig"]].seq[max(0, lo - 1 - 15):hi + 15]
    if a > b:
        seq = seq.reverse_complement()
    rec["source_contig"], rec["source_start"], rec["source_end"] = r_gap["contig"], a, b
    rec["source_percent_identity"] = r_gap["percent_identity"]
    with tempfile.TemporaryDirectory(dir=workdir) as td:
        msa, add = os.path.join(td, "msa.fna"), os.path.join(td, "add.fna")
        with open(msa, "w") as f:
            for n, sq in msa_pair:
                f.write(f">{n}\n{sq}\n")
        with open(add, "w") as f:
            f.write(f">{gid}_source_{r_gap['contig']}_{a}_{b}\n{seq}\n")
        res = subprocess.run([p.mafft_bin, "--auto", "--thread", "1", "--add", add, msa],
                             capture_output=True, text=True, check=True)
    return rec, res.stdout


# ---------------------------------------------------------------------------
# One candidate gap through steps 9-16 (runs in a worker process)
# ---------------------------------------------------------------------------

_G = {}  # per-process globals: params, blast db, outgroup genome


def _init_worker(p, db_prefix, outgroup_fa, workdir, genome_dbs=None, genome_fas=None):
    _G["p"], _G["db"], _G["workdir"] = p, db_prefix, workdir
    _G["gdbs"] = genome_dbs or {}
    _G["genomes"] = {k: SeqIO.index(v, "fasta") for k, v in (genome_fas or {}).items()}
    _G["og"] = SeqIO.to_dict(SeqIO.parse(outgroup_fa, "fasta")) if outgroup_fa else None


def process_gap(task):
    """task: dict from extract_gaps plus 'gid'. Returns dict with stage reached and results."""
    p, workdir = _G["p"], _G["workdir"]
    gid = task["gid"]
    out = {"gid": gid, "stage": "extracted"}
    try:
        gnum = task["seq_id"].split(":")[0]
        onum = task["other_id"].split(":")[0]
        n1, n2 = f"Sequence_{gnum}", f"Corresponding_Sequence_{onum}"

        aln = run_mafft([(n1, task["region"]), (n2, task["other_region"])], p, workdir=workdir)
        if len(aln) < 2:
            out["stage"] = "mafft_failed"
            return out
        seq, corr = aln[0][1], aln[1][1]
        out["stage"] = "mafft"

        region = gap_region_in_corresponding(seq, corr, p.min_gap)
        if region is not None and has_ssr(region, p.ssr_threshold):
            out["stage"] = "removed_ssr"
            return out
        if not passes_identity_filter(seq, corr, p.min_gap):
            out["stage"] = "removed_identity_or_N"
            return out
        out["stage"] = "filtered"

        m = messer_classify(seq, corr, p)
        if m is None:
            out["stage"] = "no_messer_result"
            return out
        out["messer"] = m
        out["stage"] = "messer"
        if m["cls"] not in ("b", "d"):
            return out
        if _G["db"] is None:
            return out

        gaps = [i for i, c in enumerate(seq) if c == "-"]
        gs, ge = gaps[0], gaps[-1]
        F = p.outgroup_flank
        left_fl = seq[max(0, gs - F):gs]
        flanks = left_fl + seq[ge + 1:ge + 1 + F]
        corr_seq = (corr[max(0, gs - F):gs] + corr[gs:ge + 1] + corr[ge + 1:ge + 1 + F]).replace("-", "N")
        nleft = len(left_fl)
        qf = f">{gid}_{n1}_{nleft}_query_flanks\n{flanks}\n"
        qc = f">{gid}_{n2}_{len(corr[max(0, gs - F):gs])}_query_corr\n{corr_seq}\n"

        hit = _blast_first_hit(qc, _G["db"], p, workdir)
        if hit is None or len(hit) < 11:
            out["stage"] = "no_outgroup_hit"
            return out
        og_seq = outgroup_sequence(hit, _G["og"], F)
        if og_seq is None:
            out["stage"] = "no_outgroup_hit"
            return out

        prof = run_mafft([(f"{gid}_{n2}_corr", corr_seq), (f"{gid}_{n1}_flanks", flanks)], p, workdir=workdir)
        with tempfile.TemporaryDirectory(dir=workdir) as td:
            profile_fa, add_fa = os.path.join(td, "profile.fna"), os.path.join(td, "add.fna")
            with open(profile_fa, "w") as f:
                for n, s in prof:
                    f.write(f">{n}\n{s}\n")
            with open(add_fa, "w") as f:
                f.write(f">blast_sequence\n{og_seq}\n")
            res = subprocess.run([p.mafft_bin, "--auto", "--thread", "1", "--addprofile", add_fa, profile_fa],
                                 capture_output=True, text=True, check=True)
        recs = list(SeqIO.parse(StringIO(res.stdout), "fasta"))
        if len(recs) < 3:
            out["stage"] = "no_outgroup_hit"
            return out
        cls = classify_three_way(str(recs[0].seq), str(recs[1].seq), str(recs[2].seq), nleft)
        if cls is None:
            out["stage"] = "no_event_call"
            return out
        out["event"] = {k: v for k, v in cls.items() if not k.startswith("_")}
        out["alignment"] = res.stdout
        out["stage"] = "event"
        if cls["Event"] == "Insertion" and _G["gdbs"] and int(onum) in _G["gdbs"]:
            rec, text = analyse_insertion(str(recs[0].seq), str(recs[1].seq), cls["_gs"], cls["_ge"], int(onum),
                                          p, workdir, _G["gdbs"], _G["genomes"], aln[:2], gid)
            out["insertion"] = rec
            out["ins_alignment"] = text
        return out
    except Exception as e:  # keep the run alive; report per gap
        out["stage"] = "error"
        out["error"] = f"{type(e).__name__}: {e}"
        return out


# ---------------------------------------------------------------------------
# Pair driver
# ---------------------------------------------------------------------------

def run_mauve(g1, g2, xmfa, mauve_bin="progressiveMauve"):
    subprocess.run([mauve_bin, f"--output={xmfa}", g1, g2], check=True)


def run_pair(xmfa, outgroup_fa, outdir, p: Params, keep_alignments=True, log=print, genomes=None):
    """Run steps 1-16 for one genome pair. Returns (summary dict, messer_df, event_df)."""
    os.makedirs(outdir, exist_ok=True)
    work = os.path.join(outdir, "tmp")
    os.makedirs(work, exist_ok=True)

    blocks = parse_xmfa(xmfa)
    selected = select_blocks(blocks, p)
    log(f"  {len(blocks)} XMFA blocks, {len(selected)} contain a gap >= {p.min_gap} bp")

    tasks = []
    for n, blk in enumerate(selected, start=1):
        for k, g in enumerate(extract_gaps(blk, p), start=1):
            g["gid"] = f"block_{n}_gap_results_{k}"
            tasks.append(g)
    log(f"  {len(tasks)} candidate gaps after N / identity filters")

    db_prefix = None
    if outgroup_fa:
        db_prefix = os.path.join(work, "outgroup_db")
        make_blast_db(outgroup_fa, db_prefix, p)

    genome_dbs = {}
    if genomes:
        for num, path in genomes.items():
            genome_dbs[num] = os.path.join(work, f"genome{num}_db")
            make_blast_db(path, genome_dbs[num], p)
    initargs = (p, db_prefix, outgroup_fa, work, genome_dbs, genomes or {})
    if p.threads > 1:
        with Pool(p.threads, initializer=_init_worker, initargs=initargs) as pool:
            results = pool.map(process_gap, tasks, chunksize=4)
    else:
        _init_worker(*initargs)
        results = [process_gap(t) for t in tasks]

    funnel = pd.Series([r["stage"] for r in results]).value_counts().to_dict()
    errors = [(r["gid"], r["error"]) for r in results if r["stage"] == "error"]

    mrows = []
    for r in results:
        if "messer" in r:
            m = r["messer"]
            mrows.append({"File Name": r["gid"], "Right Rep": m["right_rep"], "Left Rep": m["left_rep"],
                          "l": m["l"], "d": m["d"], "l-d": m["l_minus_d"], "Classification": m["cls"]})
    mdf = pd.DataFrame(mrows, columns=["File Name", "Right Rep", "Left Rep", "l", "d", "l-d", "Classification"])

    erows = []
    aln_dir = os.path.join(outdir, "alignments")
    if keep_alignments:
        os.makedirs(aln_dir, exist_ok=True)
    for r in results:
        if "event" in r:
            row = {"File": f"{r['gid']}_query_corr_mafft_addprofile_output.fna"}
            row.update(r["event"])
            erows.append(row)
            if keep_alignments:
                with open(os.path.join(aln_dir, row["File"]), "w") as f:
                    f.write(r["alignment"])
    edf = pd.DataFrame(erows)

    ins_df = write_insertion_outputs(results, outdir, mdf)
    shutil.rmtree(work, ignore_errors=True)
    mdf.to_csv(os.path.join(outdir, "messer_results.csv"), index=False)
    edf.to_csv(os.path.join(outdir, "class_event.csv"), index=False)

    events = merge_events(mdf, edf)
    events.to_csv(os.path.join(outdir, "events.csv"), index=False)

    summary = {"n_xmfa_blocks": len(blocks), "n_blocks_with_gap": len(selected),
               "n_candidate_gaps": len(tasks), **{f"stage_{k}": v for k, v in funnel.items()}}
    if len(mdf):
        for c, v in mdf["Classification"].value_counts().items():
            summary[f"messer_{c}"] = int(v)
    if len(events):
        for c, v in events["Event"].fillna("unassigned").value_counts().items():
            summary[f"event_{c}"] = int(v)
        dels = events[events["Event"] == "Deletion"]
        summary["n_confirmed_deletions"] = len(dels)
        if len(dels):
            summary["pct_ge_10bp_mh"] = round((dels["d"] >= 10).mean() * 100, 1)
            summary["mean_mh"] = round(dels["d"].mean(), 2)
    if ins_df is not None:
        summary["n_insertions_analysed"] = len(ins_df)
        summary["n_insertions_source_copy"] = int(ins_df["source_copy_found"].sum())
        summary["n_insertions_recombination_flank50"] = int(ins_df["recombination_flank50"].sum())
        summary["n_insertions_final_population"] = int(ins_df["final_population"].sum())
    summary["n_errors"] = len(errors)
    if errors:
        with open(os.path.join(outdir, "errors.txt"), "w") as f:
            for g, e in errors:
                f.write(f"{g}\t{e}\n")
    return summary, mdf, edf


def write_insertion_outputs(results, outdir, mdf):
    """insertions.csv, rec_blast_summary.csv, mh_blast_summary.csv and mh_alignments/ (steps 17-21)."""
    ins = [r for r in results if "insertion" in r]
    if not any("insertion" in r for r in results):
        return None
    rows, rec_rows, mh_rows = [], [], []
    mh_dir = os.path.join(outdir, "mh_alignments")
    os.makedirs(mh_dir, exist_ok=True)
    cols = ["query_length", "alignment_length", "percent_identity", "contig", "start_position", "end_position",
            "first_subject_contig", "first_subject_start_position", "first_subject_end_position",
            "second_hit_on_first_hit_contig", "total_hits_per_file"]
    for r in ins:
        i = r["insertion"]
        row = {"Block_Gap": r["gid"], **{k: v for k, v in i.items() if k != "reports"}}
        row["final_population"] = not i.get("recombination_flank50", False)
        rows.append(row)
        rep = i.get("reports", {})
        for key, target, ftype in (("gap", rec_rows, "gap"), ("flank50", rec_rows, "flanks"),
                                   ("gap", mh_rows, "gap"), ("flank10", mh_rows, "flanks")):
            if rep.get(key):
                target.append({"file_name": f"{r['gid']}_blast_result_{ftype}.txt", "file_type": ftype,
                               **{c: rep[key][c] for c in cols}})
        if r.get("ins_alignment"):
            with open(os.path.join(mh_dir, f"{r['gid']}_updated.fna"), "w") as f:
                f.write(r["ins_alignment"])
    df = pd.DataFrame(rows)
    d = mdf.rename(columns={"File Name": "Block_Gap", "Classification": "Messer_class"})
    df = df.merge(d[["Block_Gap", "Messer_class", "l", "d"]], on="Block_Gap", how="left")
    df.to_csv(os.path.join(outdir, "insertions.csv"), index=False)
    pd.DataFrame(rec_rows).to_csv(os.path.join(outdir, "rec_blast_summary.csv"), index=False)
    pd.DataFrame(mh_rows).to_csv(os.path.join(outdir, "mh_blast_summary.csv"), index=False)
    return df


def merge_events(mdf, edf):
    """Event table restricted to Messer b/d INDELs with their l, d and class.

    Deletion microhomology equals d. For insertions d is reported but is not the
    microhomology (see README).
    """
    if edf is None or not len(edf):
        return pd.DataFrame(columns=["Block_Gap", "Event", "Messer_class", "l", "d"])
    e = edf.copy()
    e["Block_Gap"] = e["File"].str.replace("_query_corr_mafft_addprofile_output.fna", "", regex=False)
    m = mdf.rename(columns={"File Name": "Block_Gap", "Classification": "Messer_class"})
    out = e.merge(m[["Block_Gap", "Messer_class", "l", "d"]], on="Block_Gap", how="left")
    cols = ["Block_Gap", "Event", "Messer_class", "l", "d"] + [c for c in e.columns if c not in ("File", "Block_Gap", "Event")]
    return out[cols]
