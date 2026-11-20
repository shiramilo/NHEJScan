"""Simulate a genome triplet (A, B, outgroup) with planted deletions/insertions of known
length and microhomology, to test the whole pipeline end to end."""
import json
import random


def _rand(rng, n):
    return "".join(rng.choice("ACGT") for _ in range(n))


def _snps(rng, seq, rate):
    s = list(seq)
    for i in rng.sample(range(len(s)), int(len(s) * rate)):
        s[i] = rng.choice([b for b in "ACGT" if b != s[i]])
    return "".join(s)


def simulate(outdir, seed=1, genome_len=500_000, n_contigs=4, n_events=60,
             snp_ab=0.01, snp_out=0.04, min_len=12, max_len=80, max_mh=12, dup_frac=0.4):
    rng = random.Random(seed)
    clen = genome_len // n_contigs
    anc = [_rand(rng, clen) for _ in range(n_contigs)]
    spacing = 4000
    events = []
    for ci in range(n_contigs):
        pos = 3000
        while pos < clen - 3000 - max_len - 2 * max_mh and len(events) < n_events:
            kind = rng.choice(["DEL", "INS"])
            who = rng.choice(["A", "B"])
            L = rng.randint(min_len, max_len)
            mh = rng.randint(0, max_mh)
            events.append(dict(contig=ci, pos=pos, kind=kind, genome=who, length=L, mh=mh))
            pos += spacing + rng.randint(0, 2000)
    rng.shuffle(events)
    events = events[:n_events]
    # a fraction of insertions are copies of another locus of the same contig (templated insertions)
    for e in events:
        e["dup"] = False
        if e["kind"] == "INS" and rng.random() < dup_frac:
            pos_on_contig = [x["pos"] for x in events if x["contig"] == e["contig"]]
            for _ in range(100):
                q = rng.randint(500, clen - 500 - e["length"])
                if all(abs(q - x) > 300 for x in pos_on_contig):
                    e["seq"], e["dup"] = anc[e["contig"]][q:q + e["length"]], True
                    break

    # Deletions: the ancestor carries X [MH] D [MH] Y. The lineage that deletes D and one MH
    # copy carries X [MH] Y, so the microhomology of the deletion is exactly `mh`.
    for e in events:
        if e["kind"] == "DEL":
            s = anc[e["contig"]]
            p, L, m = e["pos"], e["length"], e["mh"]
            s = s[:p + m + L] + s[p:p + m] + s[p + m + L + m:]
            s = list(s)
            # make the repeat maximal: bases flanking the two MH copies must differ
            if s[p - 1] == s[p + m + L - 1]:
                s[p - 1] = next(b for b in "ACGT" if b != s[p + m + L - 1])
            if s[p + m] == s[p + m + L + m]:
                s[p + m] = next(b for b in "ACGT" if b != s[p + m + L + m])
            anc[e["contig"]] = "".join(s)

    def build(who):
        out = []
        for ci, seq in enumerate(anc):
            evs = sorted([e for e in events if e["contig"] == ci and e["genome"] == who],
                         key=lambda e: -e["pos"])
            s = seq
            for e in evs:
                p, L, m = e["pos"], e["length"], e["mh"]
                if e["kind"] == "DEL":
                    s = s[:p + m] + s[p + m + L + m:]
                else:
                    s = s[:p] + e.get("seq", None) + s[p:] if e.get("seq") else s[:p] + _rand(rng, L) + s[p:]
            out.append(s)
        return out

    A, B = build("A"), build("B")
    O = list(anc)

    def finish(contigs, rate, tag):
        return [_snps(rng, c, rate) for c in contigs]

    A, B, O = finish(A, snp_ab, "A"), finish(B, snp_ab, "B"), finish(O, snp_out, "O")
    for name, g in (("A", A), ("B", B), ("O", O)):
        with open(f"{outdir}/{name}.fna", "w") as f:
            for i, c in enumerate(g):
                f.write(f">{name}_ctg{i+1} simulated\n")
                for j in range(0, len(c), 80):
                    f.write(c[j:j + 80] + "\n")
    json.dump(events, open(f"{outdir}/truth.json", "w"), indent=1)
    return events
