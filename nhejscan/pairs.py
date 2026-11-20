"""Candidate-pair enumeration from a species tree and genome-file lookup."""
import glob
import os
import re
from itertools import combinations

import pandas as pd
from Bio import Phylo


def find_nearest_outgroup(tree, s1, s2, max_levels=6):
    """Nearest tip outside the (s1, s2) clade, walking up the tree."""
    mrca = tree.common_ancestor([s1, s2])
    path = tree.get_path(mrca)
    for level in range(2, max_levels + 1):
        if len(path) < level:
            return None, None
        parent = path[-level]
        exclude = path[-level + 1]
        siblings = [c for c in parent.clades if c != exclude]
        cands = [t.name for sib in siblings for t in sib.get_terminals() if t.name not in (s1, s2)]
        if cands:
            best = min(cands, key=lambda c: tree.distance(s1, c))
            return best, tree.distance(s1, best)
    return None, None


def enumerate_candidates(tree, groups, max_distance, exclude_pairs=None):
    """groups: {clade_name: [species,...]}. Pairs and their outgroup must be within max_distance."""
    exclude_pairs = exclude_pairs or set()
    tips = {t.name for t in tree.get_terminals()}
    rows = []
    for clade, species in groups.items():
        cands = [s for s in species if s in tips]
        for s1, s2 in combinations(cands, 2):
            if tuple(sorted((s1, s2))) in exclude_pairs:
                continue
            pd_ = tree.distance(s1, s2)
            if pd_ > max_distance:
                continue
            og, ogd = find_nearest_outgroup(tree, s1, s2)
            if og is None or ogd > max_distance:
                continue
            rows.append(dict(clade=clade, species1=s1, species2=s2, pair_distance=round(pd_, 4),
                             outgroup=og, outgroup_distance=round(ogd, 4),
                             sum_distance=round(pd_ + ogd, 4)))
    return sorted(rows, key=lambda r: r["pair_distance"])


def groups_from_clade_map(df):
    """Clade groups used in the study (columns: species, subphylum, phylum, class)."""
    g = {
        "Zoopagomycota": df[df["phylum"] == "Zoopagomycota"]["species"].tolist(),
        "Mucoromycota": df[df["phylum"] == "Mucoromycota"]["species"].tolist(),
        "Ustilaginomycotina": df[df["subphylum"] == "Ustilaginomycotina"]["species"].tolist(),
    }
    if "class" in df.columns:
        g["Agaricomycetes"] = df[df["class"] == "Agaricomycetes"]["species"].tolist()
    return g


def load_exclude_pairs(path):
    pairs = set()
    if path and os.path.exists(path):
        for line in open(path):
            line = line.strip()
            if line:
                a, b = line.split(",")
                pairs.add(tuple(sorted((a.strip(), b.strip()))))
    return pairs


def load_accession_map(path):
    if not path or not os.path.exists(path):
        return {}
    df = pd.read_csv(path)
    return {r["species"]: re.sub(r"\.\d+$", "", str(r["accession"]).strip()) for _, r in df.iterrows()}


def find_genome_file(genome_dir, species, accession_map=None):
    exts = (".fna", ".fa", ".fasta", ".fna.gz")
    if accession_map and species in accession_map:
        hits = [c for c in glob.glob(os.path.join(genome_dir, f"*{accession_map[species]}*")) if c.endswith(exts)]
        if hits:
            return sorted(hits)[0]
    hits = [m for m in glob.glob(os.path.join(genome_dir, f"{species}*")) if m.endswith(exts)]
    return sorted(hits)[0] if hits else None


def read_tree(path):
    return Phylo.read(path, "newick")
