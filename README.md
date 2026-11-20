# NHEJScan pipeline

Pairwise-genome INDEL detection and end-joining signature analysis, as used in
"Loss of Canonical Nonhomologous End-Joining Shapes Genomic INDEL Signatures in Fungi".

```
genome A + genome B --progressiveMauve--> XMFA
  -> gaps 10-1000 bp with 5x flanks, N and identity filters
  -> MAFFT realignment, SSR filter, identity/N filter
  -> Messer & Arndt trace extension: l, d, class a/b/c/d
  -> b/d events: BLAST against the outgroup genome,
     3-way MAFFT, L/M/S call: Deletion / Insertion
     (outgroup-confirmed) or *Deletion* / *Insertion*
  -> clade x length-class counts --> weighted scoring model
```

## Install

Programs required on PATH: `progressiveMauve`, `mafft`, `blastn`, `makeblastdb`, and a C++ compiler
(the bundled `trace_ext` is compiled on first use into `~/.cache/nhejscan`).

```bash
conda env create -f environment.yml && conda activate nhejscan   # or: apt install mafft ncbi-blast+ progressivemauve g++
pip install .
pytest                       # 16 tests, about 20 s; includes a simulated end-to-end run
```
A `Dockerfile` is provided as well.

## Use

One pair, from genomes (runs Mauve) or from an existing XMFA:
```bash
nhejscan pair --genome1 A.fna --genome2 B.fna --outgroup O.fna --outdir out/A__B
nhejscan pair --xmfa A_B.xmfa --genome1 A.fna --genome2 B.fna --outgroup O.fna --outdir out/A__B   # genomes needed for the insertion analysis
```
All pairs of a tree that meet the distance cut-off (pair and outgroup both within the cut-off),
for every pair whose three genomes are in `--genome-dir`; finished pairs are skipped on re-run:
```bash
nhejscan candidates --tree tree.nwk --clade-map species_clade_map.csv --max-distance 0.5747 --out candidates.csv
nhejscan screen --tree tree.nwk --clade-map species_clade_map.csv --genome-dir genomes/ \
                --accession-map accession_map.csv --max-distance 0.5747 --outdir runs/ --summary screen_summary.csv
```
Scoring model (`indel_mh_counts.csv`: clade, INS_0_1, INS_2_4, INS_5_9, INS_10_plus, DEL_0_1, ...):
```bash
nhejscan model --counts examples/indel_mh_counts_example.csv --outdir model_out --seed 42
```
`examples/run_simulated_demo.sh` runs everything on a simulated triplet in about a minute.

## Output per pair (`--outdir`)

| file | content |
|---|---|
| `messer_results.csv` | `File Name, Right Rep, Left Rep, l, d, l-d, Classification` (a/b/c/d) |
| `class_event.csv` | outgroup polarization: unique-base counts, L/M/S calls, `Event`|
| `events.csv` | class_event joined to l, d and Messer class. Deletion microhomology = `d` |
| `alignments/` | the 3-way alignment of every polarized event |
| `summary.json` | counts at every filter stage, Messer classes, event labels |

`l` is the gap length, `d` the trace extension (repeat length). Only Messer classes b and d are
end-joining events; a and c are reported but are not polarized.

## Insertions

Needs the two genome FASTAs (`--genome1/--genome2`, same order as the Mauve run; with `--xmfa` they are
optional but without them this part is skipped). For every outgroup-confirmed `Insertion`
(`*Insertion*` is not processed), the inserted sequence is searched in the genome it comes
from (BLAST: dust off, word size 7, match +1, mismatch -3, gap 5/2):

| output | content |
|---|---|
| `rec_blast_summary.csv` | second full-length (length >= query-5, >90% identity) hit of the insertion alone and of insertion + 50 bp flanks |
| `mh_blast_summary.csv` | same for insertion alone and insertion + 10 bp flanks |
| `mh_alignments/*_updated.fna` | source copy (+/-15 bp) added by MAFFT `--add` to the 2-genome alignment of the gap, ready for the manual microhomology reading |
| `insertions.csv` | one row per insertion with the flags above, `l`, `d`, Messer class |

`insertions.csv` columns: `source_copy_found` (insertion alone has a second copy), `recombination_flank50`
(insertion with 50 bp flanks occurs twice), `mh_blast_flank10`, `final_population` (= not `recombination_flank50`).
For insertions `d` is the trace extension, not the microhomology; the microhomology is read manually from `mh_alignments`.
Short insertions (about 15 bp or less) can pass the "second hit" test by chance, because the criterion tolerates 5 missing bases.

## Reproducibility notes

Results depend on the versions of progressiveMauve, MAFFT and BLAST+; record them with the outputs
(`mafft --version`, `blastn -version`). Tested on: MAFFT v7.505, BLAST+ 2.12.0, progressiveMauve (Ubuntu 24.04 build, 2024-04-08).
