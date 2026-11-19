# Weyl-midpoint semiclassics for Wigner 3j and 6j symbols: supplement and verification programs

**Repository:** <https://github.com/cmbant/weyl-midpoint-semiclassics> · **Compiled PDF:** [Supplement](supplement/joint_rank_position_uniformity.pdf) · **Paper:** arXiv {{ARXIV}}

This repository accompanies the paper *Semiclassical Asymptotics of Wigner 3j and 6j Symbols at Growing Rank: Phase Horizons and
Turning Connections* by Antony Lewis (arXiv: {{ARXIV}}), the sequel to
*Even Asymptotic Expansions and Uniform Hard-Edge Approximations for Wigner 3j and 6j Symbols at Fixed Rank* (the fixed-rank companion paper, Hexagon 2611.00533v1, <https://hexagonmath.org/2611.00533v1>). It holds

* the technical supplement *Joint Rank–Position Uniformity for Weyl–Midpoint Wigner 3j Symbols*, which proves, for the
  fixed-offset Hahn/3j chain, a natural-gauge error bound `O([N(1-|x|)]^-2)` at every physical position and every rank below a
  fixed turning ratio (Theorem 5.4), with Racah/6j extensions; no theorem of the paper depends on it;
* the verification programs cited by the paper and by the supplement, with their captured output.

The paper itself is on arXiv and is not part of this repository. The supplement's references to the fixed-rank paper and the main paper (theorem,
equation and lemma numbers) are fixed to the numbers of their published versions (Hexagon for the fixed-rank paper, arXiv for the main paper), so the supplement compiles on its own.

## Contents

| folder | contents |
|---|---|
| `supplement/` | `joint_rank_position_uniformity.tex` and its compiled PDF |
| `verifiers/` | the verification programs: Python scripts, their captured output in `logs/`, `MAP.md` (script, log, and the labels of the paper and of the supplement that name it) and `requirements.txt` |

## Claim classes

Statements in the paper and the supplement are proved unless they are called a conjecture or numerical evidence. The
programs check exact symbolic identities used in proofs, compare the asymptotic formulas with exact symbols, and run
numerical benchmarks and negative controls. A program backs a proof only where the text says that a step was verified by
exact symbolic computation; numerical comparisons illustrate accuracy and are never part of a proof, and no finite test
certifies the universal constants of the supplement's theorems.

## Running the verifiers

Python 3.10 or later.

```bash
python3 -m pip install -r verifiers/requirements.txt
cd verifiers
python3 verify_joint_3j_symbolic_exact.py      # about a minute; exits 0 and prints the same as logs/verify_joint_3j_symbolic_exact.txt
```

Every `verify_*` script prints its checks and exits with status 0 if all pass; its captured output, where one was captured,
is `logs/<script>.txt` (runtimes and floating-point details may differ in the last digits). `MAP.md` lists, for each
script, the labels of the paper and of the supplement that cite it.

## Building the PDF

```bash
cd supplement
pdflatex joint_rank_position_uniformity.tex && pdflatex joint_rank_position_uniformity.tex
```

The supplement uses only standard LaTeX packages.

## Citation

If you use this work, please cite the paper and this repository:

```bibtex
@article{Lewis_weyl_midpoint_semiclassics,
  author  = {Lewis, Antony},
  title   = {Semiclassical Asymptotics of Wigner 3j and 6j Symbols at Growing Rank: Phase Horizons and Turning Connections},
  eprint  = "{{ARXIV}}",
  year    = {2026}
}

@software{Lewis_weyl_midpoint_semiclassics_companion,
  author  = {Lewis, Antony},
  title   = {Supplement and verification programs for ``Semiclassical Asymptotics of Wigner 3j and 6j Symbols at Growing Rank''},
  doi     = "{{DOI}}",
  url     = "https://github.com/cmbant/weyl-midpoint-semiclassics",
  year    = {2026}
}
```

## Licence

The code in `verifiers/` is MIT-licensed and the supplement in `supplement/` is CC BY 4.0. See `LICENSE`.
