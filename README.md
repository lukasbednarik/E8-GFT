# E8-GFT

Numerical verification scripts and supporting code for a series of
working papers on a group field theory of the compact real form of the
exceptional Lie group $E_8$.

## Repository layout

- `e8sim/` — shared library (root system of $E_8$, EIX coset
  geometry, invariants, Dirac operator, PDG constants). Imported by
  every numbered folder below.
- `NN-<topic>/` — one folder per paper. Each folder is self-contained
  and contains the scripts, notes, and verification protocols cited
  in the corresponding manuscript (e.g. `01-triality-test/`,
  `10-cc-tests/`).

The numbering of `NN-<topic>/` folders matches the numbering of the
papers, not chronological order; gaps are intentional.

## Papers

| # | Title | DOI |
|---|---|---|
| 01 | Foundations of an $E_8$ group field theory: action uniqueness, vacuum selection, and a four-dimensional algebraic substrate | [10.5281/zenodo.19843240](https://doi.org/10.5281/zenodo.19843240) |
| 02 | Effective sigma model of an $E_8$ group field theory: kinetic uniqueness, the Skyrme sector, and topological terms on the Wolf space EIX | [10.5281/zenodo.19843827](https://doi.org/10.5281/zenodo.19843827) |
| 10 | Notes on the cosmological constant in $E_8$ group field theory | [10.5281/zenodo.19844636](https://doi.org/10.5281/zenodo.19844636) |

## License

All code in this repository is released under the MIT License — see
`LICENSE`. You are free to use, modify, and redistribute it; the only
requirement is that the copyright notice and license text be retained
in derivative works.

## Contact

Lukáš Bednařík — `lukas@lukasbednarik.cz`
ORCID: [0009-0008-1839-688X](https://orcid.org/0009-0008-1839-688X)
