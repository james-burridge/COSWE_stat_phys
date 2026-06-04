# Statistical Physics of Language Change Inferred from Time-Evolving Maps

Code and data for the paper:

> **Statistical physics of language change inferred from time evolving maps**  
> James Burridge¹ and Bert Vaux²  
> ¹ School of Mathematics and Physics, University of Portsmouth  
> ² Department of Theoretical & Applied Linguistics, University of Cambridge  
> *Journal of the Royal Society Interface* (submitted)  
> Funded by the Royal Society APEX award APX\R1\241139

## Overview

This paper develops a statistical physics framework for modelling the spatial and temporal evolution of linguistic dialect variation across the USA during 1950–2000. The approach has two components:

1. **Empirical field inference** — a Bayesian Gaussian process method for reconstructing spatially- and temporally-smooth variant frequency maps (state fields) from the Cambridge Online Survey of World Englishes (COSWE).

2. **State field model** — a stochastic replicator dynamics model incorporating realistic long-range migration (fitted to IRS county-to-county data), local diffusion, frequency-dependent accommodation, and spatially-varying bias fields. Model parameters are inferred by matching predicted and empirical state field dynamics.

The six linguistic variables studied are: *soda/pop/coke*, *tennis shoes/sneakers*, *crawfish/crayfish/crawdad*, *you-guys/you-all*, *roly-poly/pill-bug*, and *sunshower/devil-beating-his-wife*.

## Repository structure

```
REPO/
├── code/
│   ├── notebooks/      Section notebooks (executable, self-contained)
│   └── src/            Python source modules imported by the notebooks
├── data/
│   ├── voronois/       4000-cell Voronoi tessellation of mainland USA
│   ├── demed_variants/ Survey response counts per deme per year (CSVs, Q2–Q7)
│   └── probability_fields/  Inferred MAP state fields (.npy, 4000 demes × 51 years)
└── figures/            Key figures from the paper
```

## Code

### `code/src/`

| File | Description |
|------|-------------|
| `GP_fitting.py` | MAP estimation for multinomial GP regression — softmax latent field model fitted by L-BFGS-B |
| `model_inference.py` | Constructs the spatial basis (lifting matrix) from eigenvectors of the degree-normalised Gram matrix at length scale η |
| `spatial_model.py` | Migration and copying generators; replicator dynamics model class; parameter inference objectives; KL divergence |

### `code/notebooks/`

| File | Paper section | Description |
|------|--------------|-------------|
| `section2_empirical_field_inference.ipynb` | §2 | Demonstrates GP state field inference for one variable; loads and displays pre-computed fields for all six |
| `section3_state_field_model.ipynb` | §3 | Builds migration and copying generators; generates Figures 5 and 6 |
| `section4_model_inference.ipynb` | §4 | Infers β and bias fields; generates Figures 7–11 |
| `additional_checks.ipynb` | — | Additional model checks: in-sample KL divergence as a function of bias field length scale η |

## Data

### Survey data

The raw COSWE survey data are not publicly available. They may be obtained from Bert Vaux (University of Cambridge) upon reasonable request; see the [COSWE project page](https://www.tekstlab.uio.no/cambridge_survey/). Once obtained, the data are processed into the `data/demed_variants/` CSVs using the scripts in the source repository.

### `data/voronois/`

| File | Description |
|------|-------------|
| `Voronoi_4000_raw_kmeans_demes.gpkg` | GeoPackage of the 4000-cell Voronoi tessellation used throughout the paper. Cells are seeded by centroids of k-means clusters of mainland US ZIP codes weighted by 2020 Census population |
| `zip_codes.csv` | ZIP code coordinates and populations used to construct the tessellation |

### `data/demed_variants/`

Variant response counts aggregated to 4000 Voronoi cells, one CSV per variable:

| File prefix | Variable |
|-------------|----------|
| `Q2_soda_pop_coke` | Sweetened carbonated beverage |
| `Q3_tennis_sneakers` | Rubber-soled athletic shoes |
| `Q4_crawfish_crayfish_crawdad` | Freshwater crustacean |
| `Q5_you-guys_you-all` | Plural second-person pronoun |
| `Q6_roly_bug` | Pill bug / roly-poly |
| `Q7_sunshower_devil` | Rain while the sun shines |

Each CSV has columns `deme_id`, `year`, and one column per variant, containing response counts for respondents born in each year 1950–2000.

### `data/probability_fields/`

Inferred MAP state field arrays for the six paper variables, stored as NumPy `.npy` files with shape `(N, T, K)` where N = 4000 demes, T = 51 years (1950–2000), K = number of variants.

File names encode the fitted GP hyperparameters: `v_4_<variable>_4000_demes_tau_<τ>_sigma_<σ>_alpha_<α>_var_<κ>_start_1950_end_2000.npy`.

## Model summary

The state field **X**ᵢ(t) ∈ Δ^{K−1} gives the vector of variant frequencies in deme i at time t. Its expected increment is (eq. 3.1):

$$\mathbb{E}_t(\delta \mathbf{X}_i) = \left(\mathbf{q}_i \circ \mathbf{X}_i - \bar{q}_i \mathbf{X}_i + \sum_j \left(w_{ij} + Jl_{ij} - (w_i + J)\delta_{ij}\right)\mathbf{X}_j\right)\delta t$$

where:
- **q**ᵢ = **s**ᵢ + β**X**ᵢ is the fitness vector (bias **s**ᵢ plus accommodation term β**X**ᵢ)
- w_{ij} is the migration rate from j to i, following a gravity model with parameters d₀ = 48.9 km, γ₀ = 5.57, γ₁ = −0.35, α = 0.60
- l_{ij} is the local copying kernel, $l_{ij} \propto P_j\exp(-d_{ij}^2/2R^2)$, with R = 100 km and copying rate J = 0.1; effective diffusion coefficient D = JR²/2
- **s**ᵢ = A**Ψ** is the spatially-varying bias field, expressed in a reduced basis A at length scale η

Parameters β (accommodation), **Ψ** (bias field), and η (bias field complexity) are inferred by minimising residuals between empirical and model state field increments.

## Requirements

```
numpy
scipy
matplotlib
geopandas
pandas
scikit-learn
shapely
optuna
jupyter
```

Install with:

```bash
pip install numpy scipy matplotlib geopandas pandas scikit-learn shapely optuna jupyter
```

## Citation

If you use this code or data, please cite:

```
Burridge, J. and Vaux, B. (2026). Statistical physics of language change inferred 
from time evolving maps. Journal of the Royal Society Interface.
```
