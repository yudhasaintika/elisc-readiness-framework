# ELISC: Readiness Assessment Framework for Collaborative eLeadership

Analysis code for the manuscript "A Proposed Readiness Assessment Model for
Collaborative eLeadership in Smart Cities" (Saintika, Riasetiawan, & Mustofa),
under review at the *Journal of Computational and Cognitive Engineering*.

The scripts here generate the simulated dataset and reproduce every computational
result reported in Section 5 of the manuscript and in Supplementary File S3.

## Contents

| File | Produces |
|------|----------|
| `worked_example.py` | Readiness classification, dimension scores, membership function sensitivity, algorithm ablation, SHAP feature importance |
| `sample_size_analysis.py` | Performance and stability across assessed-sample sizes |

## Requirements

```
python >= 3.10
numpy
pandas
scikit-learn
catboost
shap
scipy
```

Install with:

```bash
pip install numpy pandas scikit-learn catboost shap scipy
```

Exact versions used for the reported results:

```
numpy       x.x.x
pandas      x.x.x
scikit-learn x.x.x
catboost    x.x.x
shap        x.x.x
scipy       x.x.x
```

## Running

```bash
python worked_example.py
python sample_size_analysis.py
```

Both scripts write their output as CSV files in the working directory. Runtime for
`worked_example.py` is a few minutes; `sample_size_analysis.py` takes longer, since
it regenerates and re-evaluates the pipeline at seven sample sizes.

## Reproducibility

All results use the random seed `20260722`. With this seed, the reported values are
reproduced exactly on the versions listed above. Note that CatBoost and the SHAP
explainer may show small numerical differences across platforms and library
versions; the ordering of features and the direction of all reported comparisons are
stable.

## Simulation design

`worked_example.py` generates 300 assessment units. Four correlated dimension factors
(inter-factor correlation 0.35) generate fifteen sub-dimension latent variables with
loadings of 0.75. Each sub-dimension is observed through three indicators with
loadings drawn from U(0.65, 0.85), mapped to a five-point response scale, giving
13,500 simulated responses.

The external readiness label used by the validation layer is generated from the
error-free latent scores through a function structurally different from the linear
readiness index, in order to satisfy the independence requirement stated in Section
4.4 of the manuscript. The label is not a deterministic function of the framework's
own features.

## A note on the data

The data are simulated. They demonstrate that the pipeline composes correctly and
show the consequences of its design choices; they carry no evidence about
eLeadership readiness in any real setting. Field validation is described as future
work in the manuscript.

## Citation

Citation details will be added once the manuscript is published. In the meantime,
please cite the repository and note that the associated manuscript is under review.

## License

Released under the MIT License. See [LICENSE](LICENSE) for the full text.
