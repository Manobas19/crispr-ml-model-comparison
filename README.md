# Compare machine-learning models for CRISPR activity prediction

**Task chosen:** regression of measured guide on-target activity from guide sequence. This is not an off-target classifier, screen-hit ranking or indel-outcome model.

Models: **Mean baseline, Ridge regression, SVR, Random Forest, XGBoost**. Tools from the brochure: NumPy, Pandas, scikit-learn, XGBoost, Matplotlib and Seaborn. SciPy supplies rank correlation; joblib exports the selected model.

The notebook includes a synthetic demonstration and a real-data upload route. **Synthetic performance is not biological validation.** No API key or GPU is required.

## Required input
`activity.csv` must contain:

| Column | Meaning |
|---|---|
| guide_id | Unique identifier for each guide |
| guide_sequence | 20 A/C/G/T bases, spacer only, excluding PAM; consistent 5′→3′ convention |
| activity | Measured numeric activity, with the same assay definition, units and normalization across records |
| target_group | Gene/locus or a more conservative cluster used to prevent related guides crossing splits |

Use experimentally measured labels. Guide-design scores predicted by another model are not experimental truth. Growth/fitness screen counts are not automatically direct editing activity. Do not pool incompatible assays, cell contexts, nucleases or label scales without an explicit harmonization design.

The default is 20-base spacers; use a fixed alternative length only with a compatible study and the matching sequence-length option. N bases and mixed lengths are rejected. This benchmark is sequence-only: it cannot account for cell type, chromatin or assay effects not represented in the features.

At least 40 unique guides and 12 independent groups are required by this workflow, with at least three held-out and five training groups. These are software minimums, not guarantees of statistical power. More groups and experimental validation are preferable. Repeated sequence records are rejected: resolve exact duplicates and technical/biological replicate labels within a coherent assay before input. Do not average incompatible conditions.

Choose groups before examining performance. Use the broadest relevant dependency unit: if near-identical sequences, related loci or study batches can cause leakage, assign them to the same appropriate cluster. This code blocks exact duplicate guides and group overlap, but does not automatically cluster sequence homology or detect all batch confounding.

## Benchmark design
1. Encode each sequence into positional one-hot features, GC fraction, longest homopolymer, base entropy and overlapping dinucleotide frequencies. For a 20-base guide there are 99 features. These features require no fitted data transformations.
2. Hold out 20% of groups using a fixed seed; this is a group fraction, not an exact row fraction.
3. Tune only training data with identical GroupKFold splits for all models. Scaling for Ridge and SVR is fitted inside each fold using a Pipeline.
4. Choose the model family and parameters using training-CV RMSE only. No early stopping uses the test set.
5. Evaluate all fitted models once on held-out groups. Save the training-CV-selected model fitted only on the training partition.
6. Resample whole held-out groups for 95% cluster-bootstrap intervals. All models use the same resampled groups, enabling paired comparison to the baseline.

Primary metric is **row-weighted RMSE** (lower is better); groups with more guides contribute more rows. MAE is also lower-is-better; R² and Spearman are higher-is-better. R² can be negative. Constant predictions have undefined Spearman; the mean baseline therefore has a missing Spearman value rather than a misleading zero. Metrics depend on label scale.

The best tuning-CV score is used for selection and can be optimistic; it is not an independent generalization estimate. This workflow uses one held-out split, not repeated nested CV or external validation. Intervals are conditional on the trained models and observed test groups, exclude training/tuning uncertainty, and can be unstable with few groups. Paired baseline intervals are descriptive and are not adjusted for multiple model comparisons.

Do not change features, seed or hyperparameters in response to the test results and continue calling that same set an untouched test. An external dataset with comparable assay labels is the next evaluation step.

## Colab / Jupyter
Upload CRISPR_ML_Model_Comparison.ipynb to https://colab.research.google.com/ and run cells in order.
Default DEMO_MODE=True creates synthetic data. For real data set DEMO_MODE=False and upload activity.csv.
QUICK_MODE=True uses a small search for teaching. For the full comparison use QUICK_MODE=False before the first real benchmark run.

## Ubuntu (Python 3.10+)
Extract this package and open a terminal in crispr_ml_benchmark:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python benchmark.py --make-demo demo_activity.csv
python benchmark.py --input demo_activity.csv --out demo_results --demo-label --quick
```

Real data:

```bash
python benchmark.py --input activity.csv --out real_results --bootstrap 1000 --jobs 2
python -m pip freeze > real_results/installed_versions.txt
```

Use a fresh output folder per run. See Ubuntu_Benchmark_Commands.sh for the separate commands.

## Outputs
- training_cv_comparison.csv: training-only tuning/selection results and measured fitting/search times.
- heldout_metrics.csv: test RMSE, MAE, R², Spearman and cluster-bootstrap intervals; ordered by training CV.
- heldout_predictions.csv: observed and predicted activities, guide IDs and group labels.
- improvement_over_baseline.csv: mean-baseline RMSE minus model RMSE (positive favors the model) and paired bootstrap intervals.
- split_manifest.csv: train/test assignments and training validation-fold assignments.
- *_tuning.csv: complete hyperparameter search results.
- model_comparison.png and predicted_vs_observed.png.
- selected_model.joblib: training-CV-selected estimator, feature names and sequence length; use the same features() function to transform new sequences.
- run_manifest.json: configuration, input hash, best parameters, selection criterion and feature schema.
- installed_versions.txt: exact resolved versions (written by notebook/terminal commands).

Only load joblib/pickle model files you trust. Model exports contain Python objects, not a safe interchange format. The saved estimator expects the feature matrix, not raw sequence strings.

## Interpretation
A model beating the mean baseline on synthetic data only shows that the pipeline can learn the synthetic signal. It does not establish performance on real CRISPR experiments. No particular model is guaranteed to win. A strong result requires sufficient independent groups, meaningful labels and external biological validation.
This task uses regression: accuracy, F1 and ROC-AUC are not appropriate unless you first define a separate biologically justified classification task. DeepCRISPR, DNABERT and neural networks need different pretrained assets/data requirements and are not falsely included as trained models here. SHAP is not required for performance comparison.

## Documentation
https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html
https://scikit-learn.org/stable/modules/cross_validation.html
https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html
https://xgboost.readthedocs.io/en/stable/python/python_api.html

Created 20 October 2026. No user experimental data were supplied.

## Included input examples
activity_format_example.csv illustrates columns only and is too small to run. demo_activity.csv contains 360 synthetic examples across 30 groups; use --demo-label. See VALIDATION.txt for the exact locally exercised paths. XGBoost execution remains to be verified in your installed environment.
