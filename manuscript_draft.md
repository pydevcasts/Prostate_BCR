# Predicting Biochemical Recurrence After Radical Prostatectomy Using Clinical and RNA-Seq Data

> Working manuscript draft. This file is the single evolving document for the paper. Values marked `TBD` must be replaced only after the final leakage-controlled experiments are complete.

## 1. Working Title

**Leakage-Aware Multi-Modal Machine Learning for Biochemical Recurrence Prediction After Radical Prostatectomy: Integration of Clinical and RNA-Seq Features with Explainable Feature Selection**

## 2. Study Status

| Item | Current status |
|---|---|
| Dataset | TCGA-PRAD clinical data and RNA-Seq expression |
| Primary outcome | Biochemical recurrence (BCR) binary classification |
| Modeling architecture | Parallel clinical and genomic branches with late probability fusion |
| Candidate genomic features | Approximately 18,905 after variance filtering |
| MI screening | Top 200 genomic candidates; clinical branch uses `min(200, available features)` |
| Final genomic selection | Binary PSO, currently 40 genes plus 3 pathway scores |
| Clinical selection | Engineered clinical features followed by selection |
| Primary model | Tuned XGBoost |
| External cohort | 106 samples; genomic-only because clinical data are unavailable |
| Final analysis | `TBD`: repeated nested CV, OOF fusion, calibrated threshold, external validation |

## 3. Abstract (Working Version)

### Background

Biochemical recurrence after radical prostatectomy is clinically important and is difficult to predict from heterogeneous patient data. Clinical variables provide established prognostic information, whereas RNA-Seq data may capture molecular signals associated with recurrence risk. We developed a leakage-aware machine-learning framework that models these modalities separately and combines their predicted probabilities using late fusion.

### Methods

We used TCGA-PRAD clinical and RNA-Seq data to construct a binary BCR prediction task. The pipeline included outcome-leakage auditing, missing-value handling, variance filtering, mutual-information screening, and binary particle swarm optimization (PSO). Clinical variables were converted into domain-informed features, while the genomic branch was restricted to gene-expression features and predefined pathway scores. Models were trained with stratified cross-validation and hyperparameter tuning. Model performance was assessed using ROC-AUC, PR-AUC, sensitivity, specificity, F1-score, balanced accuracy, calibration, and bootstrap confidence intervals. An independent external cohort was analyzed separately, with a genomic-only fallback when clinical variables were unavailable.

### Results

In the current internal test analysis, genomic, clinical, and late-fusion ROC-AUC values were 0.686, 0.721, and 0.753, respectively. The current external genomic-only analysis produced an ROC-AUC of 0.560 using the previous saved model. A controlled experiment with 30 PSO-selected genes produced lower internal test performance and an external ROC-AUC of 0.488. These results indicate that reducing the genomic selection target from 40 to 30 features is not currently supported. Final confidence intervals and repeated nested cross-validation results are `TBD`.

### Conclusions

The preliminary results support the feasibility of combining clinical and transcriptomic information, but the model is not yet ready for a strong clinical-performance claim. The next analysis must remove fusion-weight optimism, harmonize genomic features across cohorts, and quantify uncertainty before manuscript submission.

## 4. Introduction

### 4.1 Clinical motivation

BCR is a clinically relevant endpoint after radical prostatectomy. Accurate risk estimation could support postoperative surveillance intensity, patient counseling, and identification of patients who may benefit from additional evaluation or treatment.

### 4.2 Limitations of existing prediction

Clinical-only models may miss molecular heterogeneity. Conversely, high-dimensional transcriptomic data contain many irrelevant, correlated, and cohort-sensitive variables. A useful model must therefore control dimensionality, avoid outcome leakage, remain interpretable, and be evaluated on data that were not used during feature selection or model development.

### 4.3 Study objective

The objective is to develop and rigorously evaluate a multimodal BCR prediction framework that:

1. Separates clinical and genomic information into parallel branches.
2. Performs preprocessing and feature selection only within the relevant training data.
3. Uses domain-informed clinical engineering and genomic pathway scores.
4. Combines branch-level probabilities without allowing one modality to dominate by construction.
5. Provides transparent biomarker interpretation and external validation.

### 4.4 Hypotheses

- A multimodal model will outperform either clinical-only or genomic-only modeling when evaluated without leakage.
- Domain-informed clinical features will provide stable signal with fewer variables.
- Pathway-level genomic features will be more transferable across cohorts than individual genes alone.
- Performance estimates will decrease after optimistic training-set fusion is replaced by out-of-fold fusion; the corrected estimate will be more credible.

## 5. Materials and Methods

### 5.1 Data sources and cohort definition

The development cohort is derived from TCGA-PRAD. Clinical data and RNA-Seq expression data are matched by patient identifier. The primary outcome is `Biochemical_Recurrence_Code`, derived before modeling from the designated clinical recurrence variable.

The external validation data contain 106 observations and genomic measurements but no complete clinical feature set. Therefore, the external analysis is currently genomic-only. The final paper must specify the external dataset accession, inclusion criteria, preprocessing compatibility, outcome definition, and class distribution.

**To add before submission:**

- Development sample count and positive/negative counts.
- External cohort provenance and inclusion flow diagram.
- Follow-up definition and recurrence ascertainment.
- Ethics/data-access statement.

### 5.2 Leakage prevention

Variables representing recurrence, post-recurrence status, survival after the relevant time point, later PSA measurements, or direct outcome proxies are excluded before feature selection. Imputation, variance filtering, mutual-information ranking, PSO, model tuning, threshold selection, and fusion-weight estimation must be fitted using training data only.

The final analysis must verify this rule inside every outer fold. A leakage audit table should report each excluded variable, its reason for exclusion, and the stage at which it was removed.

### 5.3 Preprocessing

Clinical and genomic matrices are imputed using training-fitted median imputation where required. Low-variance features are removed using a threshold of 0.01. The preprocessing object fitted on training data is reused unchanged for validation, test, and external data. RNA-Seq transformation and scaling choices must be described precisely, including whether values are log-transformed, normalized, or batch-adjusted.

### 5.4 Feature engineering

The clinical branch includes domain-informed variables such as Gleason-derived risk, stage-related risk, lymph-node and margin interactions, and encoded clinical categories. The genomic branch uses individual genes selected from the expression matrix and three predefined pathway scores:

- `PSA_Pathway_Score`
- `AR_Signaling_Score`
- `Proliferation_Score`

Pathway definitions, constituent genes, directionality, and score calculation must be reported in a supplementary table.

### 5.5 Feature selection

The genomic selection pipeline is:

1. Variance filtering.
2. Mutual-information ranking.
3. Retention of the top 200 genomic candidates.
4. Binary PSO wrapper selection with a target of 40 genes.
5. Addition of three prespecified pathway scores.

The value 200 is a screening cap, not the final number of genomic predictors. It is appropriate for the genomic dimensionality because approximately 18,905 genes remain after variance filtering. In the clinical branch, fewer than 200 features are available, so the effective MI cap is the number of surviving clinical features.

The final paper should compare `MI_TOP_K` values of 50, 100, and 200 and PSO targets of 20, 30, and 40 using repeated nested CV. The selected value must be chosen by a prespecified performance and stability rule, not by the best single test result.

### 5.6 Model development

The primary classifier is tuned XGBoost. Candidate-model comparisons may include logistic regression, random forest, SVM, LightGBM, and CatBoost, but model selection must occur inside the training procedure. Hyperparameter search must use stratified inner cross-validation and ROC-AUC or a prespecified metric appropriate for class imbalance.

### 5.7 Late fusion

The late-fusion model combines the class probabilities from the genomic and clinical branches:

`p_fusion = w_genomic * p_genomic + w_clinical * p_clinical`

where `w_genomic + w_clinical = 1`. The current saved model used weights of 0.38 and 0.62, but the previous optimization was performed on training predictions and may be optimistic. The corrected implementation must estimate weights from out-of-fold predictions only, then freeze them before evaluating the untouched test set or external cohort.

### 5.8 Threshold selection

ROC-AUC is threshold-independent. Accuracy, sensitivity, specificity, precision, and F1 depend on the classification threshold. The threshold must be selected using training/validation data only. The untouched test set must not be used to choose the threshold. The primary threshold objective should be stated in advance, for example:

- maximum Youden index;
- sensitivity at a prespecified minimum specificity; or
- maximum F1 when the goal is balanced classification.

### 5.9 Evaluation and uncertainty

The primary endpoint should be ROC-AUC with 95% confidence interval. Secondary endpoints should include PR-AUC, balanced accuracy, sensitivity, specificity, precision, F1, MCC, Brier score, calibration slope/intercept, and decision-curve analysis if clinically appropriate. Report the number of positive cases because the current internal test set contains only 12 positives among 86 samples, making single-split estimates unstable.

The final model should be evaluated using repeated stratified nested CV and one untouched test set. Confidence intervals should be generated by stratified bootstrap or repeated outer-fold distributions. Differences between models should be tested using paired fold-level comparisons or a suitable paired ROC test, with multiplicity handled where necessary.

## 6. Current Results

### 6.1 Current internal test results

| Model | ROC-AUC | Accuracy | Sensitivity | Specificity | F1 |
|---|---:|---:|---:|---:|---:|
| Genomic, 40-feature pipeline | 0.6858 | 0.7558 | 0.4167 | 0.8108 | 0.3226 |
| Clinical, 40-feature pipeline | 0.7207 | 0.8256 | 0.1667 | 0.9324 | 0.2105 |
| Late fusion, 40-feature pipeline | **0.7534** | 0.5930 | 1.0000 | 0.5270 | 0.4068 |

These are preliminary single-split results. They should not be presented as definitive generalization performance.

### 6.2 Feature-count experiment

| Setting | Genomic test AUC | Clinical test AUC | Fusion test AUC | External genomic AUC |
|---|---:|---:|---:|---:|
| PSO target 40 | 0.6858 | 0.7207 | **0.7534** | 0.5604* |
| PSO target 30 | 0.5383 | 0.6847 | 0.6295 | 0.4877 |

`*` The 0.5604 value is from the previously saved external prediction file and must be regenerated from the finalized reproducible pipeline before publication.

### 6.3 Clinical benchmark

The current three-fold clinical benchmark ranked variance filtering plus PSO with 30 features first by mean ROC-AUC (0.7690 +/- 0.0254), closely followed by all clinical features (0.7681 +/- 0.0254). This benchmark is useful for hypothesis generation, but it is not a replacement for repeated nested CV or external validation.

### 6.4 Out-of-fold fusion correction

As a first correction for fusion-weight optimism, each current branch model was refit in five stratified folds and generated predictions only for its held-out fold. Fusion weights were selected from the complete OOF predictions, rather than from in-sample training predictions. The resulting weights were 0.48 for the genomic branch and 0.52 for the clinical branch, with OOF fusion ROC-AUC of 0.8530. Applying these frozen weights and an OOF-derived threshold to the untouched internal test set produced ROC-AUC of 0.7376, sensitivity of 58.3%, specificity of 70.3%, and F1-score of 0.3415. This corrected estimate is more conservative than the previous training-optimized fusion result (ROC-AUC 0.7534) and should be preferred for interpretation until full repeated nested CV is complete.

### 6.5 External validation limitation

The external cohort currently contains only a subset of the selected genomic predictors. The 30-gene experiment found 10 of 33 selected gene/pathway columns available in the external matrix. Filling missing genes with imputation is not equivalent to measuring them and may explain the poor transferability. The final external model should use a prespecified common-feature intersection or, preferably, robust pathway-level scores computed from genes available in both cohorts.

## 7. Next Experimental Stage

### Priority 1: Correct evaluation optimism

Implement repeated nested CV with the following order inside each outer training fold:

1. Fit preprocessing.
2. Fit feature selection.
3. Fit model and tune hyperparameters.
4. Generate validation predictions.
5. Generate out-of-fold predictions for each branch.
6. Estimate fusion weights from OOF predictions.
7. Select the operating threshold from OOF predictions.
8. Evaluate once on the untouched outer validation fold.

This is the most important next step because a high training fusion AUC is not evidence of generalization.

### Priority 2: Improve external compatibility

Create a genomic common-feature pipeline using only genes reliably measured in both development and external cohorts. Add pathway scores calculated from common genes. Compare:

- common individual genes;
- common genes plus pathway scores;
- pathway scores alone.

Do not silently treat absent external genes as observed data.

### Priority 3: Compare selection settings

Run repeated nested CV for:

- `MI_TOP_K`: 50, 100, 200;
- `PSO_FINAL_K`: 20, 30, 40;
- clinical strategies: all features, variance+PSO, MI+PSO;
- models: regularized logistic regression and tuned tree model.

Select a configuration using mean performance, uncertainty, and feature-selection stability. A small performance difference with much better stability and external compatibility should be preferred.

### Priority 4: Calibration and clinical utility

Calibrate the final probabilities using training folds only. Report calibration curves, Brier score, calibration intercept/slope, and decision-curve analysis. Select a threshold according to the clinical use case rather than maximizing accuracy on the test set.

### Priority 5: Reproducibility and reporting

Freeze the final configuration, rerun notebooks 04, 05, 07, and 08 from a clean environment, and save:

- final feature lists;
- fold-level predictions;
- test and external predictions;
- confidence intervals;
- model parameters;
- software versions;
- leakage audit;
- data-flow diagram;
- TRIPOD-AI or equivalent reporting checklist.

## 8. What Would Count as a Successful Result?

The goal should not be an isolated accuracy of 90%. A credible result would show:

- stable performance across repeated outer folds;
- ROC-AUC and PR-AUC above reasonable baselines;
- sensitivity and specificity appropriate to the intended clinical use;
- calibration and decision benefit;
- no material performance collapse in external validation;
- stable selected biomarkers across folds;
- a clear improvement over clinical-only and genomic-only baselines.

If accuracy reaches 90% only after lowering the threshold or predicting the majority class, this should not be described as a clinically successful model.

## 9. Discussion Points

### Principal findings

The current analysis suggests complementary information between clinical and genomic branches, because late fusion outperformed each branch by ROC-AUC on the internal test split. However, the gain is preliminary and may be affected by training-based fusion-weight optimization and a small number of positive test cases.

### Why 30 features did not improve performance

Reducing the PSO target from 40 to 30 may remove weak but complementary predictors, increase selection instability, or produce a feature set that is less compatible with the external cohort. The experiment does not show that 30 is universally inferior; it shows that this single seeded 30-feature configuration was inferior and should not replace the 40-feature default without repeated validation.

### Limitations

Important limitations include the modest sample size, class imbalance, single internal split, incomplete clinical data in the external cohort, cross-cohort measurement differences, and possible optimism from prior fusion-weight and threshold selection. These limitations must be stated directly.

## 10. Conclusion (To Be Finalized)

This study presents a leakage-aware framework for integrating clinical and RNA-Seq information to predict biochemical recurrence after radical prostatectomy. Preliminary results support late fusion as a promising modeling strategy, but final claims require corrected out-of-fold fusion, repeated nested validation, feature harmonization, calibration, and independent external confirmation.

## 11. Tables and Figures Required for Submission

1. Cohort selection flow diagram.
2. Data-processing and model architecture diagram.
3. Leakage audit table.
4. Baseline clinical characteristics table.
5. Model comparison with repeated nested-CV estimates and 95% CIs.
6. ROC and PR curves on the untouched test set.
7. Calibration plot and Brier scores.
8. Decision-curve analysis.
9. External validation performance.
10. Feature-selection stability plot.
11. SHAP summary for the final locked model.
12. Supplementary table of selected genes and pathway definitions.

## 12. Change Log

| Date | Change |
|---|---|
| 2026-09-15 | Created manuscript draft from current Late Fusion outputs. |
| 2026-09-15 | Documented the controlled PSO target-30 experiment and restored the default target to 40. |
| 2026-09-15 | Identified the external common-feature limitation and training-based fusion-weight optimism. |
| 2026-09-15 | Added OOF fusion weight/threshold estimation and recorded the first corrected internal test result. |

## 13. Source Artifacts

- Configuration: `core/config.py`
- Feature selection: `core/src/genomic_selector.py`, `core/src/clinical_selector.py`
- Clinical benchmark: `core/src/clinical_benchmark.py`
- Late fusion: `core/src/fusion/late_fusion.py`
- OOF fusion utility: `core/src/fusion/late_fusion.py` (`estimate_oof_fusion`)
- Current internal metrics: `core/outputs/tables/final_evaluation.json`
- OOF fusion metrics: `core/outputs/tables/oof_fusion_results.json`
- Clinical benchmark results: `core/outputs/tables/clinical_strategy_benchmark_summary.csv`
- PSO target-30 experiment: `core/outputs/tables/experiment_k30_results.json`
- External predictions: `core/outputs/tables/external_validation_results.csv`

## 14. Final Pre-Submission Checklist

- [ ] Resolve the discrepancy between README results and current Late Fusion results.
- [ ] Rerun all final notebooks from a clean kernel.
- [ ] Confirm every preprocessing and selection step is fold-local.
- [ ] Replace training-based fusion weights with OOF weights.
- [ ] Choose threshold without touching the final test set.
- [ ] Rebuild external validation using common measured features.
- [ ] Add 95% confidence intervals and statistical comparisons.
- [ ] Add calibration and decision-curve analysis.
- [ ] Report class counts, missingness, and cohort flow.
- [ ] Lock code, environment, seeds, and artifact hashes.
- [ ] Complete TRIPOD-AI reporting items and journal-specific requirements.
