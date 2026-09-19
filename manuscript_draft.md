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
| External cohort 1 | GSE54460: 106 samples, genomic-only fallback, ROC-AUC 0.560 |
| External cohort 2 (primary) | MSKCC 2010: 131 primary-tumor samples (27 events, 20.6%), six prespecified transferable features, frozen late fusion ROC-AUC 0.711-0.717 |
| Headline external finding | The frozen transferable fusion performs on par with a three-feature pure-clinical baseline (0.695); the added value of the genomic branch externally is NOT demonstrated and is reported as a documented limitation |
| Strategic decision (2026-09-18) | Clinical-centric narrative locked; no further external transfer attempts without a new prespecified hypothesis (external overfitting risk) |
| Final analysis | Repeated nested CV from raw TCGA complete (branch-mask corrected): mean OOF ROC-AUC 0.7819 +/- 0.0203 (5 folds x 3 repeats, fully fold-local); stability analysis complete (6.13); manuscript drafting remains |

## 3. Abstract (Working Version)

### Background

Biochemical recurrence after radical prostatectomy is clinically important and is difficult to predict from heterogeneous patient data. Clinical variables provide established prognostic information, whereas RNA-Seq data may capture molecular signals associated with recurrence risk. We developed a leakage-aware machine-learning framework that models these modalities separately and combines their predicted probabilities using late fusion.

### Methods

We used TCGA-PRAD clinical and RNA-Seq data to construct a binary BCR prediction task. The pipeline included outcome-leakage auditing, missing-value handling, variance filtering, mutual-information screening, and binary particle swarm optimization (PSO). Clinical variables were converted into domain-informed features, while the genomic branch was restricted to gene-expression features and predefined pathway scores. Models were trained with stratified cross-validation and hyperparameter tuning. Model performance was assessed using ROC-AUC, PR-AUC, sensitivity, specificity, F1-score, balanced accuracy, calibration, and bootstrap confidence intervals. An independent external cohort was analyzed separately, with a genomic-only fallback when clinical variables were unavailable.

### Results

A five-fold nested, leakage-aware evaluation on preselected artifacts yielded an OOF fusion ROC-AUC of 0.861; the definitive repeated nested evaluation, which refits preprocessing, feature selection and both branch models from the raw matrix inside every fold (5 folds x 3 repeats), gave a corrected mean OOF ROC-AUC of 0.7819 +/- 0.0203 with the OOF procedure weighting the clinical branch at 0.66-1.00. Feature-selection stability over 35 prespecified PSO events showed the genomic selection to be unstable (maximum selection frequency 0.31, no feature reaching the prespecified 0.70 threshold) whereas the clinical branch was stable (top features 0.80-1.00). Under a prespecified transfer rule, a late-fusion model retrained on six transferable features (three pathway scores plus Gleason_Total, High_Risk_Gleason, and T_Stage_Risk) and applied frozen to the MSKCC 2010 cohort (131 primary-tumor samples, 27 recurrences) reached ROC-AUC 0.711-0.717 depending on the genomic transfer variant (raw-scaler or within-cohort rank normalization), with 95% bootstrap CIs of 0.579-0.831 and 0.588-0.830. Branch decomposition showed the external performance was carried by the clinical branch (MSKCC clinical AUC 0.711); the genomic branch scored 0.500 under raw scaling (platform-scale artifact) and 0.586 after rank normalization, while remaining strong within TCGA (0.836 on the internal test rows). A prespecified pure-clinical baseline (logistic regression on the three clinical features) reached 0.695 externally and was statistically indistinguishable from both fusion variants (paired bootstrap p >= 0.34). The rank-transfer fusion showed the best calibration (Brier 0.145, recalibration slope 1.24), and decision-curve analysis showed no consistent net-benefit dominance over the baseline. The earlier GSE54460 genomic-only analysis produced ROC-AUC 0.560.

### Conclusions

A leakage-aware, transferable late-fusion framework achieves external discrimination on par with a parsimonious clinical model on an independent microarray cohort, with acceptable calibration. The transferable genomic signal could not be demonstrated beyond the clinical baseline across two prespecified transfer rules and is reported as an open limitation, plausibly due to cross-platform probe differences and cohort heterogeneity. The methodological contribution is the leakage-free transfer pipeline itself; claims of genomic added value require platform-robust signatures and multi-cohort re-estimation.

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

The validation code now includes an explicit common-feature alignment utility that preserves reference-column order and reports missing predictors without fabricating them. External performance should be regenerated with this rule after the common-gene or pathway-only model is locked.

### 6.6 Nested feature-selection and OOF fusion experiment

The first five-fold nested experiment performed MI/PSO selection separately within each fold, fitted both branch models only on fold-specific training data, and generated held-out predictions for every observation. Fusion weights and the operating threshold were then estimated from the complete OOF predictions. The resulting genomic and clinical weights were 0.62 and 0.38. OOF fusion performance was ROC-AUC 0.8606, PR-AUC 0.5533, accuracy 0.8513, balanced accuracy 0.7947, sensitivity 0.7174, specificity 0.8721, and F1-score 0.5641.

This result is encouraging, but it is not yet the final unbiased estimate because the input matrices were already feature-selected artifacts. The definitive analysis must repeat the same procedure from raw clinical and transcriptomic matrices, including upstream preprocessing and feature engineering inside each outer fold.

### 6.7 MSKCC 2010 external cohort construction

A clean external cohort was built from the cBioPortal PRAD-MSKCC 2010 export (Agilent microarray platform, Entrez gene identifiers) with an auditable filter chain. Of 238 patients, 198 carried a valid disease-free survival status (61 recurred, 137 disease-free). Of 240 clinical samples, 218 were tumor class and 181 were primary tumors; all 181 matched to patients with valid survival status. Intersecting with the expression matrix (156 profiled sample columns) left 131 unique patients, because many primary tumors were not profiled on the expression platform.

The resulting analysis cohort therefore contains 131 samples with 27 recurrence events (event rate 20.6%). The patient-level figures (198 patients, 30.8% event rate) should not be quoted as the modeling cohort; the sample-level intersection is the effective external validation set. One sample has a missing Gleason total and Gleason patterns. The cohort table is stored at `core/data/external/mskcc_cohort.csv` and the filter audit at `core/outputs/tables/mskcc_cohort_summary.json` (built by `core/src/mskcc_cohort.py`).

### 6.8 Transferable feature construction (Step 2)

Six prespecified features were built identically on both cohorts so a fusion model can be retrained on TCGA and applied to MSKCC: the three pathway scores (mean expression of the PSA, AR-signaling, and proliferation gene sets; all 16 constituent genes are present in the MSKCC Agilent matrix via direct Entrez-ID mapping) and Gleason_Total, High_Risk_Gleason (Gleason pattern primary >= 4 or secondary >= 4), and T_Stage_Risk (T3/T4 indicator; from the TCGA one-hot columns and from MSKCC PATH_T_STAGE). Margin x LymphNode was dropped because MSKCC has no margin or lymph-node fields.

The canonical TCGA split was reproduced deterministically from `X_features_final.csv` (train_test_split, test_size 0.20, stratified, seed 42) and hard-validated: the regenerated train and test targets match the saved y_train/y_test exactly (343/86 rows, 58 events). The T_Stage_Risk values also agree with the engineered artifact after accounting for a historical quirk: the old pipeline engineered features after standardization, so its T_Stage_Risk stores sums of scaled one-hot columns, while the transferable table stores the raw 0/1 indicator; binary agreement was verified for every training row.

Cohort distributions differ in case mix and must be reported: T_Stage_Risk is positive in 269/429 TCGA samples (62.7%) versus 46/131 MSKCC samples (35.1%); High_Risk_Gleason is positive in 391/429 TCGA samples (91.1%) versus 89/130 known MSKCC samples (68.5%); Gleason_Total spans 6-10 (TCGA) and 6-9 (MSKCC, one sample missing). One important scale caveat is recorded for the next stage: TCGA expression columns in the merged table are on a raw-count-like scale while the MSKCC Agilent matrix is log-scale, so the raw pathway scores are not directly comparable across cohorts; the retraining step must standardize within cohort (scaler fitted on TCGA train) under a prespecified transfer rule.

Outputs: `core/data/processed/tcga_transferable_features.csv` (429 rows with split labels), `core/data/external/mskcc_transferable_features.csv` (131 rows), and audits at `core/outputs/tables/tcga_transferable_features_summary.json` and `core/outputs/tables/mskcc_transferable_features_summary.json` (built by `core/src/transferable_features.py`).

### 6.9 Transferable late fusion retraining and external application (Step 3)

The fusion branches were retrained on TCGA using only the six transferable features under a prespecified transfer rule: a StandardScaler fitted on the 343 TCGA training rows and applied unchanged to the TCGA test rows and to MSKCC; XGBoost branch models with the project's default hyperparameters and automatic scale_pos_weight; fusion weights and the operating threshold estimated only from TCGA out-of-fold predictions (estimate_oof_fusion, five folds); and the frozen pipeline applied to MSKCC with no external tuning of any kind. The single MSKCC sample with missing Gleason fields (PCA0171) was imputed with TCGA-train medians before scaling.

On TCGA, the OOF fusion weights were 0.27 (genomic) and 0.73 (clinical) with OOF fusion ROC-AUC 0.6913 and an OOF Youden threshold of 0.4786. On the untouched TCGA test rows the frozen fusion reached ROC-AUC 0.7736 (95% CI 0.635-0.903), with a genomic-branch AUC of 0.8361 and a clinical-branch AUC of 0.5974.

Applied frozen to MSKCC, the fusion reached ROC-AUC 0.7110 (95% CI 0.579-0.831). At the transferred threshold the model flagged 12 of 131 patients with sensitivity 0.370, specificity 0.981, PPV 0.833, NPV 0.857, and balanced accuracy 0.676. A post-hoc Youden threshold recomputed on MSKCC (0.4578, diagnostic only, not a prespecified operating point) would give sensitivity 0.630 and specificity 0.788.

The decisive finding is the branch decomposition: the external fusion AUC is carried entirely by the clinical branch (MSKCC clinical AUC 0.7110), while the genomic branch collapsed to AUC 0.5000 with a constant prediction (0.5645) for all 131 samples. The audit explains why: after the frozen TCGA scaler, the MSKCC pathway features sit 2.05, 2.05, and 0.96 standard deviations below the TCGA training mean with near-zero variance (standard deviations of 6e-6, 7e-6, and 1e-3). The five-orders-of-magnitude expression-scale difference (raw counts versus log-scale) therefore compresses the external genomic signal to noise under within-cohort standardization. The external result should be reported as a fusion number with this branch decomposition, not as evidence that the genomic signature transferred.

Outputs: `core/outputs/tables/transferable_fusion_results.json`, `core/outputs/tables/mskcc_transferable_predictions.csv`, `core/outputs/tables/tcga_transferable_test_predictions.csv`, and the three frozen artifacts under `core/outputs/models/transferable_*.joblib` (built by `core/src/transferable_fusion.py`).

### 6.10 Rank-based genomic transfer (Step 4, prespecified single run)

To address the genomic-branch collapse identified in 6.9, a second transfer rule was declared and written down before any external metric for it was computed or inspected: within each cohort, independently and label-free, every pathway gene's expression column is replaced by its percentile rank among that cohort's own samples, pathway scores become mean ranks, and the TCGA-train-fitted scaler and otherwise identical leakage-free fusion recipe are applied unchanged. The rule equalizes the score marginal distributions across cohorts by construction (uniform in both), removing the raw-counts-versus-log-scale platform shift without sharing cross-cohort statistics or touching any label. The pipeline was executed exactly once.

Results: on TCGA the OOF fusion weights shifted further toward the clinical branch (0.19 genomic / 0.81 clinical, OOF AUC 0.6991, threshold 0.4088), and the rank-transformed genomic branch scored 0.7218 on the untouched TCGA test rows (versus 0.8361 with raw features in 6.9 — rank normalization costs within-cohort signal). Applied frozen to MSKCC, the fusion reached ROC-AUC 0.7169 (95% CI 0.588-0.830) with a genomic-branch AUC of 0.5856 (up from 0.5000) and an unchanged clinical-branch AUC of 0.7110; at the transferred threshold sensitivity was 0.444, specificity 0.885, and balanced accuracy 0.665. The post-hoc Youden threshold on MSKCC (0.3495, diagnostic only) would give sensitivity 0.630 and specificity 0.750.

Interpretation: the rank rule removed the platform-scale artifact (MSKCC z-means moved from -2.05/-2.05/-0.96 to +0.045/+0.034/+0.017) and partially revived the genomic branch, but the external genomic signal remains weak (0.586) and the external fusion performance is still carried predominantly by the clinical branch (the OOF procedure itself down-weights the genomic branch to 0.19). Both transfer variants should be reported transparently; a transferable genomic signature remains an open limitation, plausibly due to platform probe differences and cohort heterogeneity beyond marginal-scale correction. Outputs: `core/outputs/tables/rank_transfer_fusion_results.json`, `core/outputs/tables/mskcc_rank_transfer_predictions.csv`, `core/outputs/tables/tcga_rank_transfer_test_predictions.csv`, and the rank artifacts under `core/outputs/models/rank_transfer_*.joblib` (built by `core/src/rank_transfer_fusion.py`).

### 6.11 Clinical-value diagnostics: baseline, calibration, and decision analysis (Step 5)

A prespecified pure-clinical baseline (logistic regression on Gleason_Total, High_Risk_Gleason, and T_Stage_Risk, trained on TCGA train only with the same frozen-scaler recipe) was applied to MSKCC and compared against the frozen fusion variants with paired stratified bootstrap (3000 iterations). The baseline reached ROC-AUC 0.6950, statistically indistinguishable from the raw-transfer fusion (delta AUC +0.0160, 95% CI -0.019 to +0.047, bootstrap p = 0.344) and the rank-transfer fusion (+0.0219, 95% CI -0.035 to +0.073, p = 0.411); the two fusion variants were also mutually indistinguishable (+0.0059, p = 0.769). The rank-transfer genomic branch alone scored below the baseline (delta AUC -0.1093, 95% CI -0.283 to +0.067, p = 0.222). On this external cohort the added value of the genomic signal over the three-feature clinical baseline is therefore NOT demonstrated, and the honest headline is that the transferable fusion performs on par with a parsimonious clinical model.

Calibration on MSKCC (post-hoc diagnostics): the rank-transfer fusion had the best Brier score (0.1446, slope 1.24, intercept -0.41), slightly better than the clinical baseline (0.1507, slope 1.00, intercept -0.59, indicating systematic under-prediction); the raw-transfer fusion was over-dispersed (slope 1.52) and the genomic branch alone was badly miscalibrated (Brier 0.205, slope 0.24). Decision Curve Analysis showed no consistent dominance: the fusion variants offered modest net-benefit advantages over the baseline only in scattered mid-range threshold bands (e.g., 0.50-0.70), while the baseline was preferable around 0.25-0.35; treat-all and treat-none strategies bracketed small net benefits throughout, reflecting the 20.6% event rate.

For the manuscript, these diagnostics reframe the contribution: the Late Fusion architecture and the leakage-free transfer pipeline are methodologically sound, but the external evidence supports a clinically driven model; the genomic branch requires either a platform-robust signature (e.g., rank/ComBat-harmonized retraining or cross-platform probe remapping) or multi-cohort re-estimation before any transfer claim is made. Outputs: `core/outputs/tables/clinical_value_diagnostics.json`, `core/outputs/tables/mskcc_dca_curves.csv`, and `core/outputs/figures/mskcc_calibration_dca.png` (built by `core/src/clinical_value_diagnostics.py`).

### 6.12 Repeated nested CV from raw TCGA (definitive internal estimate)

The final internal estimate was computed with repeated nested CV directly from the raw merged TCGA table (429 samples x 19,019 features) so that no preselected artifact enters an outer validation fold. Inside every outer fold the complete pipeline was refitted: preprocessing (log1p transformation, winsorization, z-scaling via `build_combined_pipeline`), variance filtering, mutual-information screening (top 200 per branch), binary PSO selection (target 40 per branch), and both XGBoost branch models. Fusion weights and the operating threshold were then estimated only from the aggregate out-of-fold predictions of all 343 training rows of that repeat. Three repeats (seeds 42, 43, 44) were run; the procedure was executed three times end-to-end and produced identical repeat AUCs each time, confirming determinism.

> **Branch-mask correction (2026-09-19).** An earlier revision of this module grouped the branch columns with a hand-rolled clinical-prefix list that matched only 70 of the 115 clinical columns. The 45 missed one-hot clinical features (e.g. `Tumor Other Histologic Subtype_*`) therefore entered the genomic branch, and the positional split mislabelled transformed columns. All numbers below are from the corrected run, which derives both masks from the project's canonical `identify_column_groups` (the same function `build_combined_pipeline` uses) and is gated by an explicit consistency check against that function's transformer column lists. The affected revision reported 0.7551 +/- 0.0199 with genomic-leaning weights; that number is superseded and must not be cited.

| Repeat seed | OOF fusion AUC | Genomic/clinical weights | OOF threshold | Runtime |
|---|---:|---|---:|---:|
| 42 | 0.7817 | 0.33 / 0.67 | 0.2712 | 9.6 min |
| 43 | 0.8023 | 0.00 / 1.00 | 0.1602 | 9.6 min |
| 44 | 0.7616 | 0.34 / 0.66 | 0.1881 | 9.7 min |
| **Mean +/- SD** | **0.7819 +/- 0.0203** | 0.22 / 0.78 (mean) | — | 28.9 min total |

Per-fold PSO selected exactly 40 genomic and 40 clinical features in every fold. Fold-mean branch AUCs per repeat were 0.605 / 0.594 / 0.673 (genomic) and 0.778 / 0.805 / 0.751 (clinical); per-fold rows are in the artifact. Four findings matter for interpretation:

1. The definitive internal estimate is 0.7819 +/- 0.0203, below the 0.861 artifact-based nested estimate (6.6) — the expected direction, because that earlier estimate ran MI/PSO on top of already-preselected matrices and inherited their optimism. The corrected revision is also *higher* than the contaminated one (0.7551), i.e. the mask bug depressed, not inflated, the fusion estimate: leaked clinical one-hots diluted the genomic branch while starving the clinical branch of 45 of its own columns.
2. Branch balance now leans clinical, consistent with the external result: the OOF procedure weights the clinical branch 0.66-1.00 (genomic 0.00-0.34), and fold-mean clinical AUC exceeds the genomic branch in every repeat (0.751-0.805 versus 0.594-0.673). Internal (6.12) and external (6.9-6.11) evidence therefore agree that the usable signal in this cohort is carried by the clinical modality — a coherence argument that should be stated in the Discussion.
3. The OOF-derived thresholds remain low (0.16-0.27) relative to 0.5, consistent with scale_pos_weight-adjusted XGBoost probabilities under the 13.5% event rate; thresholds must be reported with this caveat and recalibrated before any deployment framing.
4. In one of three repeats (seed 43) the OOF weight optimisation assigned zero weight to the genomic branch, i.e. the procedure itself selects a clinical-only fusion on that split. This is reported rather than smoothed away; it matches the branch-AUC ordering above.

Per-fold PSO selections for both branches were logged for all 15 folds (`nested_cv_raw_pso_selections.csv`), providing the empirical selection-frequency input for the feature-stability analysis (next stage).

Outputs: `core/outputs/tables/nested_cv_raw_results.json`, `nested_cv_raw_folds.csv`, `nested_cv_raw_oof_predictions.csv`, and `nested_cv_raw_pso_selections.csv` (built by `core/src/nested_cv_raw.py`).

### 6.13 Feature-selection stability across 35 prespecified PSO events (Step 7)

Because the wrapper selector is stochastic, the reproducibility of the selected feature sets was quantified under rules fixed before any stability number was computed: 35 selection events per branch (the 15 fold-local selections from 6.12 plus 20 fresh runs on the full canonical 343-row train split with the identical preprocessing/variance/MI/PSO recipe, PSO seeds 500-519), stability defined as selected-events / 35, and a stability threshold of 0.70 frozen in advance. The event count, seed list, and threshold are recorded in `stability_summary.json` with `frozen_before_running: true`.

| Branch | Unique features over 35 events | Max fold-based frequency (of 15) | Max same-data frequency (of 20) | Features with stability >= 0.70 |
|---|---:|---:|---:|---:|
| Genomic | 644 | 8/15 | 10/20 | **0** |
| Clinical | 80 | 15/15 | 20/20 | **8** |

**Genomic selection is not reproducible.** Across 35 events the PSO selected 644 distinct genes out of 1,400 selection slots; the highest stability of any gene is 0.371 (TROAP and ACVRL1, each 13/35), only 6 genes exceed 10/35 events, and the median gene stability is 0.029. Instability has both sources: across folds no gene is ever selected by more than 8 of 15 fold selections (different training subsets), and even on identical data the best gene is selected in only 10 of 20 runs (optimizer stochasticity). No genomic feature reaches the prespecified 0.70 threshold, so no stable gene-level signature can be reported from this cohort. This result is fully consistent with the external findings in 6.9-6.11: a gene set that is not reproducible under resampling and seed variation within the development cohort cannot be expected to transfer across platforms, and it explains why the genomic branch could not be shown to add external value. The paper should therefore present the prespecified pathway scores - not the PSO gene list - as the genomic representation, and report gene-level instability as a result rather than hiding it in a supplementary note.

**Clinical selection is reproducible.** Only 80 distinct clinical features appear across the same 35 events, the median feature stability is 0.486, and 8 features reach the frozen 0.70 threshold:

| Clinical feature | Fold events (of 15) | Same-data events (of 20) | Stability |
|---|---:|---:|---:|
| Person Neoplasm Status_WITH TUMOR | 15 | 20 | 1.000 |
| Year Cancer Initial Diagnosis | 12 | 20 | 0.914 |
| SEX | 9 | 19 | 0.800 |
| Mri results_Extraprostatic Extension Localized | 14 | 13 | 0.771 |
| Positive Finding Lymph Node H&E Microscopy Count | 12 | 15 | 0.771 |
| Surgical Margin Resection Status_R0 | 12 | 14 | 0.743 |
| Neoplasm AJCC Clinical Primary Tumor T Stage_T2 | 9 | 16 | 0.714 |
| Primary Therapy Outcome Success Type_Partial Remission/Response | 9 | 16 | 0.714 |

Five of these are established prostatectomy prognostic factors (extraprostatic extension on MRI, lymph-node involvement, surgical margin status, T stage, and sex) and require no special comment. Three, however, must be treated as audit items before submission, because their stability may reflect outcome proximity rather than prognostic signal: `Person Neoplasm Status_WITH TUMOR` (perfectly stable at 35/35) and `Primary Therapy Outcome Success Type_Partial Remission/Response` are recorded in a disease-status/response field, and `Year Cancer Initial Diagnosis` (12/15 folds, 20/20 same-data) is a calendar variable that can encode cohort or batch structure. The next revision must verify the temporal provenance of these three fields against the TCGA data dictionary and, if any is recorded after the recurrence endpoint, re-run this analysis without it. Until then, the clinical feature set should be described with this caveat rather than presented as a validated signature.

Together the two analyses define the paper's honest stability profile: the fusion performance is carried by a stable clinical branch and by prespecified pathway scores, while the machine-selected 40-gene signature is an artifact of a single optimizer run and does not survive resampling. Outputs: `core/outputs/tables/stability_summary.json`, `core/outputs/tables/stability_genomic.csv`, `core/outputs/tables/stability_clinical.csv`, `core/outputs/tables/stability_fresh_runs.csv` (per-run selections, resumable checkpoint), and `core/outputs/figures/feature_stability.png` (built by `core/src/stability_selection.py`; the aggregation code counts distinct events - an earlier revision that counted each feature once per group was corrected before these numbers were produced).

## 7. Next Experimental Stage

### Current decision: defer external validation

External validation is intentionally deferred until the combined clinical-plus-expression cohort is downloaded, its outcome definition is confirmed, and the feature schema is harmonized. The downloaded data should be stored locally under `core/data/external/`; it must not be committed or used to tune the current internal model.

### Priority 1: Correct evaluation optimism — COMPLETED (see 6.12)

The raw-data repeated nested CV is implemented as `core/src/nested_cv_raw.py` and reported in section 6.12. The prespecified order inside each outer training fold:

1. Fit preprocessing.
2. Fit feature selection.
3. Fit model and tune hyperparameters.
4. Generate validation predictions.
5. Generate out-of-fold predictions for each branch.
6. Estimate fusion weights from OOF predictions.
7. Select the operating threshold from OOF predictions.
8. Evaluate once on the untouched outer validation fold.

This is the most important next step because a high training fusion AUC is not evidence of generalization.

The artifact-level first implementation (`evaluate_nested_late_fusion` in `core/src/fusion/nested_evaluation.py`) provided fold-level results, OOF probabilities, OOF-derived weights, and an OOF-derived Youden threshold. The raw-data wrapper (`core/src/nested_cv_raw.py`) wraps preprocessing and selection around the same recipe so that no preselected artifact enters an outer validation fold; its result (mean OOF AUC 0.7819 +/- 0.0203 after the branch-mask correction, 6.12) is the definitive internal estimate.

### Priority 2: Improve internal model stability — COMPLETED (see 6.13)

Feature-selection stability was quantified over 35 prespecified PSO events: the genomic selection is unstable (no feature at the frozen 0.70 threshold; median stability 0.029) while the clinical branch is stable (8 features >= 0.70). Any future change to the final feature count must be judged against this stability profile, not only against mean performance.

Run repeated nested CV for the current internal cohort before changing the final feature count. Compare all clinical features, variance filtering plus PSO, and MI plus PSO using the same outer folds. Select the configuration using mean performance, uncertainty, and feature-selection stability rather than a single test accuracy.

The current benchmark slightly favors variance filtering plus PSO with 30 clinical features (mean ROC-AUC 0.7690) over all clinical features (0.7681), while the current 40-feature Late Fusion test result remains the strongest single-split fusion result. This difference is too small to justify a final choice without repeated validation.

### Priority 3: Improve external compatibility

Create a genomic common-feature pipeline using only genes reliably measured in both development and external cohorts. Add pathway scores calculated from common genes. Compare:

- common individual genes;
- common genes plus pathway scores;
- pathway scores alone.

Do not silently treat absent external genes as observed data.

### Priority 4: Compare selection settings

Run repeated nested CV for:

- `MI_TOP_K`: 50, 100, 200;
- `PSO_FINAL_K`: 20, 30, 40;
- clinical strategies: all features, variance+PSO, MI+PSO;
- models: regularized logistic regression and tuned tree model.

Select a configuration using mean performance, uncertainty, and feature-selection stability. A small performance difference with much better stability and external compatibility should be preferred.

### Priority 5: Calibration and clinical utility

Calibrate the final probabilities using training folds only. Report calibration curves, Brier score, calibration intercept/slope, and decision-curve analysis. Select a threshold according to the clinical use case rather than maximizing accuracy on the test set.

### Priority 6: Reproducibility and reporting

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
10. Feature-selection stability plot (produced: `core/outputs/figures/feature_stability.png`, Step 7).
11. SHAP summary for the final locked model.
12. Supplementary table of selected genes and pathway definitions.

## 12. Change Log

| Date | Change |
|---|---|
| 2026-09-15 | Created manuscript draft from current Late Fusion outputs. |
| 2026-09-15 | Documented the controlled PSO target-30 experiment and restored the default target to 40. |
| 2026-09-15 | Identified the external common-feature limitation and training-based fusion-weight optimism. |
| 2026-09-15 | Added OOF fusion weight/threshold estimation and recorded the first corrected internal test result. |
| 2026-09-15 | Added five-fold nested feature selection with OOF fusion and recorded the first corrected OOF metrics. |
| 2026-09-18 | Built the audited MSKCC 2010 cohort (131 samples, 27 events) and the six transferable features with hard validation gates. |
| 2026-09-18 | Retrained the late fusion on transferable features and applied it frozen to MSKCC (ROC-AUC 0.711; genomic branch collapsed under the raw scaler). |
| 2026-09-18 | Prespecified rank-based genomic transfer, executed once: external fusion 0.717, genomic branch 0.586, platform artifact removed by construction. |
| 2026-09-18 | Added the pure-clinical baseline, paired bootstrap comparisons, calibration, and DCA: fusion vs baseline statistically indistinguishable; clinical-centric narrative locked. |
| 2026-09-19 | Repeated nested CV from raw TCGA (5 folds x 3 repeats, fully fold-local): mean OOF AUC 0.7551 +/- 0.0199 vs 0.861 artifact-based; PSO selections logged for stability analysis. |
| 2026-09-19 | Found and fixed a branch-mask bug (45 one-hot clinical columns leaked into the genomic branch); canonical `identify_column_groups` masks + consistency gate. Corrected internal estimate 0.7819 +/- 0.0203 with clinical-leaning OOF weights. |
| 2026-09-19 | Added feature-selection stability analysis (35 prespecified PSO events): genomic selection unstable (max 0.31), clinical branch stable (top feature 1.00); no genomic feature reaches the frozen 0.70 threshold. |

## 13. Source Artifacts

- Configuration: `core/config.py`
- Feature selection: `core/src/genomic_selector.py`, `core/src/clinical_selector.py`
- Clinical benchmark: `core/src/clinical_benchmark.py`
- Late fusion: `core/src/fusion/late_fusion.py`
- OOF fusion utility: `core/src/fusion/late_fusion.py` (`estimate_oof_fusion`)
- External feature alignment: `core/src/validation.py` (`align_common_features`)
- Current internal metrics: `core/outputs/tables/final_evaluation.json`
- OOF fusion metrics: `core/outputs/tables/oof_fusion_results.json`
- Nested fusion implementation: `core/src/fusion/nested_evaluation.py`
- Nested fusion metrics: `core/outputs/tables/nested_late_fusion_results.json`
- Repeated nested CV from raw data: `core/src/nested_cv_raw.py`, `core/outputs/tables/nested_cv_raw_results.json`, `core/outputs/tables/nested_cv_raw_folds.csv`, `core/outputs/tables/nested_cv_raw_oof_predictions.csv`, `core/outputs/tables/nested_cv_raw_pso_selections.csv`
- Feature-selection stability: `core/src/stability_selection.py`, `core/outputs/tables/stability_summary.json`, `core/outputs/tables/stability_genomic.csv`, `core/outputs/tables/stability_clinical.csv`, `core/outputs/tables/stability_fresh_runs.csv`, `core/outputs/figures/feature_stability.png`
- Clinical benchmark results: `core/outputs/tables/clinical_strategy_benchmark_summary.csv`
- PSO target-30 experiment: `core/outputs/tables/experiment_k30_results.json`
- External predictions: `core/outputs/tables/external_validation_results.csv`
- MSKCC cohort: `core/src/mskcc_cohort.py`, `core/data/external/mskcc_cohort.csv`, `core/outputs/tables/mskcc_cohort_summary.json`
- Transferable features: `core/src/transferable_features.py`, `core/outputs/tables/{tcga,mskcc}_transferable_features_summary.json`
- Transferable fusion (raw scaler): `core/src/transferable_fusion.py`, `core/outputs/tables/transferable_fusion_results.json`
- Rank-based transfer: `core/src/rank_transfer_fusion.py`, `core/outputs/tables/rank_transfer_fusion_results.json`
- Clinical-value diagnostics: `core/src/clinical_value_diagnostics.py`, `core/outputs/tables/clinical_value_diagnostics.json`, `core/outputs/tables/mskcc_dca_curves.csv`, `core/outputs/figures/mskcc_calibration_dca.png`

## 14. Final Pre-Submission Checklist

- [ ] Resolve the discrepancy between README results and current Late Fusion results.
- [ ] Rerun all final notebooks from a clean kernel.
- [x] Confirm every preprocessing and selection step is fold-local.
- [x] Replace training-based fusion weights with OOF weights.
- [x] Choose threshold without touching the final test set.
- [x] Rebuild external validation using transferable prespecified features (MSKCC 2010, five-step pipeline).
- [x] Add 95% confidence intervals and statistical comparisons (external paired bootstrap).
- [x] Add calibration and decision-curve analysis (MSKCC external).
- [x] Report class counts, missingness, and cohort flow (MSKCC filter audit).
- [x] Run repeated nested CV from raw TCGA data as the final internal estimate (mean OOF AUC 0.7819 +/- 0.0203, 5 folds x 3 repeats).
- [x] Quantify feature-selection stability (35 PSO events, prespecified 0.70 threshold) and report the unstable genomic selection honestly.
- [ ] Verify the temporal provenance of the most stable clinical features (Person Neoplasm Status, Year of diagnosis) against the TCGA data dictionary.
- [ ] Lock code, environment, seeds, and artifact hashes.
- [ ] Complete TRIPOD-AI reporting items and journal-specific requirements.
