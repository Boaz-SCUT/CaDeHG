# CaDe-HG: Causality Debiasing and Pathway Hypergraph Learning for Survival Prediction

Implementation of the paper: CaDe-HG: Causality Debiasing and Pathway Hypergraph Learning for Survival Prediction.

## Overview

CaDe-HG is a multimodal survival prediction framework that integrates:

* **Gene expression profiles**
* **Pathology report texts**

for cancer survival analysis.

The proposed framework addresses shortcut learning and spurious correlations in pathology reports through causality-inspired debiasing, while modeling high-order biological interactions using pathway-guided hypergraph learning on gene expression data.

---
## Dataset Preparation

### 1. Gene Expression Data

Gene expression data can be obtained from the TCGA project.

Experiments are conducted on the following five cancer cohorts:

* BLCA (Bladder Urothelial Carcinoma)
* BRCA (Breast Invasive Carcinoma)
* LUAD (Lung Adenocarcinoma)
* COADREAD (Colon and Rectal Adenocarcinoma)
* STAD (Stomach Adenocarcinoma)

The processed gene expression data can be prepared following the protocol provided by the [SurvPath Repository](https://github.com/mahmoodlab/SurvPath).

### 2. Pathology Report Data

Pathology reports can be obtained from [TCGA Pathology Reports Repository](https://github.com/tatonetti-lab/tcga-path-reports).

Please match pathology reports with corresponding TCGA patients using the patient identifiers.

---

## Training and Evaluation

The final experimental results reported in the paper are obtained using:

```bash
bash scripts/CADEHG.sh
```

This script performs:

* Model training
* Validation
* Five-fold cross-validation evaluation

After training is completed, the corresponding survival prediction results and evaluation metrics will be generated automatically.

---

---

## Acknowledgements

This work builds upon publicly available resources from:

* [SurvPath Repository](https://github.com/mahmoodlab/SurvPath)
* [TCGA Pathology Reports Repository](https://github.com/tatonetti-lab/tcga-path-reports)

We thank the authors of these resources for making their datasets and code publicly available.

---
