# Aura-Twin Lite: Uncovering Gender Bias in Alzheimer's Diagnosis

> A comparative study demonstrating how male-normed clinical AI systematically under-detects Alzheimer's risk in women — and a sex-calibrated ensemble model that corrects it.

---

## The Problem

Women represent nearly **two-thirds of all Alzheimer's patients**, yet most clinical diagnostic models are trained on male-dominant datasets with male-normed baselines. This creates three structural blind spots:

- **Cognitive masking** — women's higher verbal memory baselines hide early decline from standard scoring
- **APOE-ε4 underweighting** — genetic risk interacts differently with female hormonal biology, a signal traditional models miss
- **Social determinants** — caregiving stress, sleep disruption, and social isolation disproportionately affect women and accelerate neurodegeneration, but are absent from traditional risk models

This project makes that bias **empirically measurable.**

---

## Key Finding

| Metric | Traditional Model | Aura-Twin Ensemble |
|---|---|---|
| Female Recall | 0.458 | 0.924 |
| Female Precision | 1.000 | 0.968 |
| **Female False Negative Rate** | **0.542** | **0.076** |

**The traditional model misses 54% of women who actually have Alzheimer's.**

A false negative here is not a statistic — it is a woman sent home undiagnosed, losing her window for early intervention. Aura-Twin reduces that miss rate to 7.6% while maintaining near-perfect precision.

---

## How It Works

Two Random Forest models are trained on the same dataset — but with fundamentally different approaches:

### Traditional Model (Baseline)
- Raw clinical features as-is
- No sex-specific adjustment
- Reflects standard clinical AI practice today

### Aura-Twin Ensemble
Four specialist Random Forest models, one per hormonal subgroup, each trained on a feature-enriched dataset:

| Subgroup | Patients Routed |
|---|---|
| Male | All male patients |
| Pre-menopausal | Youngest tertile of female patients |
| Peri-menopausal | Middle tertile of female patients |
| Post-menopausal | Oldest tertile — highest compound risk |

#### 8 Female-Specific Features Engineered

| Feature | Clinical Basis |
|---|---|
| `HormoneStage` | Menopausal status inferred from age distribution |
| `APOE_proxy` | Family history as APOE-ε4 proxy |
| `CognitiveReserveMask` | High verbal baseline masking true decline |
| `CaregiverStressScore` | Chronic caregiver burden proxy |
| `VascularHormonalRisk` | Estrogen loss compounding vascular risk |
| `SocialDeterminantBurden` | Social isolation and access burden |
| `APOE_x_HormoneStage` | Core Aura-Twin interaction term |
| `APOE_x_Vascular` | APOE risk amplified by vascular vulnerability |

#### Aura-Twin Risk Formula
```
Risk = β₁(APOE) + β₂(HormoneStage) + β₃(APOE × HormoneStage)
```

---

## Project Structure

```
aura-twin-lite/
│
├── AuraTwin_Alzheimers.ipynb     # Main research notebook
├── README.md                     # This file
│
└── outputs/
    ├── population_bias_dashboard.png    # 4-panel population visualization
    ├── feature_importance_comparison.png
    └── patient_[n]_bias_signal.png      # Per-patient trajectory plots
```

---

## How to Run

### 1. Get the dataset
Download from Kaggle:
[Alzheimer's Disease Dataset — Rabie El Kharoua](https://www.kaggle.com/datasets/rabieelkharoua/alzheimers-disease-dataset)

### 2. Open the notebook in Google Colab
```
File → Open Notebook → GitHub → paste this repo URL
```
Or upload `AuraTwin_Alzheimers.ipynb` directly to Colab.

### 3. Upload the dataset
In Colab, use the file browser (left sidebar) to upload `alzheimers_disease_data.csv`

### 4. Run all cells in order
```
Runtime → Run All
```

### 5. Interactive patient explorer
At Cell 14, the notebook prompts:
```
Enter patient index (or 'q' to quit): 
```
Enter any number from `0` to `2148` to see that patient's individual bias trajectory.

---

## Requirements

All dependencies are pre-installed in Google Colab. If running locally:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

---

## Visualizations

### Individual Patient Trajectory
For any patient in the dataset, the notebook plots:
- **Curve A** — Traditional model risk probability over 10 simulated years
- **Curve B** — Aura-Twin risk probability over 10 simulated years
- The gap between detection points is the **structural bias signal**

### Population-Level Dashboard (4 panels)
1. Risk score distributions — where the two models disagree across all patients
2. Risk gap by sex — female patients show the largest divergence
3. Bias gap by hormonal stage — post-menopausal women most affected
4. Diagnosis rescue panel — patients caught by Aura-Twin, missed by Traditional

---

## Limitations

This is a research prototype. The following limitations should be acknowledged before any clinical use:

- **No APOE genotype data** — `FamilyHistoryAlzheimers` is used as a proxy; prospective studies with genetic data are needed to validate
- **HormoneStage is inferred** — derived from age distribution, not measured hormone levels (estradiol, FSH)
- **Cross-sectional dataset** — patient trajectories are simulated, not longitudinal observations
- **Relabeling heuristic** — Aura-Twin's adjusted ground truth labels are a modeled correction, not a clinical validation
- **Single dataset** — findings require replication on ADNI or similar longitudinal cohorts with sex-stratified biomarkers

---

## Future Work

- Validate on ADNI dataset which includes APOE genotype and longitudinal follow-up
- Replace age-inferred HormoneStage with measured hormone levels
- Extend subgroup framework to racial and socioeconomic underrepresentation
- Apply methodology to other female-underrepresented conditions (cardiovascular disease, autoimmune disorders)
- Submit findings to a women's health or clinical AI journal

---

## Dataset Credit

Rabie El Kharoua. *Alzheimer's Disease Dataset.* Kaggle, 2024.
[https://www.kaggle.com/datasets/rabieelkharoua/alzheimers-disease-dataset](https://www.kaggle.com/datasets/rabieelkharoua/alzheimers-disease-dataset)

---

## Authors

Built as part of the Aura-Twin research initiative — developing sex-calibrated clinical AI to address representation bias in neurodegenerative disease diagnosis.

---

## License

MIT License — free to use, adapt, and build upon with attribution.
