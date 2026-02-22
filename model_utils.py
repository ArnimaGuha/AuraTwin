import joblib
import numpy as np
import pandas as pd

# ── Load all saved models ─────────────────────────────────────────
rf_traditional  = joblib.load('models/rf_traditional.pkl')
rf_aura         = joblib.load('models/rf_aura.pkl')
subgroup_models = joblib.load('models/subgroup_models.pkl')
scaler          = joblib.load('models/scaler.pkl')
scaler2         = joblib.load('models/scaler2.pkl')
trad_columns    = joblib.load('models/trad_columns.pkl')
aura_columns    = joblib.load('models/aura_columns.pkl')
results_df      = pd.read_csv('models/results_df.csv')

# ── HormoneStage assignment ───────────────────────────────────────
# Thresholds must match what was used during training
# These are recalculated here from results_df
_female_probs = results_df[results_df['Gender'] == 1]['Age']
Q33 = _female_probs.quantile(0.33)
Q66 = _female_probs.quantile(0.66)

def assign_hormone_stage(gender, age):
    if gender == 0:
        return 0
    elif age < Q33:
        return 1
    elif age <= Q66:
        return 2
    else:
        return 3


def engineer_aura_features(p: dict) -> dict:
    """
    Takes a patient dict with raw clinical features,
    adds all Aura-Twin engineered features and returns full dict.
    """
    p = p.copy()

    p['HormoneStage']  = assign_hormone_stage(p['Gender'], p['Age'])
    p['APOE_proxy']    = p['FamilyHistoryAlzheimers']

    # Cognitive reserve mask
    p['CognitiveReserveMask'] = (
        (p['MMSE'] - 24) * 0.4
        if (p['Gender'] == 1 and p['Age'] > Q33
            and p['EducationLevel'] >= 2 and p['MMSE'] >= 24)
        else 0
    )

    # Caregiver stress
    p['CaregiverStressScore'] = (
        (p['Depression'] * 2.5) +
        (np.clip(10 - p['SleepQuality'], 0, 10) * 0.8) +
        (p['BehavioralProblems'] * 1.2)
        if p['Gender'] == 1 else 0
    )

    # Vascular hormonal risk
    p['VascularHormonalRisk'] = (
        p['HormoneStage'] *
        (max(0, p['SystolicBP'] - 120) / 20 +
         max(0, p['CholesterolLDL'] - 100) / 40)
    )

    # Social determinant burden
    p['SocialDeterminantBurden'] = (
        ((10 - p['PhysicalActivity']) * 0.5) +
        ((10 - p['DietQuality']) * 0.5) +
        (p['Depression'] * 2.0)
        if p['Gender'] == 1 else 0
    )

    # Interaction terms
    p['APOE_x_HormoneStage'] = p['APOE_proxy'] * p['HormoneStage']
    p['APOE_x_Vascular']     = p['APOE_proxy'] * p['VascularHormonalRisk']

    return p


def predict_single(patient_dict: dict) -> tuple:
    """
    Given a raw patient dict, returns (trad_prob, aura_prob).
    """
    p = engineer_aura_features(patient_dict)

    trad_row = pd.DataFrame([{c: p[c] for c in trad_columns}])
    trad_sc  = scaler.transform(trad_row)
    trad_prob = rf_traditional.predict_proba(trad_sc)[0][1]

    aura_row = pd.DataFrame([{c: p[c] for c in aura_columns}])
    aura_sc  = scaler2.transform(aura_row)

    # Route to subgroup specialist
    stage = p['HormoneStage']
    group = {0: 'male', 1: 'pre_meno',
             2: 'peri_meno', 3: 'post_meno'}.get(stage, 'male')

    if group in subgroup_models:
        aura_prob = subgroup_models[group].predict_proba(aura_sc)[0][1]
    else:
        aura_prob = rf_aura.predict_proba(aura_sc)[0][1]

    return trad_prob, aura_prob


def simulate_trajectory(patient_dict: dict, years: int = 10) -> tuple:
    """
    Simulates year-by-year risk probabilities for both models.
    Returns (probs_trad, probs_aura) as lists of length years+1.
    """
    base = patient_dict.copy()
    probs_trad = []
    probs_aura = []

    for year in range(years + 1):
        p = base.copy()
        p['Age']                     = base['Age'] + year
        p['MMSE']                    = max(0, base['MMSE'] - year * 1.2)
        p['FunctionalAssessment']    = max(0, base['FunctionalAssessment'] - year * 0.6)
        p['ADL']                     = max(0, base['ADL'] - year * 0.5)
        p['SystolicBP']              = base['SystolicBP'] + year * 1.5
        p['CholesterolLDL']          = base['CholesterolLDL'] + year * 2.0
        p['CholesterolTriglycerides']= base['CholesterolTriglycerides'] + year * 3.0

        if year >= 1: p['MemoryComplaints']          = 1
        if year >= 2: p['DifficultyCompletingTasks'] = 1
        if year >= 3: p['Confusion']                 = 1
        if year >= 4: p['Disorientation']            = 1
        if year >= 5: p['PersonalityChanges']        = 1
        if year >= 5: p['BehavioralProblems']        = 1

        t, a = predict_single(p)
        probs_trad.append(t)
        probs_aura.append(a)

    return probs_trad, probs_aura