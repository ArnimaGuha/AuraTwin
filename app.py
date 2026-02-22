import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from model_utils import (
    results_df, predict_single, simulate_trajectory,
    assign_hormone_stage, Q33, Q66
)

# ── Page config ───────────────────────────────────────────────────
st.set_page_config(
    page_title="Aura-Twin Lite",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Global styles ─────────────────────────────────────────────────
st.markdown("""
<style>
    body, .stApp { background-color: #0f0f1a; color: #f0f0f0; }
    .metric-card {
        background: #1a1a2e;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        border: 1px solid #2a2a4a;
    }
    .metric-value { font-size: 2.4rem; font-weight: 800; }
    .metric-label { font-size: 0.85rem; color: #aaa; margin-top: 4px; }
    .finding-box {
        background: #1a1a2e;
        border-left: 4px solid #f97316;
        padding: 16px 20px;
        border-radius: 8px;
        margin: 12px 0;
    }
    .section-divider { border-top: 1px solid #2a2a4a; margin: 32px 0; }
    h1, h2, h3 { color: #f0f0f0 !important; }
    .stSelectbox label, .stSlider label,
    .stNumberInput label { color: #ccc !important; }
    .stTabs [data-baseweb="tab"] { color: #aaa; }
    .stTabs [aria-selected="true"] { color: #f97316 !important; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar navigation ────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/fluency/96/brain.png", width=60)
st.sidebar.markdown("## Aura-Twin Lite")
st.sidebar.markdown("*Sex-Calibrated Alzheimer's Risk*")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    ["Home — Key Findings",
     "Patient Risk Explorer",
     "Live Patient Assessment"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
**Dataset**
Rabie El Kharoua
Alzheimer's Disease Dataset
Kaggle, 2024 · n=2,149

**Models**
Random Forest (Traditional)
Aura-Twin Ensemble (4 subgroups)

**Validation**
5-Fold Cross-Validation
Chi-squared p < 0.000001
""")

# ═══════════════════════════════════════════════════════════════════
# PAGE 1 — HOME
# ═══════════════════════════════════════════════════════════════════
if page == "Home — Key Findings":

    st.markdown("# Aura-Twin Lite")
    st.markdown("### Uncovering Gender Bias in Alzheimer's Diagnosis")
    st.markdown("""
    Women represent **two-thirds of all Alzheimer's patients** — yet most clinical
    AI models are trained on male-dominant datasets with male-normed baselines.
    This project makes that bias empirically measurable and demonstrates a correction.
    """)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ── Key metrics row ───────────────────────────────────────────
    st.markdown("## The Numbers")
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value" style="color:#ef4444">54.2%</div>
            <div class="metric-label">Traditional Model<br>Female False Negative Rate</div>
        </div>""", unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value" style="color:#4ade80">7.6%</div>
            <div class="metric-label">Aura-Twin<br>Female False Negative Rate</div>
        </div>""", unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value" style="color:#f97316">94.9%</div>
            <div class="metric-label">Aura-Twin<br>Female Recall (5-fold CV)</div>
        </div>""", unsafe_allow_html=True)

    with c4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value" style="color:#c084fc">p≈0</div>
            <div class="metric-label">Statistical Significance<br>χ² = 65.13</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ── Cross-validated results chart ─────────────────────────────
    st.markdown("## Cross-Validated Recall by Sex")

    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor('#0f0f1a')
    ax.set_facecolor('#1a1a2e')

    categories  = ['Traditional\n(Males)', 'Traditional\n(Females)', 'Aura-Twin\n(Females)']
    means       = [0.834, 0.783, 0.949]
    errors      = [0.023, 0.037, 0.018]
    colors      = ['#34d399', '#ef4444', '#f97316']

    bars = ax.bar(categories, means, color=colors, alpha=0.85,
                  width=0.45, edgecolor='none', yerr=errors,
                  capsize=6, error_kw={'color': 'white', 'linewidth': 2})

    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.015,
                f'{mean:.3f}', ha='center', va='bottom',
                color='white', fontsize=13, fontweight='bold')

    ax.axhline(0.783, color='#ef4444', linestyle='--',
               linewidth=1.2, alpha=0.5, label='Traditional female baseline')
    ax.set_ylim(0.6, 1.05)
    ax.set_ylabel('Recall (Sensitivity)', color='white', fontsize=11)
    ax.tick_params(colors='white', labelsize=11)
    ax.spines[['top', 'right']].set_visible(False)
    for s in ['left', 'bottom']: ax.spines[s].set_color('#444')
    ax.grid(True, alpha=0.1, color='white', axis='y')
    ax.legend(facecolor='#1a1a2e', edgecolor='#444',
              labelcolor='white', fontsize=9)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ── Three blind spots ─────────────────────────────────────────
    st.markdown("## Why Traditional Models Fail Women")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="finding-box">
        <b>Cognitive Masking</b><br><br>
        Women's higher verbal memory baseline causes standard MMSE scoring 
        to underestimate decline. A woman scoring 26 may be experiencing 
        significant neurodegeneration invisible to a male-normed model.
        </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="finding-box">
        <b>APOE-ε4 Interaction</b><br><br>
        The APOE-ε4 genetic risk factor interacts with estrogen loss 
        post-menopause — amplifying Alzheimer's risk in ways not captured 
        when hormonal stage is absent from the model.
        </div>""", unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="finding-box">
        <b>Social Determinants</b><br><br>
        Women disproportionately carry unpaid caregiving burdens. 
        Chronic stress, sleep disruption, and social isolation 
        accelerate neurodegeneration — but appear in no traditional 
        clinical feature set.
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ── Aura-Twin methodology ─────────────────────────────────────
    st.markdown("## How Aura-Twin Corrects This")
    st.markdown("""
    Aura-Twin trains **four specialist models**, one per hormonal subgroup,
    on a feature set enriched with 8 female-specific clinical signals:
    """)

    method_df = pd.DataFrame({
        'Feature': ['HormoneStage', 'APOE_proxy', 'CognitiveReserveMask',
                    'CaregiverStressScore', 'VascularHormonalRisk',
                    'SocialDeterminantBurden', 'APOE × HormoneStage',
                    'APOE × Vascular'],
        'Clinical Basis': [
            'Menopausal status inferred from age distribution',
            'Family history as APOE-ε4 carrier proxy',
            'High education + MMSE masking true cognitive deficit',
            'Depression + sleep loss + behavioral burden proxy',
            'Estrogen loss compounding cardiovascular risk',
            'Activity, diet, depression as social burden proxy',
            'Core Aura-Twin interaction term',
            'APOE risk amplified by vascular vulnerability'
        ],
        'Applied To': [
            'All patients', 'All patients', 'Females only',
            'Females only', 'All patients', 'Females only',
            'All patients', 'All patients'
        ]
    })
    st.dataframe(method_df, use_container_width=True, hide_index=True)

    st.markdown("""
    **Risk Formula:**
    ```
    Risk = β₁(APOE) + β₂(HormoneStage) + β₃(APOE × HormoneStage)
    ```
    """)


# ═══════════════════════════════════════════════════════════════════
# PAGE 2 — PATIENT RISK EXPLORER
# ═══════════════════════════════════════════════════════════════════
elif page == "Patient Risk Explorer":

    st.markdown("# Patient Risk Explorer")
    st.markdown("""
    Select any patient from the dataset by index to see how the Traditional 
    and Aura-Twin models diverge in their 10-year risk trajectory.
    The gap between curves is the **structural bias signal** made visible.
    """)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    col_input, col_info = st.columns([1, 2])

    with col_input:
        patient_idx = st.number_input(
            "Patient Index",
            min_value=0,
            max_value=len(results_df) - 1,
            value=0,
            step=1,
            help=f"Choose a patient from 0 to {len(results_df)-1}"
        )
        threshold = st.slider(
            "Detection Threshold",
            min_value=0.3, max_value=0.8,
            value=0.5, step=0.05,
            help="Risk probability at which a model flags a patient as at-risk"
        )
        years = st.slider("Simulation Years", 5, 15, 10)
        run_btn = st.button("Run Trajectory", type="primary",
                            use_container_width=True)

    with col_info:
        row = results_df.iloc[patient_idx]
        gender_label = "Female" if row['Gender'] == 1 else "Male"
        stage_map = {0: "N/A (Male)", 1: "Pre-menopausal",
                     2: "Peri-menopausal", 3: "Post-menopausal"}
        stage_label = stage_map.get(int(row['HormoneStage']), "Unknown")
        diagnosis_label = "Positive" if row['ActualDiagnosis'] == 1 else "Negative"
        diagnosis_color = "#ef4444" if row['ActualDiagnosis'] == 1 else "#4ade80"

        st.markdown(f"""
        <div class="metric-card" style="text-align:left; padding: 20px;">
            <b style="font-size:1.1rem">Patient #{patient_idx}</b><br><br>
            <table style="width:100%; color:#ccc; font-size:0.9rem">
                <tr><td>Age</td><td><b style="color:white">{int(row['Age'])}</b></td></tr>
                <tr><td>Sex</td><td><b style="color:white">{gender_label}</b></td></tr>
                <tr><td>Hormonal Stage</td><td><b style="color:#c084fc">{stage_label}</b></td></tr>
                <tr><td>APOE Risk</td><td><b style="color:white">{"Yes" if row['APOE_proxy'] == 1 else "No"}</b></td></tr>
                <tr><td>Actual Diagnosis</td>
                    <td><b style="color:{diagnosis_color}">{diagnosis_label}</b></td></tr>
                <tr><td>Traditional Prob (Y0)</td>
                    <td><b style="color:#4e9af1">{row['Prob_Traditional']:.3f}</b></td></tr>
                <tr><td>Aura-Twin Prob (Y0)</td>
                    <td><b style="color:#f97316">{row['Prob_Aura']:.3f}</b></td></tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

    if run_btn:
        with st.spinner("Simulating trajectory..."):
            # Build patient dict from results_df row
            # We need all clinical features — load from results_df
            p_row = results_df.iloc[patient_idx]
            patient_dict = p_row.drop(['PatientIdx', 'ActualDiagnosis',
                                       'Prob_Traditional', 'Prob_Aura',
                                       'RiskGap', 'Pred_Traditional',
                                       'Pred_Aura', 'AuraCatchesMissed',
                                       'HormoneStage', 'APOE_proxy',
                                       'APOE_x_HormoneStage',
                                       'APOE_x_Vascular'],
                                      errors='ignore').to_dict()

            probs_trad, probs_aura = simulate_trajectory(patient_dict, years)
            year_range = list(range(years + 1))

            def detect_year(probs, thr):
                for yr, p in enumerate(probs):
                    if p >= thr:
                        return yr
                return None

            yr_trad = detect_year(probs_trad, threshold)
            yr_aura = detect_year(probs_aura, threshold)

        # ── Trajectory plot ───────────────────────────────────────
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        fig.patch.set_facecolor('#0f0f1a')
        for ax in [ax1, ax2]:
            ax.set_facecolor('#1a1a2e')

        ax1.plot(year_range, probs_trad, color='#4e9af1', linewidth=2.5,
                 marker='o', markersize=5,
                 label='Traditional Model (Male-Normed)')
        ax1.plot(year_range, probs_aura, color='#f97316', linewidth=2.5,
                 marker='s', markersize=5,
                 label='Aura-Twin (Sex-Calibrated)')
        ax1.axhline(threshold, color='#ef4444', linestyle='--',
                    linewidth=1.5, alpha=0.8,
                    label=f'Threshold ({threshold})')

        if yr_trad is not None:
            ax1.axvline(yr_trad, color='#4e9af1', linestyle=':',
                        linewidth=1.5, alpha=0.7)
            ax1.annotate(f'Traditional\nYear {yr_trad}',
                         xy=(yr_trad, threshold),
                         xytext=(min(yr_trad + 0.5, years - 1), threshold + 0.12),
                         color='#4e9af1', fontsize=9, fontweight='bold',
                         arrowprops=dict(arrowstyle='->', color='#4e9af1'))

        if yr_aura is not None:
            ax1.axvline(yr_aura, color='#f97316', linestyle=':',
                        linewidth=1.5, alpha=0.7)
            ax1.annotate(f'Aura-Twin\nYear {yr_aura}',
                         xy=(yr_aura, threshold),
                         xytext=(max(yr_aura - 2.5, 0.2), threshold - 0.18),
                         color='#f97316', fontsize=9, fontweight='bold',
                         arrowprops=dict(arrowstyle='->', color='#f97316'))

        if yr_trad and yr_aura and yr_trad != yr_aura:
            lo, hi = min(yr_trad, yr_aura), max(yr_trad, yr_aura)
            ax1.axvspan(lo, hi, alpha=0.10, color='white',
                        label=f'Bias Window: {abs(yr_trad - yr_aura)} yr(s)')

        ax1.set_xlim(0, years)
        ax1.set_ylim(-0.02, 1.08)
        ax1.set_xlabel('Years from Baseline', color='white', fontsize=11)
        ax1.set_ylabel('Predicted Risk Probability', color='white', fontsize=11)
        ax1.set_title(f'Patient #{patient_idx} — Risk Trajectory',
                      color='white', fontsize=12, fontweight='bold')
        ax1.tick_params(colors='white')
        ax1.spines[['top', 'right']].set_visible(False)
        for s in ['left', 'bottom']: ax1.spines[s].set_color('#444')
        ax1.legend(facecolor='#0f0f1a', edgecolor='#444',
                   labelcolor='white', fontsize=9)
        ax1.grid(True, alpha=0.12, color='white')

        # Annual gap bars
        gaps = [pa - pt for pa, pt in zip(probs_aura, probs_trad)]
        bar_colors = ['#f97316' if g > 0 else '#4e9af1' for g in gaps]
        ax2.bar(year_range, gaps, color=bar_colors,
                alpha=0.85, width=0.6, edgecolor='none')
        ax2.axhline(0, color='white', linewidth=0.8, alpha=0.4)
        ax2.set_xlabel('Year', color='white', fontsize=11)
        ax2.set_ylabel('Risk Gap (Aura - Traditional)',
                       color='white', fontsize=11)
        ax2.set_title('Annual Bias Magnitude',
                      color='white', fontsize=12, fontweight='bold')
        ax2.tick_params(colors='white')
        ax2.spines[['top', 'right']].set_visible(False)
        for s in ['left', 'bottom']: ax2.spines[s].set_color('#444')
        ax2.grid(True, alpha=0.12, color='white', axis='y')

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # ── Detection summary ─────────────────────────────────────
        if yr_trad is not None and yr_aura is not None:
            delay = yr_trad - yr_aura
            if delay > 0:
                st.error(f"Bias detected — Aura-Twin flags risk {delay} year(s) earlier than Traditional (Year {yr_aura} vs Year {yr_trad})")
            elif delay == 0:
                st.success(f"Both models detect risk at Year {yr_trad} — no detection delay for this patient")
            else:
                st.info(f"Traditional detects {abs(delay)} year(s) earlier for this patient")
        elif yr_aura is not None and yr_trad is None:
            st.error(f"Traditional model never flags this patient — Aura-Twin detects risk at Year {yr_aura}")
        elif yr_trad is not None and yr_aura is None:
            st.info(f"Aura-Twin does not flag this patient — Traditional detects at Year {yr_trad}")
        else:
            st.warning("Neither model crosses the detection threshold in this simulation window")


# ═══════════════════════════════════════════════════════════════════
# PAGE 3 — LIVE PATIENT ASSESSMENT
# ═══════════════════════════════════════════════════════════════════
elif page == "Live Patient Assessment":

    st.markdown("# Live Patient Assessment")
    st.markdown("""
    Enter clinical values manually to get an immediate side-by-side 
    risk prediction from both models — and see exactly where they diverge.
    """)
    st.warning("This tool is for research demonstration only — not for clinical use.")

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ── Input form ────────────────────────────────────────────────
    with st.form("patient_form"):
        st.markdown("### Demographics")
        d1, d2, d3, d4 = st.columns(4)
        age             = d1.number_input("Age", 50, 95, 65)
        gender          = d2.selectbox("Sex", ["Female", "Male"])
        ethnicity       = d3.selectbox("Ethnicity", [0, 1, 2, 3])
        education       = d4.selectbox("Education Level", [0, 1, 2, 3])

        st.markdown("### Lifestyle")
        l1, l2, l3, l4, l5, l6 = st.columns(6)
        bmi              = l1.number_input("BMI", 15.0, 45.0, 26.5)
        smoking          = l2.selectbox("Smoking", [0, 1])
        alcohol          = l3.number_input("Alcohol (units/wk)", 0.0, 20.0, 3.0)
        physical         = l4.number_input("Physical Activity", 0.0, 10.0, 5.0)
        diet             = l5.number_input("Diet Quality", 0.0, 10.0, 6.0)
        sleep            = l6.number_input("Sleep Quality", 0.0, 10.0, 6.5)

        st.markdown("### Medical History")
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        family_history   = m1.selectbox("Family History Alzheimer's", [0, 1])
        cardio           = m2.selectbox("Cardiovascular Disease", [0, 1])
        diabetes         = m3.selectbox("Diabetes", [0, 1])
        depression       = m4.selectbox("Depression", [0, 1])
        head_injury      = m5.selectbox("Head Injury", [0, 1])
        hypertension     = m6.selectbox("Hypertension", [0, 1])

        st.markdown("### Clinical Measurements")
        c1, c2, c3, c4, c5 = st.columns(5)
        sbp              = c1.number_input("Systolic BP", 90, 200, 130)
        dbp              = c2.number_input("Diastolic BP", 60, 120, 80)
        chol_total       = c3.number_input("Total Cholesterol", 150, 300, 210)
        chol_ldl         = c4.number_input("LDL", 50, 200, 130)
        chol_hdl         = c5.number_input("HDL", 20, 100, 55)

        c6, c7, c8, c9, c10 = st.columns(5)
        chol_trig        = c6.number_input("Triglycerides", 50, 400, 150)
        mmse             = c7.number_input("MMSE Score", 0, 30, 26)
        func_assess      = c8.number_input("Functional Assessment", 0.0, 10.0, 7.5)
        adl              = c9.number_input("ADL Score", 0.0, 10.0, 8.0)
        memory_comp      = c10.selectbox("Memory Complaints", [0, 1])

        st.markdown("### Symptoms")
        s1, s2, s3, s4, s5 = st.columns(5)
        confusion        = s1.selectbox("Confusion", [0, 1])
        disorientation   = s2.selectbox("Disorientation", [0, 1])
        personality      = s3.selectbox("Personality Changes", [0, 1])
        difficulty       = s4.selectbox("Difficulty w/ Tasks", [0, 1])
        forgetfulness    = s5.selectbox("Forgetfulness", [0, 1])
        behavioral       = st.selectbox("Behavioral Problems", [0, 1])

        submitted = st.form_submit_button("Assess Patient Risk",
                                          type="primary",
                                          use_container_width=True)

    if submitted:
        patient_dict = {
            'Age': age, 'Gender': 1 if gender == "Female" else 0,
            'Ethnicity': ethnicity, 'EducationLevel': education,
            'BMI': bmi, 'Smoking': smoking,
            'AlcoholConsumption': alcohol, 'PhysicalActivity': physical,
            'DietQuality': diet, 'SleepQuality': sleep,
            'FamilyHistoryAlzheimers': family_history,
            'CardiovascularDisease': cardio, 'Diabetes': diabetes,
            'Depression': depression, 'HeadInjury': head_injury,
            'Hypertension': hypertension, 'SystolicBP': sbp,
            'DiastolicBP': dbp, 'CholesterolTotal': chol_total,
            'CholesterolLDL': chol_ldl, 'CholesterolHDL': chol_hdl,
            'CholesterolTriglycerides': chol_trig, 'MMSE': mmse,
            'FunctionalAssessment': func_assess, 'ADL': adl,
            'MemoryComplaints': memory_comp, 'BehavioralProblems': behavioral,
            'Confusion': confusion, 'Disorientation': disorientation,
            'PersonalityChanges': personality,
            'DifficultyCompletingTasks': difficulty,
            'Forgetfulness': forgetfulness,
        }

        with st.spinner("Running both models..."):
            trad_prob, aura_prob = predict_single(patient_dict)
            probs_trad, probs_aura = simulate_trajectory(patient_dict, years=10)

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.markdown("## Results")

        # ── Risk score cards ──────────────────────────────────────
        r1, r2, r3 = st.columns(3)
        trad_color = "#ef4444" if trad_prob >= 0.5 else "#4ade80"
        aura_color = "#ef4444" if aura_prob >= 0.5 else "#4ade80"
        gap_color  = "#f97316" if aura_prob > trad_prob else "#4e9af1"

        r1.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color:{trad_color}">
                {trad_prob:.1%}
            </div>
            <div class="metric-label">Traditional Model Risk<br>
            {"HIGH RISK" if trad_prob >= 0.5 else "LOW RISK"}</div>
        </div>""", unsafe_allow_html=True)

        r2.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color:{aura_color}">
                {aura_prob:.1%}
            </div>
            <div class="metric-label">Aura-Twin Risk<br>
            {"HIGH RISK" if aura_prob >= 0.5 else "LOW RISK"}</div>
        </div>""", unsafe_allow_html=True)

        r3.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color:{gap_color}">
                {aura_prob - trad_prob:+.1%}
            </div>
            <div class="metric-label">Risk Gap<br>
            (Aura-Twin minus Traditional)</div>
        </div>""", unsafe_allow_html=True)

        # ── Hormonal context ──────────────────────────────────────
        if patient_dict['Gender'] == 1:
            hs = assign_hormone_stage(1, age)
            stage_names = {1: "Pre-menopausal", 2: "Peri-menopausal",
                           3: "Post-menopausal"}
            st.info(f"Hormonal Stage: **{stage_names.get(hs, 'Unknown')}** "
                    f"— Aura-Twin routes this patient to the "
                    f"{['pre_meno','peri_meno','post_meno'][hs-1]} specialist model")

        # ── Trajectory chart ──────────────────────────────────────
        st.markdown("### 10-Year Risk Trajectory")
        year_range = list(range(11))

        fig, ax = plt.subplots(figsize=(12, 4))
        fig.patch.set_facecolor('#0f0f1a')
        ax.set_facecolor('#1a1a2e')

        ax.plot(year_range, probs_trad, color='#4e9af1', linewidth=2.5,
                marker='o', markersize=5, label='Traditional Model')
        ax.plot(year_range, probs_aura, color='#f97316', linewidth=2.5,
                marker='s', markersize=5, label='Aura-Twin')
        ax.axhline(0.5, color='#ef4444', linestyle='--',
                   linewidth=1.5, alpha=0.7, label='Risk Threshold (0.5)')
        ax.fill_between(year_range, probs_trad, probs_aura,
                        alpha=0.12, color='#f97316',
                        label='Bias gap')

        ax.set_xlim(0, 10)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel('Years from Baseline', color='white', fontsize=11)
        ax.set_ylabel('Predicted Risk Probability', color='white', fontsize=11)
        ax.set_title('Risk Trajectory — Traditional vs Aura-Twin',
                     color='white', fontsize=12, fontweight='bold')
        ax.tick_params(colors='white')
        ax.spines[['top', 'right']].set_visible(False)
        for s in ['left', 'bottom']: ax.spines[s].set_color('#444')
        ax.legend(facecolor='#0f0f1a', edgecolor='#444',
                  labelcolor='white', fontsize=10)
        ax.grid(True, alpha=0.12, color='white')

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # ── Feature contribution table ────────────────────────────
        if patient_dict['Gender'] == 1:
            st.markdown("### Aura-Twin Female-Specific Risk Factors")
            from model_utils import engineer_aura_features
            enriched = engineer_aura_features(patient_dict)

            factor_df = pd.DataFrame({
                'Feature': ['HormoneStage', 'CognitiveReserveMask',
                            'CaregiverStressScore', 'VascularHormonalRisk',
                            'SocialDeterminantBurden'],
                'Value': [
                    enriched['HormoneStage'],
                    round(enriched['CognitiveReserveMask'], 3),
                    round(enriched['CaregiverStressScore'], 3),
                    round(enriched['VascularHormonalRisk'], 3),
                    round(enriched['SocialDeterminantBurden'], 3)
                ],
                'Flagged': [
                    "Yes" if enriched['HormoneStage'] >= 2 else "No",
                    "Yes" if enriched['CognitiveReserveMask'] > 0 else "No",
                    "Yes" if enriched['CaregiverStressScore'] > 3 else "No",
                    "Yes" if enriched['VascularHormonalRisk'] > 1 else "No",
                    "Yes" if enriched['SocialDeterminantBurden'] > 5 else "No",
                ]
            })
            st.dataframe(factor_df, use_container_width=True, hide_index=True)