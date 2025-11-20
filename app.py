import streamlit as st
import numpy as np
import pandas as pd

# ----------------- PAGE SETUP -----------------
st.set_page_config(
    page_title="Pauschaloption vs EEG – Expert PV+Battery+EV Simulator",
    layout="wide",
)

# ----------------- HELPER FUNCTIONS -----------------


def safe_div(a, b):
    if b == 0:
        return 0.0
    return a / b


def seasonal_pv_generation(
    pv_kwp,
    location_yield,
    module_eff,
    temp_derate,
    system_losses,
    tilt_boost,
    winter_penalty,
):
    """
    Compute monthly + annual PV output for PV1 using a simple seasonal model.

    - pv_kwp: installed kWp
    - location_yield: kWh/kWp/year for the site (before detailed derating)
    - module_eff: panel efficiency (used as relative factor vs 20%)
    - temp_derate: fraction (0–1) reduction due to temperature
    - system_losses: fraction (0–1) reduction due to inverter, wiring, dirt, etc.
    - tilt_boost: fraction (0–1) extra gain on summer months
    - winter_penalty: fraction (0–1) extra loss on winter months
    """

    # Typical German irradiance distribution by month (rough)
    monthly_share = np.array(
        [0.03, 0.04, 0.07, 0.10, 0.12, 0.14, 0.14, 0.13, 0.10, 0.08, 0.03, 0.02]
    )

    # Summer months (Apr–Aug -> indices 3..7)
    summer_idx = [3, 4, 5, 6, 7]
    # Winter months (Nov, Dec, Jan, Feb)
    winter_idx = [10, 11, 0, 1]

    monthly_share_adj = monthly_share.copy()
    monthly_share_adj[summer_idx] *= 1 + tilt_boost
    monthly_share_adj[winter_idx] *= 1 - winter_penalty

    # Normalize back to 1
    monthly_share_adj = monthly_share_adj / monthly_share_adj.sum()

    # Basic yield scaled by relative efficiency and losses
    # Assume 20% eff = reference
    eff_factor = module_eff / 0.20
    effective_yield = (
        location_yield
        * eff_factor
        * (1 - temp_derate)
        * (1 - system_losses)
    )

    annual_gen = pv_kwp * effective_yield
    monthly_gen = annual_gen * monthly_share_adj

    return monthly_gen, annual_gen


def compute_core_flows(params):
    """
    Compute aggregate annual energy flows from inputs.
    """
    gen_total = params["pv1_gen"] + (params["pv2_gen"] if params["multi_pv"] else 0.0)
    total_load = (
        params["household_load"] + params["ev_load"] + params["other_load"]
    )

    if params["override_self_consumption"]:
        self_cons = params["self_consumption_manual"]
    else:
        # self consumption cannot exceed both generation and total load
        self_cons = min(gen_total, total_load)

    grid_import = params["grid_import"]
    grid_feed_in = gen_total - self_cons

    cap_total = (
        params["pv1_kwp"] + (params["pv2_kwp"] if params["multi_pv"] else 0.0)
    ) * params["cap_per_kwp"]

    warnings = []
    if grid_feed_in < -1e-6:
        warnings.append(
            "Grid feed-in (P2) is negative – check generation vs self-consumption."
        )
    if self_cons > total_load + 1e-6:
        warnings.append(
            "Self-consumption exceeds total load – check manual override."
        )
    if gen_total <= 0:
        warnings.append("Total PV generation is zero or negative.")

    return {
        "generation_total": gen_total,
        "total_load": total_load,
        "self_consumption": self_cons,
        "grid_import": grid_import,
        "grid_feed_in": max(grid_feed_in, 0.0),
        "cap_total": cap_total,
        "warnings": warnings,
    }


def compute_pauschal(params, flows):
    """
    Pauschaloption (Annex 2) calculation.
    """
    P2 = flows["grid_feed_in"]
    P3 = flows["cap_total"]
    aw_share = params["aw_share"]
    aw_value_eur = params["aw_ct"] / 100.0
    eeg_eur = params["eeg_ct"] / 100.0
    retail_eur = params["retail_ct"] / 100.0

    # P8 = MIN(P2, P3)
    P8 = min(P2, P3)
    eligible = aw_share * P8  # P11
    pauschal_feed_revenue = eligible * aw_value_eur

    self_save = flows["self_consumption"] * retail_eur
    import_cost = flows["grid_import"] * retail_eur

    net_pauschal = pauschal_feed_revenue + self_save - import_cost

    eeg_feed_revenue = P2 * eeg_eur
    net_eeg = eeg_feed_revenue + self_save - import_cost

    return {
        "P2": P2,
        "P3": P3,
        "P8": P8,
        "eligible": eligible,
        "pauschal_feed_revenue": pauschal_feed_revenue,
        "self_save": self_save,
        "import_cost": import_cost,
        "net_pauschal": net_pauschal,
        "eeg_feed_revenue": eeg_feed_revenue,
        "net_eeg": net_eeg,
        "diff_eeg_minus_pauschal": net_eeg - net_pauschal,
    }


def compute_annex1_storage(params):
    """
    Simplified Annex 1 (A3-like) storage+EV separation.
    """
    charge_total = params["batt_charge"]
    discharge_total = params["batt_discharge"]
    pv_share_charge = params["batt_pv_share"]
    grid_share_charge = 1.0 - pv_share_charge
    discharge_to_grid_share = params["batt_to_grid_share"]
    foreign_charge = params["foreign_charge"]

    v4 = charge_total
    v5 = discharge_total
    v6 = charge_total * grid_share_charge
    v7 = discharge_total * discharge_to_grid_share
    v8 = foreign_charge
    v9 = v4 + v8
    v10 = safe_div(v6, v9)
    v11 = v10 * v7
    v12 = v4 - v5
    if params["privileged_losses"]:
        v13 = v10 * v12
    else:
        v13 = 0.0

    v3 = params["grid_import_for_annex1"]
    v16 = max(v3 - v11 - v13, 0.0)

    return {
        "(3)_grid_import": v3,
        "(4)_cons_storage_ev": v4,
        "(5)_output_storage_ev": v5,
        "(6)_simul_grid_to_storage_ev": v6,
        "(7)_simul_storage_ev_to_grid": v7,
        "(8)_foreign_charge": v8,
        "(9)_total_consumption": v9,
        "(10)_grid_share": v10,
        "(11)_balancing_eligible_feed_in": v11,
        "(12)_losses": v12,
        "(13)_privileged_losses": v13,
        "(16)_chargeable_grid_import": v16,
    }


def compute_sensitivity_table(base_params):
    """
    Sensitivity over PV1 generation and AW_share.
    """
    gen_values = np.linspace(
        base_params["pv1_gen"] * 0.5, base_params["pv1_gen"] * 1.5, 7
    )
    aw_shares = np.linspace(0.5, 1.0, 6)

    rows = []
    for gen in gen_values:
        for share in aw_shares:
            params = base_params.copy()
            params["pv1_gen"] = gen
            params["aw_share"] = share

            flows = compute_core_flows(params)
            res = compute_pauschal(params, flows)
            rows.append(
                {
                    "Generation_kWh": gen,
                    "AW_share": share,
                    "Diff_EEG_minus_Pauschal_€": res[
                        "diff_eeg_minus_pauschal"
                    ],
                }
            )
    df = pd.DataFrame(rows)
    return df.pivot(
        index="Generation_kWh",
        columns="AW_share",
        values="Diff_EEG_minus_Pauschal_€",
    )


# ----------------- SIDEBAR INPUTS -----------------

st.title("🔌 Pauschaloption vs EEG – Expert PV+Battery+EV Simulator")

st.markdown(
    """
This app lets you play with **all key parameters** of a German PV system with
battery & EV and compare:

- **Pauschaloption (Annex 2)** – cap at 500 kWh/kWp, AW>0 filter.  
- **EEG feed-in tariff** – fixed ct/kWh for all feed-in.  
- **Annex 1-style storage separation** – approximate A3 (PV + storage + EV).

The PV generation of **PV1** is now calculated automatically from:
**installed kWp, seasonal sun, module efficiency and losses**.
"""
)

with st.sidebar:
    st.header("PV System Parameters")

    col_pv = st.columns(2)
    with col_pv[0]:
        pv1_kwp = st.number_input(
            "PV1 size (kWp)", 0.1, 1000.0, 9.5, 0.1
        )
    with col_pv[1]:
        multi_pv = st.checkbox("Add PV2 (manual)", value=False)

    if multi_pv:
        pv2_kwp = st.number_input(
            "PV2 size (kWp)", 0.0, 1000.0, 0.0, 0.1
        )
        pv2_gen = st.number_input(
            "PV2 generation (kWh/yr)", 0.0, 100000.0, 0.0, 100.0
        )
    else:
        pv2_kwp = 0.0
        pv2_gen = 0.0

    st.markdown("---")
    st.subheader("PV1 Seasonal Production Model")

    location_yield = st.slider(
        "Location yield (kWh/kWp/year)",
        600,
        1400,
        1050,
        help="Northern DE ~900–1000, Southern DE ~1100–1300",
    )

    module_eff = (
        st.slider(
            "Module efficiency (%)", 15.0, 25.0, 20.0, help="Used relative to 20% baseline"
        )
        / 100.0
    )

    temp_derate = (
        st.slider(
            "Temperature derating (%)",
            0.0,
            15.0,
            8.0,
            help="Overall performance loss due to higher cell temperature",
        )
        / 100.0
    )

    system_losses = (
        st.slider(
            "System losses (%)",
            5.0,
            30.0,
            14.0,
            help="Inverter, wiring, mismatch, dirt, shading",
        )
        / 100.0
    )

    tilt_boost = (
        st.slider(
            "Summer tilt boost (%)",
            -20.0,
            20.0,
            5.0,
            help="If roof tilt favors summer (south-facing), set >0",
        )
        / 100.0
    )

    winter_penalty = (
        st.slider(
            "Winter penalty (%)",
            0.0,
            40.0,
            15.0,
            help="Extra winter loss due to low sun, shading, snow, etc.",
        )
        / 100.0
    )

    st.markdown("---")
    st.subheader("Loads")

    household_load = st.number_input(
        "Household load (kWh/yr)", 0.0, 100000.0, 3500.0, 100.0
    )
    ev_load = st.number_input(
        "EV charging at home (kWh/yr)", 0.0, 100000.0, 2500.0, 100.0
    )
    other_load = st.number_input(
        "Other loads (heat pump, etc.) (kWh/yr)",
        0.0,
        100000.0,
        0.0,
        100.0,
    )

    grid_import = st.number_input(
        "Grid import (kWh/yr)", 0.0, 100000.0, 1000.0, 100.0
    )

    override_self = st.checkbox(
        "Override self-consumption manually?", value=False
    )
    if override_self:
        self_manual = st.number_input(
            "Self-consumption (kWh/yr, manual)", 0.0, 200000.0, 5000.0, 100.0
        )
    else:
        self_manual = 0.0

    st.markdown("---")
    st.subheader("Tariffs & Prices")

    aw_share = st.slider(
        "AW>0 share (0–1)", 0.0, 1.0, 0.82, 0.01
    )
    aw_ct = st.number_input(
        "AW (Anzulegender Wert) (ct/kWh)", 0.0, 50.0, 8.26, 0.01
    )
    eeg_ct = st.number_input(
        "EEG feed-in tariff (ct/kWh)", 0.0, 50.0, 7.86, 0.01
    )
    retail_ct = st.number_input(
        "Retail price (ct/kWh)", 0.0, 100.0, 40.1, 0.1
    )

    st.markdown("**Grid surcharge components (informational):**")
    kwkg_ct = st.number_input(
        "KWKG Umlage (ct/kWh)", 0.0, 10.0, 0.277, 0.001
    )
    offshore_ct = st.number_input(
        "Offshore Umlage (ct/kWh)", 0.0, 10.0, 0.816, 0.001
    )
    stromnev19_ct = st.number_input(
        "§19 StromNEV Umlage (ct/kWh)", 0.0, 10.0, 1.558, 0.001
    )

    st.markdown("---")
    st.subheader("Pauschaloption")

    cap_per_kwp = st.number_input(
        "Cap per kWp (kWh/kWp)", 0.0, 2000.0, 500.0, 10.0
    )

    st.markdown("---")
    st.subheader("Storage & EV (Annex 1-style)")

    batt_charge = st.number_input(
        "Battery annual charge (kWh)", 0.0, 100000.0, 3500.0, 100.0
    )
    batt_discharge = st.number_input(
        "Battery annual discharge (kWh)", 0.0, 100000.0, 3000.0, 100.0
    )
    batt_pv_share = (
        st.slider(
            "Share of battery charge from PV (%)", 0.0, 100.0, 80.0, 1.0
        )
        / 100.0
    )
    batt_to_grid_share = (
        st.slider(
            "Share of battery discharge to grid (%)",
            0.0,
            100.0,
            20.0,
            1.0,
        )
        / 100.0
    )
    foreign_charge = st.number_input(
        "Foreign charge (kWh, external EV charging)",
        0.0,
        100000.0,
        0.0,
        10.0,
    )
    privileged_losses = st.checkbox(
        "Include privileged storage losses (A1/A4)", value=False
    )
    grid_import_for_annex1 = st.number_input(
        "Grid import (kWh) used as (3) in Annex 1",
        0.0,
        100000.0,
        grid_import,
        100.0,
    )

# ----------------- SEASONAL PV CALCULATION -----------------

monthly_gen_pv1, auto_pv1_gen = seasonal_pv_generation(
    pv_kwp=pv1_kwp,
    location_yield=location_yield,
    module_eff=module_eff,
    temp_derate=temp_derate,
    system_losses=system_losses,
    tilt_boost=tilt_boost,
    winter_penalty=winter_penalty,
)

# ----------------- PACK PARAMS & COMPUTE -----------------

params = {
    "pv1_kwp": pv1_kwp,
    "pv1_gen": auto_pv1_gen,
    "multi_pv": multi_pv,
    "pv2_kwp": pv2_kwp,
    "pv2_gen": pv2_gen,
    "household_load": household_load,
    "ev_load": ev_load,
    "other_load": other_load,
    "grid_import": grid_import,
    "override_self_consumption": override_self,
    "self_consumption_manual": self_manual,
    "aw_share": aw_share,
    "aw_ct": aw_ct,
    "eeg_ct": eeg_ct,
    "retail_ct": retail_ct,
    "cap_per_kwp": cap_per_kwp,
    "batt_charge": batt_charge,
    "batt_discharge": batt_discharge,
    "batt_pv_share": batt_pv_share,
    "batt_to_grid_share": batt_to_grid_share,
    "foreign_charge": foreign_charge,
    "privileged_losses": privileged_losses,
    "grid_import_for_annex1": grid_import_for_annex1,
}

flows = compute_core_flows(params)
pauschal = compute_pauschal(params, flows)
annex1 = compute_annex1_storage(params)
sens_df = compute_sensitivity_table(params)

# ----------------- WARNINGS -----------------
if flows["warnings"]:
    st.warning(" | ".join(flows["warnings"]))

# ----------------- TABS -----------------
tab1, tab2, tab3 = st.tabs(
    ["📊 Core Results", "⚙️ Annex 1 – Storage Separation", "📈 Sensitivity"]
)

# -------- Tab 1: Core Results --------
with tab1:
    st.subheader("Core Energy Balance")

    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric(
        "Total PV generation (kWh)", f"{flows['generation_total']:.0f}"
    )
    col_b.metric("Total load (kWh)", f"{flows['total_load']:.0f}")
    col_c.metric(
        "Self-consumption (kWh)", f"{flows['self_consumption']:.0f}"
    )
    col_d.metric("Grid feed-in P2 (kWh)", f"{pauschal['P2']:.0f}")

    st.markdown("### PV1 Monthly Production (auto-computed)")
    df_month = pd.DataFrame(
        {
            "Month": [
                "Jan",
                "Feb",
                "Mar",
                "Apr",
                "May",
                "Jun",
                "Jul",
                "Aug",
                "Sep",
                "Oct",
                "Nov",
                "Dec",
            ],
            "PV1_kWh": monthly_gen_pv1,
        }
    )
    df_month_display = df_month.copy()
    df_month_display["PV1_kWh"] = df_month_display["PV1_kWh"].round(0)
    st.bar_chart(df_month.set_index("Month"))

    st.caption(
        f"Auto annual PV1 generation = **{auto_pv1_gen:.0f} kWh** "
        f"(= {pv1_kwp:.1f} kWp × {location_yield} kWh/kWp/year with losses & seasonal factors)."
    )

    st.markdown("---")
    st.subheader("Pauschal vs EEG – Financial Comparison")

    col_fin1, col_fin2 = st.columns(2)
    with col_fin1:
        st.markdown("### Pauschaloption")
        st.write(
            f"- Feed-in revenue: **{pauschal['pauschal_feed_revenue']:.2f} €**"
        )
        st.write(
            f"- Self-consumption savings: **{pauschal['self_save']:.2f} €**"
        )
        st.write(
            f"- Import cost: **{pauschal['import_cost']:.2f} €**"
        )
        st.write(
            f"👉 **Net Pauschal:** **{pauschal['net_pauschal']:.2f} € / year**"
        )

    with col_fin2:
        st.markdown("### EEG (feed-in tariff)")
        st.write(
            f"- Feed-in revenue: **{pauschal['eeg_feed_revenue']:.2f} €**"
        )
        st.write(
            f"- Self-consumption savings: **{pauschal['self_save']:.2f} €**"
        )
        st.write(
            f"- Import cost: **{pauschal['import_cost']:.2f} €**"
        )
        st.write(
            f"👉 **Net EEG:** **{pauschal['net_eeg']:.2f} € / year**"
        )

    diff = pauschal["diff_eeg_minus_pauschal"]
    if diff > 0.5:
        st.success(
            f"EEG is better by **{diff:.2f} € / year** (EEG − Pauschal)."
        )
    elif diff < -0.5:
        st.success(
            f"Pauschaloption is better by **{-diff:.2f} € / year**."
        )
    else:
        st.info(
            f"Both options are almost identical (difference = {diff:.2f} € / year)."
        )

    st.markdown("---")
    st.subheader("AW-share Threshold (for P2 ≤ P3)")

    if params["aw_ct"] > 0:
        aw_thresh = safe_div(params["eeg_ct"], params["aw_ct"])
        st.write(
            f"If P2 ≤ P3, Pauschal is better per kWh when **AW_share > EEG/AW**."
        )
        st.write(
            f"Current threshold: EEG/AW ≈ **{aw_thresh*100:.1f} %**, "
            f"your AW_share = **{params['aw_share']*100:.1f} %**."
        )
    else:
        st.write("AW is zero → threshold undefined.")

    with st.expander("Show numeric summary table"):
        raw = {
            "Metric": [
                "PV1 kWp",
                "PV2 kWp",
                "PV total kWp",
                "PV1 gen (auto)",
                "PV2 gen",
                "PV total gen",
                "Total load",
                "Self-consumption",
                "Grid feed-in P2",
                "Grid import",
                "Cap P3",
                "Eligible Pauschal kWh",
                "Net Pauschal €",
                "Net EEG €",
                "Diff EEG - Pauschal €",
            ],
            "Value": [
                params["pv1_kwp"],
                params["pv2_kwp"],
                params["pv1_kwp"] + params["pv2_kwp"],
                params["pv1_gen"],
                params["pv2_gen"],
                flows["generation_total"],
                flows["total_load"],
                flows["self_consumption"],
                flows["grid_feed_in"],
                flows["grid_import"],
                pauschal["P3"],
                pauschal["eligible"],
                pauschal["net_pauschal"],
                pauschal["net_eeg"],
                diff,
            ],
        }
        st.table(pd.DataFrame(raw))

# -------- Tab 2: Annex 1 Storage --------
with tab2:
    st.subheader("Annex 1-style Storage + EV Separation (A3-like)")

    st.markdown(
        """
We approximate Annex-1 case A3 by treating **storage + EV** as one combined unit
with:

- Annual charge/discharge (kWh)  
- Share of charge from PV vs grid  
- Share of discharge that goes to the grid vs local load  
- Optional *foreign charge* (e.g. EV charged elsewhere)

From that, we compute the standard Annex-1 variables (3,4,5,6,7,8,9,10,11,12,13,16).
"""
    )

    col_an1, col_an2, col_an3 = st.columns(3)
    col_an1.metric(
        "(4) Consumption storage+EV (kWh)",
        f"{annex1['(4)_cons_storage_ev']:.0f}",
    )
    col_an2.metric(
        "(5) Output storage+EV (kWh)",
        f"{annex1['(5)_output_storage_ev']:.0f}",
    )
    col_an3.metric(
        "(6) Grid → storage+EV (kWh)",
        f"{annex1['(6)_simul_grid_to_storage_ev']:.0f}",
    )

    col_an4, col_an5, col_an6 = st.columns(3)
    col_an4.metric(
        "(7) storage+EV → grid (kWh)",
        f"{annex1['(7)_simul_storage_ev_to_grid']:.0f}",
    )
    col_an5.metric(
        "(9) Total cons. (4+8) (kWh)",
        f"{annex1['(9)_total_consumption']:.0f}",
    )
    col_an6.metric(
        "(10) Grid share", f"{annex1['(10)_grid_share']:.3f}"
    )

    col_an7, col_an8, col_an9 = st.columns(3)
    col_an7.metric(
        "(11) Balancing-eligible feed-in (kWh)",
        f"{annex1['(11)_balancing_eligible_feed_in']:.1f}",
    )
    col_an8.metric(
        "(12) Losses (kWh)", f"{annex1['(12)_losses']:.1f}"
    )
    col_an9.metric(
        "(13) Privileged losses (kWh)",
        f"{annex1['(13)_privileged_losses']:.1f}",
    )

    st.markdown("---")
    st.metric(
        "(16) Chargeable grid import (Annex 1)",
        f"{annex1['(16)_chargeable_grid_import']:.1f} kWh",
    )
    st.write(
        f"Baseline grid import in this calc was (3) = {annex1['(3)_grid_import']:.1f} kWh."
    )

    with st.expander("Show Annex-1 raw values as table"):
        st.table(
            pd.DataFrame.from_dict(
                annex1, orient="index", columns=["Value"]
            )
        )

# -------- Tab 3: Sensitivity --------
with tab3:
    st.subheader("Sensitivity: PV1 Generation vs AW_share")

    st.markdown(
        """
Heatmap shows **EEG − Pauschal** net difference (€) for different combinations of:

- Annual PV1 generation (x-axis = rows)  
- AW_share (y-axis = columns)

> Green → EEG better  
> Red → Pauschal better  
> Near zero → almost equal
"""
    )

    st.write(
        f"Base PV1 generation: **{params['pv1_gen']:.0f} kWh/yr**, base AW_share: **{params['aw_share']:.2f}**"
    )

    sens_pivot = sens_df
    try:
        st.dataframe(
            sens_pivot.style.background_gradient(
                cmap="RdYlGn", axis=None
            )
        )
    except Exception:
        st.dataframe(sens_pivot)

    st.caption(
        "Tip: change the sliders in the sidebar (AW, EEG, PV1 size, etc.) "
        "and watch how the advantage region shifts."
    )
