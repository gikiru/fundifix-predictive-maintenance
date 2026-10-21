"""
FundiFix water point risk dashboard (Capstone Module 5).

Run locally from the repo root:   streamlit run app/dashboard.py
Deployed on Streamlit Community Cloud with app/requirements.txt.

Data: the 1,955 test-set water points from Module 4 (never used in training).
No GPS points, WPdx IDs or water point names are included (Module 3 privacy rules).
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from fairlearn.metrics import (MetricFrame, count, demographic_parity_ratio, equalized_odds_difference,
                               false_positive_rate, selection_rate, true_positive_rate)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "app" / "data"
MODEL = ROOT / "models" / "fundifix_model.joblib"

NAVY, TEAL, TERRA, GREY, INK = "#1F3864", "#0E7C86", "#C4622D", "#8A96A8", "#1E1E1E"
STATUS_COLORS = {"Functional": NAVY, "Needs Repair": TEAL, "Non-Functional": TERRA}
MIN_GROUP = 50          # groups with fewer test records are shown but marked as too small
MIN_BROKEN = 10         # recall based on fewer broken points is marked as unreliable

st.set_page_config(page_title="FundiFix water point risk", page_icon="💧", layout="wide")


# ---------- data ----------
@st.cache_resource
def load_model():
    return joblib.load(MODEL)


@st.cache_data
def load_data():
    df = pd.read_csv(DATA / "water_points.csv")
    summary = json.loads((DATA / "summary.json").read_text())
    mitigation = pd.read_csv(DATA / "fairness_mitigation_by_county.csv", header=[0, 1], index_col=0)
    return df, summary, mitigation


model = load_model()
df, S, mitigation = load_data()
LABELS = S["labels"]
FEATURES = S["features"]
CLASSES = S["classes"]
DEFAULT_T = float(S["nf_threshold"])
BASE_RATE = (df["actual_status"] == "Non-Functional").mean()


def pct(x, d=0):
    return f"{x * 100:.{d}f}%"


def flag_stats(data, t):
    broken = data["actual_status"] == "Non-Functional"
    flag = data["p_non_functional"] >= t
    return {"visits": int(flag.sum()), "share_visited": flag.mean(),
            "broken": int(broken.sum()), "found": int((flag & broken).sum()),
            "recall": (flag & broken).sum() / max(broken.sum(), 1),
            "hit_rate": (flag & broken).sum() / max(flag.sum(), 1)}


def bar(labels, values, color=TEAL, text=None, xtitle="", height=320, xrange=None):
    fig = go.Figure(go.Bar(x=values, y=labels, orientation="h", marker_color=color,
                           text=text, textposition="outside", cliponaxis=False,
                           hovertemplate="%{y}: %{x}<extra></extra>"))
    fig.update_layout(height=height, margin=dict(l=10, r=40, t=10, b=40), xaxis_title=xtitle,
                      yaxis=dict(autorange="reversed"), plot_bgcolor="white",
                      xaxis=dict(gridcolor="#E5E9F0", range=xrange), font=dict(size=13, color=INK))
    return fig


# ---------- sidebar ----------
with st.sidebar:
    st.markdown("### FundiFix water point risk")
    st.caption("MSc Data Analytics capstone, Nexford University. Final Project.")
    st.markdown(
        f"**Model:** random forest, version {S['model_version']}  \n"
        f"**Water points shown:** {S['test_rows']:,} test-set points the model never trained on  \n"
        f"**Default visit threshold:** {DEFAULT_T:.2f}")
    st.info("Decision support only. A FundiFix supervisor reviews every visit list before any dispatch.")
    st.caption("Data: Water Point Data Exchange (2024), WPdx-Plus Kenya. GPS points, WPdx IDs and "
               "water point names are removed. Point IDs (WP-0001 ...) are for this dashboard only.")

st.title("Which water points are most likely broken right now?")
tabs = st.tabs(["Overview", "Visit list", "What drives risk", "Explain a water point",
                "What-if", "Fairness check", "Monitoring", "Limits and transparency"])

# ---------- 1. Overview ----------
with tabs[0]:
    o = flag_stats(df, DEFAULT_T)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Water points in test set", f"{len(df):,}")
    c2.metric("Actually broken (Non-Functional)", f"{o['broken']} ({pct(BASE_RATE, 1)})")
    c3.metric("Flagged for a visit", f"{o['visits']} ({pct(o['share_visited'], 1)})")
    c4.metric("Broken points found", f"{o['found']} of {o['broken']} ({pct(o['recall'])})")

    left, right = st.columns([1, 1])
    with left:
        st.subheader("Broken points found per 100 visits")
        fig = bar(["Visits chosen by the model", "Visits chosen at random"],
                  [o["hit_rate"] * 100, BASE_RATE * 100], color=[TEAL, GREY],
                  text=[f"{o['hit_rate'] * 100:.0f}", f"{BASE_RATE * 100:.0f}"],
                  xtitle="Broken water points found per 100 visits", height=230, xrange=[0, 40])
        st.plotly_chart(fig, width="stretch")
        st.caption(f"Test set, threshold {DEFAULT_T:.2f}. Random visits find broken points at the "
                   f"average rate ({pct(BASE_RATE, 1)}).")
    with right:
        st.subheader("How the model works, in plain language")
        st.markdown(
            "- It looked at **9,773 past survey records** of rural water points in Kenya, each marked "
            "Functional, Needs Repair or Non-Functional.\n"
            "- It learned which **combinations of features** (water source, pump type, distance to roads "
            "and towns, age, population served, management) went with each status.\n"
            "- For a water point, it gives a **chance of being broken right now**. Points at or above the "
            "threshold go on the visit list.\n"
            "- It **does not forecast** when a working pump will fail. The data records one status per "
            "point, with no repair history.")
    st.subheader("How often the model's main guess is right, by actual status")
    cm = pd.crosstab(df["actual_status"], df["predicted_status"], normalize="index").reindex(
        index=CLASSES, columns=CLASSES).fillna(0)
    rows = [f"Actually {c}" for c in CLASSES]
    fig = go.Figure(go.Bar(x=[cm.loc[c, c] * 100 for c in CLASSES], y=rows, orientation="h",
                           marker_color=[STATUS_COLORS[c] for c in CLASSES],
                           text=[f"{cm.loc[c, c] * 100:.0f}% correctly labelled" for c in CLASSES],
                           textposition="outside", cliponaxis=False))
    fig.update_layout(height=220, margin=dict(l=10, r=40, t=10, b=40), plot_bgcolor="white",
                      xaxis=dict(range=[0, 100], title="Share labelled correctly (%)", gridcolor="#E5E9F0"),
                      yaxis=dict(autorange="reversed"), font=dict(size=13, color=INK))
    st.plotly_chart(fig, width="stretch")
    st.caption("The main guess (highest chance) is weakest for broken points. That is why the visit list "
               "uses a lower threshold on the Non-Functional chance instead of the main guess.")

# ---------- 2. Visit list ----------
with tabs[1]:
    st.subheader("Visit list, ranked by chance of being broken")
    a, b = st.columns([2, 1])
    counties = sorted(df["county"].unique())
    pick = a.multiselect("Counties (leave empty for all)", counties, default=[]) or counties
    t = b.slider("Visit threshold (chance of being broken)", 0.05, 0.80, DEFAULT_T, 0.01,
                 help="Lower = more visits and more broken points found. Higher = fewer, more targeted visits.")
    sub = df[df["county"].isin(pick)]
    s = flag_stats(sub, t)
    m1, m2, m3 = st.columns(3)
    m1.metric("Points to visit", f"{s['visits']} of {len(sub)}")
    m2.metric("Broken points this list would find", f"{s['found']} of {s['broken']} ({pct(s['recall'])})")
    m3.metric("Broken points per 100 visits", f"{s['hit_rate'] * 100:.0f} (random: {BASE_RATE * 100:.0f})")
    table = (sub[sub["p_non_functional"] >= t]
             .sort_values("p_non_functional", ascending=False)
             [["point_id", "county", "sub_county", "ward", "p_non_functional", "predicted_status", "top_reasons"]]
             .rename(columns={"point_id": "Point", "county": "County", "sub_county": "Sub-county", "ward": "Ward",
                              "p_non_functional": "Chance broken", "predicted_status": "Main guess",
                              "top_reasons": "Top 3 reasons"}))
    st.dataframe(table, width="stretch", hide_index=True, height=420,
                 column_config={"Chance broken": st.column_config.ProgressColumn(format="%.2f", min_value=0, max_value=1)})
    st.download_button("Download this visit list (CSV)", table.to_csv(index=False).encode("utf-8"),
                       file_name=f"fundifix_visit_list_t{t:.2f}.csv", mime="text/csv",
                       help="For field teams with poor connectivity. Supervisor sign-off is still required.")
    st.caption("This list is for planning only. Actual status is hidden here, as it would be in real use.")

# ---------- 3. What drives risk ----------
with tabs[2]:
    st.subheader("What the model pays most attention to")
    g = pd.Series(S["global_drivers"]).sort_values(ascending=False).head(10)
    st.plotly_chart(bar(list(g.index), list(g.values * 100), color=TEAL,
                        text=[f"{v * 100:.1f}" for v in g.values],
                        xtitle="Average effect on the predicted chances (percentage points)", height=380),
                    width="stretch")
    st.caption("Average size of each input's effect across 600 test water points (SHAP values, Module 4). "
               "Bigger bar = the input moves predictions more. It does not show direction.")
    st.markdown(
        "- **Water source and pump type matter most.** Sand and sub-surface dams are broken or in need of repair "
        "more often than protected wells. Part of this reflects who did the survey: almost all public tapstand "
        "records come from one survey (USAID KIWASH).\n"
        "- **Remoteness comes next.** Points far from towns, cities and main roads are more often broken.\n"
        "- **Age matters less than expected,** partly because many records have no installation year.")

# ---------- 4. Explain a water point ----------
with tabs[3]:
    st.subheader("Why did the model rate this water point the way it did?")
    ranked = df.sort_values("p_non_functional", ascending=False)["point_id"].tolist()
    pid = st.selectbox("Choose a water point (highest risk first)", ranked, key="explain_pick")
    r = df.set_index("point_id").loc[pid]
    a, b = st.columns([1, 1])
    with a:
        st.markdown(f"**{pid}**, {r['ward']} ward, {r['sub_county']}, {r['county']}")
        st.markdown(f"{r['water_source_clean']} · {r['water_tech_clean']} · {r['management_clean']} · {r['is_urban']}")
        probs = [r["p_functional"], r["p_needs_repair"], r["p_non_functional"]]
        fig = bar(CLASSES, [p * 100 for p in probs], color=[STATUS_COLORS[c] for c in CLASSES],
                  text=[pct(p) for p in probs], xtitle="Chance (%)", height=200, xrange=[0, 100])
        st.plotly_chart(fig, width="stretch")
        flagged = r["p_non_functional"] >= DEFAULT_T
        st.markdown(f"**On the visit list at threshold {DEFAULT_T:.2f}:** {'Yes' if flagged else 'No'}  \n"
                    f"**Actual status in the survey:** {r['actual_status']}")
    with b:
        sh = pd.Series({LABELS[f]: r[f"shap_{f}"] for f in FEATURES}).sort_values(key=np.abs, ascending=False).head(6)
        sh = sh.iloc[::-1]
        fig = go.Figure(go.Bar(x=sh.values * 100, y=sh.index, orientation="h",
                               marker_color=[TERRA if v > 0 else NAVY for v in sh.values],
                               text=[f"{v * 100:+.1f}" for v in sh.values], textposition="outside", cliponaxis=False))
        fig.update_layout(height=320, margin=dict(l=10, r=40, t=30, b=40), plot_bgcolor="white",
                          title=dict(text="Main reasons (effect on the chance of being broken)", font=dict(size=14)),
                          xaxis=dict(title="Percentage points (orange raises risk, navy lowers it)", gridcolor="#E5E9F0"),
                          font=dict(size=13, color=INK))
        st.plotly_chart(fig, width="stretch")
        st.caption("Each bar shows how much one feature pushed this water point's chance of being broken "
                   "up (orange) or down (navy), in percentage points. Reasons come from SHAP. "
                   "A second method (LIME) agrees on the first reason for about half of points and shares fewer than "
                   "half of the top three, so treat the first reason as the most reliable.")

# ---------- 5. What-if ----------
with tabs[4]:
    st.subheader("What-if: change a water point's features and see how the risk changes")
    st.caption("The model recalculates live. Changes show what the model has learned from past data, "
               "not what would happen in the field if the change were made.")
    pid2 = st.selectbox("Start from water point", ranked, key="whatif_pick")
    base = df.set_index("point_id").loc[pid2]
    opts = {c: sorted(df[c].dropna().astype(str).unique()) for c in
            ["water_source_clean", "water_tech_clean", "management_clean", "is_urban"]}
    c1, c2, c3 = st.columns(3)
    new = {f: base[f] for f in FEATURES}
    new["water_source_clean"] = c1.selectbox("Water source", opts["water_source_clean"],
                                             index=opts["water_source_clean"].index(str(base["water_source_clean"])))
    new["water_tech_clean"] = c2.selectbox("Pump / technology", opts["water_tech_clean"],
                                           index=opts["water_tech_clean"].index(str(base["water_tech_clean"])))
    new["management_clean"] = c3.selectbox("Management", opts["management_clean"],
                                           index=opts["management_clean"].index(str(base["management_clean"])))
    age0 = 0 if pd.isna(base["age_at_report"]) else int(base["age_at_report"])
    age = c1.slider("Age at report (years)", 0, 60, min(age0, 60))
    new["age_at_report"] = np.nan if pd.isna(base["age_at_report"]) and age == 0 else age
    new["install_year_missing"] = int(pd.isna(new["age_at_report"]))
    new["distance_to_town"] = c2.slider("Distance to nearest town (km)", 0.0, 60.0,
                                        float(min(base["distance_to_town"] / 1000, 60)), 0.5) * 1000
    new["local_population"] = c3.slider("Population within 1 km", 0, 5000,
                                        int(min(np.nan_to_num(base["local_population"]), 5000)), 50)
    before = model.predict_proba(pd.DataFrame([{f: base[f] for f in FEATURES}]))[0]
    after = model.predict_proba(pd.DataFrame([new]))[0]
    m1, m2, m3 = st.columns(3)
    m1.metric("Chance broken: before", pct(before[2]))
    change = (after[2] - before[2]) * 100
    m2.metric("Chance broken: after", pct(after[2]),
              f"{change:+.1f} points" if abs(change) >= 0.05 else None, delta_color="inverse")
    m3.metric("On visit list: before / after",
              f"{'Yes' if before[2] >= DEFAULT_T else 'No'} / {'Yes' if after[2] >= DEFAULT_T else 'No'}")
    fig = go.Figure()
    fig.add_bar(name="Before", x=CLASSES, y=before * 100, marker_color=GREY,
                text=[pct(p) for p in before], textposition="outside")
    fig.add_bar(name="After", x=CLASSES, y=after * 100, marker_color=TEAL,
                text=[pct(p) for p in after], textposition="outside")
    fig.update_layout(barmode="group", height=320, plot_bgcolor="white", yaxis=dict(title="Chance (%)", range=[0, 100],
                      gridcolor="#E5E9F0"), margin=dict(l=10, r=10, t=10, b=40), font=dict(size=13, color=INK),
                      legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig, width="stretch")

# ---------- 6. Fairness check ----------
with tabs[5]:
    st.subheader("Ethical compliance: does the visit list treat groups fairly?")
    st.markdown("The key question: **of the water points that are actually broken, does each group get the same "
                "share found?** A group with a lower share has broken points that the visit list misses more often.")
    a, b = st.columns([1, 1])
    group_name = a.selectbox("Compare groups by", ["County (FO1)", "Urban or rural (FO2)",
                                                   "Distance to main road (FO2)", "Survey organisation"])
    col = {"County (FO1)": "county", "Urban or rural (FO2)": "is_urban",
           "Distance to main road (FO2)": "road_access", "Survey organisation": "data_source"}[group_name]
    t2 = b.slider("Visit threshold", 0.05, 0.80, DEFAULT_T, 0.01, key="fair_t")
    y_true = (df["actual_status"] == "Non-Functional").astype(int)
    y_flag = (df["p_non_functional"] >= t2).astype(int)
    mf = MetricFrame(metrics={"records": count, "found": true_positive_rate, "flag_rate": selection_rate,
                              "false_alarm": false_positive_rate},
                     y_true=y_true, y_pred=y_flag, sensitive_features=df[col])
    tbl = mf.by_group.copy()
    tbl["broken"] = y_true.groupby(df[col]).sum()
    tbl["reliable"] = (tbl["records"] >= MIN_GROUP) & (tbl["broken"] >= MIN_BROKEN)
    no_broken = sorted(tbl.index[tbl["broken"] == 0].tolist())
    tbl = tbl[tbl["broken"] > 0].sort_values(["reliable", "found"], ascending=[False, False])
    ok = tbl[tbl["reliable"]]
    fig = go.Figure(go.Bar(
        x=tbl["found"] * 100, y=[f"{g} ({int(n)} broken)" for g, n in zip(tbl.index, tbl["broken"])],
        orientation="h", marker_color=[TEAL if rel else "#C9D3DF" for rel in tbl["reliable"]],
        text=[f"{v * 100:.0f}%" for v in tbl["found"]], textposition="outside", cliponaxis=False))
    overall = (y_flag & y_true).sum() / y_true.sum()
    fig.add_vline(x=overall * 100, line_dash="dash", line_color=TERRA,
                  annotation_text=f"overall {overall * 100:.0f}%", annotation_position="top")
    fig.update_layout(height=max(260, 34 * len(tbl) + 80), plot_bgcolor="white", margin=dict(l=10, r=50, t=30, b=40),
                      yaxis=dict(autorange="reversed"),
                      xaxis=dict(range=[0, 105], title="Share of broken points that the visit list finds (%)",
                                 gridcolor="#E5E9F0"), font=dict(size=13, color=INK))
    st.plotly_chart(fig, width="stretch")
    st.caption(f"Grey bars: fewer than {MIN_GROUP} test records or fewer than {MIN_BROKEN} broken points. "
               "These rates rest on very few cases and can swing widely."
               + (f" Not shown (no broken points in the test set): {', '.join(no_broken)}." if no_broken else ""))
    if len(ok) >= 2:
        gap = ok["found"].max() - ok["found"].min()
        di = demographic_parity_ratio(y_true[df[col].isin(ok.index)], y_flag[df[col].isin(ok.index)],
                                      sensitive_features=df.loc[df[col].isin(ok.index), col])
        eo = equalized_odds_difference(y_true[df[col].isin(ok.index)], y_flag[df[col].isin(ok.index)],
                                       sensitive_features=df.loc[df[col].isin(ok.index), col])
        k1, k2, k3 = st.columns(3)
        k1.metric("Gap in share found (teal groups)", f"{gap * 100:.0f} points")
        k2.metric("Disparate impact ratio", f"{di:.2f}")
        k3.metric("Equalized odds difference", f"{eo:.2f}")
        flags = []
        if di < 0.80:
            flags.append("disparate impact below 0.80")
        if eo > 0.10:
            flags.append("equalized odds difference above 0.10")
        if flags:
            st.warning(f"Flagged: {' and '.join(flags)}. Review before using this visit list for {group_name.split(' (')[0].lower()} planning.")
        else:
            st.success("No flag at this threshold for the teal groups.")
    else:
        st.warning("Fewer than two groups have enough data for a reliable comparison.")
    with st.expander("What these measures mean"):
        st.markdown(
            "- **Share found (recall):** of the broken points in a group, the share the visit list includes.\n"
            "- **Disparate impact ratio:** lowest group's visit rate divided by the highest. Below 0.80 is a warning "
            "sign. Groups with more broken points should be visited more, so this measure alone can mislead.\n"
            "- **Equalized odds difference:** the largest gap between groups in share found or false-alarm rate. "
            "This is the main measure, because it compares groups on points that are actually broken.")
    with st.expander("What was tried to reduce the county gap (Module 4)"):
        st.markdown("Separate thresholds for each county were tested to even out the share of broken points found "
                    "(Fairlearn ThresholdOptimizer). The gap fell from 0.86 to 0.63, but the overall share of broken "
                    "points found fell from 69% to 63%. The single threshold stays the default.")
        mit = pd.DataFrame({"Found before": mitigation[("before", "recall_nf")],
                            "Found after": mitigation[("after", "recall_nf")],
                            "Broken points": mitigation[("before", "broken_points")].astype(int)}).round(2)
        st.dataframe(mit, width="stretch")

# ---------- 7. Monitoring (Final Project) ----------
SEV = {"green": ("OK", "#2E7D32"), "amber": ("Review", "#B26A00"), "red": ("Act", "#B3261E")}


@st.cache_data
def load_monitoring():
    mdir = DATA / "monitoring"
    if not (mdir / "summary.json").exists():
        return None, None
    m = json.loads((mdir / "summary.json").read_text())
    hist = pd.read_json(mdir / "history.jsonl", lines=True) if (mdir / "history.jsonl").exists() else None
    return m, hist


with tabs[6]:
    M, H = load_monitoring()
    st.subheader("Is the model still working, and still fair?")
    if M is None:
        st.info("No monitoring run found. Run python -m src.monitoring.monitor")
    else:
        th = M["thresholds"]
        st.caption(f"Last monitoring run: {M['run_at'][:16].replace('T', ' ')} UTC. Model version {M['model_version']}, "
                   f"visit threshold {M['nf_threshold']:.2f}. Reference: {M['reference_rows']:,} training records. "
                   "Weekly job: src/monitoring/monitor.py (Evidently for drift, Fairlearn for fairness).")
        cols = st.columns(len(M["batches"]))
        for c, b in zip(cols, M["batches"]):
            word, colour = SEV[b["status"]]
            c.markdown(f"<div style='border:1px solid #D5DBE3;border-radius:6px;padding:10px'>"
                       f"<div style='font-size:13px;color:#5A6472'>{b['label']}</div>"
                       f"<div style='font-size:22px;font-weight:700;color:{colour}'>{word} ({b['status']})</div>"
                       f"<div style='font-size:13px'>{b['rows']:,} water points</div></div>", unsafe_allow_html=True)

        alerts = [{"Batch": b["label"], "Severity": a["severity"], "Area": a["area"], "Message": a["message"]}
                  for b in M["batches"] for a in b["alerts"]]
        st.markdown("**Alerts**")
        if alerts:
            st.dataframe(pd.DataFrame(alerts), width="stretch", hide_index=True)
        else:
            st.success("No alerts.")

        left, right = st.columns(2)
        with left:
            st.markdown("**Data drift: share of inputs that differ from the training data**")
            names = [b["label"].split(" (")[0].split(":")[0] for b in M["batches"]]
            vals = [b["drift"]["drifted_feature_share"] * 100 for b in M["batches"]]
            fig = bar(names, vals, color=[SEV[b["status"]][1] if b["drift"]["drifted_feature_share"] >= th["drifted_feature_share"]["amber"]
                                          else TEAL for b in M["batches"]],
                      text=[f"{v:.0f}%" for v in vals], xtitle="Inputs drifted (%)", height=260, xrange=[0, 105])
            fig.add_vline(x=th["drifted_feature_share"]["amber"] * 100, line_dash="dot", line_color=GREY,
                          annotation_text="review", annotation_position="top")
            fig.add_vline(x=th["drifted_feature_share"]["red"] * 100, line_dash="dot", line_color=TERRA,
                          annotation_text="act", annotation_position="top")
            fig.update_layout(margin=dict(t=40))
            st.plotly_chart(fig, width="stretch")
        with right:
            st.markdown("**Performance on labelled batches: share of broken points found**")
            lab = [b for b in M["batches"] if b.get("performance")]
            base = M["baseline"]["performance"]["nf_recall"] * 100
            fig = bar(["At go-live"] + [b["label"].split(" (")[0].split(":")[0] for b in lab],
                      [base] + [b["performance"]["nf_recall"] * 100 for b in lab],
                      color=[GREY] + [TEAL] * len(lab),
                      text=[f"{base:.0f}%"] + [f"{b['performance']['nf_recall'] * 100:.0f}%" for b in lab],
                      xtitle="Broken points found (%)", height=260, xrange=[0, 105])
            fig.add_vline(x=th["nf_recall"]["red"] * 100, line_dash="dot", line_color=TERRA,
                          annotation_text="act below", annotation_position="top")
            fig.update_layout(margin=dict(t=40))
            st.plotly_chart(fig, width="stretch")

        st.markdown("**Bias drift: gap between groups in share of broken points found**")
        rows = []
        for b in lab:
            for grp, f in b["fairness"].items():
                g0 = M["baseline"]["fairness"].get(grp, {})
                rows.append({"Batch": b["label"].split(" (")[0].split(":")[0], "Grouping": grp.replace("_", " "),
                             "Gap at go-live": f"{g0.get('recall_gap', float('nan')) * 100:.0f} pts",
                             "Gap now": f"{f['recall_gap'] * 100:.0f} pts",
                             "Change": f"{(f['recall_gap'] - g0.get('recall_gap', 0)) * 100:+.0f} pts",
                             "Equalized odds diff.": round(f["equalized_odds_difference"], 2),
                             "Disparate impact": round(f["disparate_impact_ratio"], 2)})
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        st.caption(f"Review when a gap rises by {th['bias_drift']['amber'] * 100:.0f} points, act at "
                   f"{th['bias_drift']['red'] * 100:.0f}. Only groups with at least 50 records and 10 broken points are compared. "
                   "A red alert is also raised if remote points (over 20 km from a main road) become less likely to be found "
                   "than nearby points.")

        pick = st.selectbox("Inspect drift by input", [b["label"] for b in M["batches"]])
        b = next(x for x in M["batches"] if x["label"] == pick)
        feat = pd.DataFrame([{"Input": LABELS.get(k, k), "Test used": v["method"], "Score": v["score"],
                              "Limit": v["threshold"], "Drifted": "yes" if v["drifted"] else "no"}
                             for k, v in b["drift"]["features"].items()]).sort_values("Drifted", ascending=False)
        st.dataframe(feat, width="stretch", hide_index=True)
        if st.toggle(f"Load the full Evidently report for this batch (about 4.5 MB)", value=False):
            html = DATA / "monitoring" / b["drift"]["html"]
            if html.exists():
                st.iframe(html, height=900)

        if H is not None and len(H) > 1:
            st.markdown("**History**")
            st.dataframe(H, width="stretch", hide_index=True)

        st.markdown("**What the results mean now**\n"
                    "- **Hold-out test set:** green. The monitoring checks agree with the results reported at go-live.\n"
                    "- **Points from other survey sources:** most inputs differ from the training data. "
                    "The model would be working outside what it learned if applied to these points. "
                    "This supports validating on FundiFix's own records before use in Kitui or Kwale.\n"
                    "- **Stress test:** an older, less documented fleet lowers balanced accuracy and widens the "
                    "road-access gap. The alerts fire as designed.")
        es = M.get("explanation_store", {})
        st.caption(f"Explanation store at last run: {es.get('rows', 0)} logged API predictions. "
                   "Every /predict call stores inputs, outputs and top reasons for audit (GET /explanations/{id}). "
                   "Response to alerts: docs/monitoring_plan.md.")

# ---------- 8. Limits and transparency ----------
with tabs[7]:
    st.subheader("What this model can and cannot do")
    a, b = st.columns(2)
    with a:
        st.markdown("**It can**\n"
                    "- Rank water points by their chance of being broken now, from features in the survey data.\n"
                    f"- Find about {pct(S['operating_point']['test_nf_recall'])} of broken points while visiting about "
                    f"{pct(S['operating_point']['test_share_flagged'])} of all points (test set).\n"
                    "- Show the main reasons behind each rating.")
    with b:
        st.markdown("**It cannot**\n"
                    "- Predict when a working water point will fail. The data has no repair history.\n"
                    "- Speak for Kitui or Kwale yet. Training data holds one Kitui record and no Kwale records.\n"
                    f"- Reach the Module 1 target of 80% balanced accuracy. It reached "
                    f"{pct(S['test_metrics']['random_forest']['balanced_accuracy'])}.\n"
                    "- Replace a field visit or a supervisor's judgement.")
    st.markdown("**Known limitations**\n"
                "1. Labels depend partly on which organisation did the survey. 12 of 16 survey sources marked every "
                "point broken and were left out of training.\n"
                "2. Training records date from 2011 to 2021, so conditions may have changed.\n"
                "3. Fairness results for small counties and urban points rest on very few broken points.")
    st.markdown("**Transparency statement.** This dashboard shows estimates from a model trained on public survey "
                "data. It supports, and does not replace, FundiFix staff decisions. Scores must not be used to "
                "reduce or withdraw service from any community. GPS points, WPdx IDs and water point names are not "
                "shown. Code and documentation: github.com/gikiru/fundifix-predictive-maintenance.")
