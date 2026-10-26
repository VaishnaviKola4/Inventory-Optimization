from __future__ import annotations
from pathlib import Path
import json
import zipfile
from io import BytesIO

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from inventory_intelligence.demo_service import build_demo_result, load_demo_tables, DATA_DIR, MODEL_PATH, METRICS_PATH
from inventory_intelligence.demo_model import train_and_save
from inventory_intelligence.reporting import build_pdf_report
from inventory_intelligence.service import generate_recommendations
from inventory_intelligence.data import DataValidationError

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="Stockwise | Inventory Intelligence", page_icon="📦", layout="wide", initial_sidebar_state="expanded")

# ---- Visual identity ---------------------------------------------------------
NAVY = "#172033"
INK = "#202B3C"
SLATE = "#667085"
MUTED = "#98A2B3"
BORDER = "#E6EAF0"
CANVAS = "#F5F7FB"
TEAL = "#0F8B8D"
ORANGE = "#F59E0B"
RED = "#D9485F"
GREEN = "#16866D"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
:root {{ --ink:{INK}; --slate:{SLATE}; --border:{BORDER}; }}
html, body, [class*="css"] {{ font-family: 'DM Sans', sans-serif; }}
.stApp {{ background: {CANVAS}; color: {INK}; }}
[data-testid="stHeader"] {{ background: rgba(245,247,251,.92); }}
.block-container {{ padding-top: 1.35rem; padding-bottom: 3.5rem; max-width: 1600px; }}
[data-testid="stSidebar"] {{ background: {NAVY}; border-right: 1px solid #263147; }}
[data-testid="stSidebar"] * {{ color: #F8FAFC; }}
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {{ color: #CFD7E5 !important; font-size: .82rem; font-weight: 600; }}
[data-testid="stSidebar"] .stCaption {{ color: #9FAEC4 !important; }}
[data-testid="stSidebar"] hr {{ border-color: #364158; }}
[data-testid="stSidebar"] div[data-baseweb="select"] > div {{ background: #202B40; border-color: #3A4962; }}
[data-testid="stSidebar"] input {{ background: #202B40 !important; }}
[data-testid="stMetric"] {{ background: white; padding: 18px 18px 14px; border: 1px solid {BORDER}; border-radius: 16px; box-shadow: 0 3px 12px rgba(24,39,75,.035); min-height: 122px; }}
[data-testid="stMetricLabel"] p {{ color: {SLATE}; font-size: .82rem; font-weight: 600; }}
[data-testid="stMetricValue"] {{ color: {INK}; font-family: 'Manrope', sans-serif; font-weight: 800; font-size: 1.7rem; }}
[data-testid="stMetricDelta"] {{ font-size: .76rem; }}
div.stButton > button {{ border-radius: 10px; border: 1px solid #D8DEE8; font-weight: 700; padding: .55rem .85rem; }}
div.stButton > button[kind="primary"] {{ background: {TEAL}; border-color: {TEAL}; color: white; }}
div.stButton > button[kind="primary"]:hover {{ background: #08787A; border-color: #08787A; color:white; }}
.stDownloadButton > button {{ border-radius: 10px; font-weight: 600; }}
[data-testid="stTabs"] button {{ font-weight: 700; }}
[data-testid="stDataFrame"] {{ border: 1px solid {BORDER}; border-radius: 12px; overflow: hidden; }}
.hero {{ background: linear-gradient(112deg, #141D2F 0%, #1D3044 58%, #14696C 100%); color: white; border-radius: 22px; padding: 28px 32px; margin-bottom: 20px; box-shadow: 0 14px 36px rgba(23,32,51,.14); position: relative; overflow: hidden; }}
.hero:after {{ content:""; position:absolute; right:-55px; top:-130px; width:360px; height:360px; border:1px solid rgba(255,255,255,.12); border-radius:50%; box-shadow: 0 0 0 38px rgba(255,255,255,.025),0 0 0 76px rgba(255,255,255,.025); }}
.hero-eyebrow {{ text-transform:uppercase; letter-spacing:.16em; color:#A7E4DE; font-size:.72rem; font-weight:800; margin-bottom:9px; }}
.hero h1 {{ font-family:'Manrope',sans-serif; font-size:2.05rem; line-height:1.12; margin:0; color:white; font-weight:800; }}
.hero p {{ margin:.7rem 0 0 0; color:#D5E1EE; max-width:760px; font-size:.99rem; line-height:1.55; }}
.hero-bottom {{ display:flex; gap:8px; flex-wrap:wrap; margin-top:18px; }}
.hero-chip {{ display:inline-flex; align-items:center; border:1px solid rgba(255,255,255,.19); color:#EAF4F6; background:rgba(255,255,255,.08); padding:5px 10px; border-radius:99px; font-size:.74rem; font-weight:600; }}
.section-head {{ display:flex; align-items:baseline; justify-content:space-between; margin:22px 0 10px; }}
.section-head h3 {{ font-family:'Manrope',sans-serif; font-size:1.14rem; margin:0; color:{INK}; font-weight:800; }}
.section-head span {{ color:{SLATE}; font-size:.79rem; }}
.panel {{ background:white; border:1px solid {BORDER}; border-radius:16px; padding:18px 18px 12px; }}
.panel-title {{ font-family:'Manrope',sans-serif; color:{INK}; font-weight:800; font-size:.98rem; margin-bottom:2px; }}
.panel-subtitle {{ color:{SLATE}; font-size:.79rem; margin-bottom:12px; }}
.pill {{ display:inline-block; padding:4px 9px; border-radius:99px; font-size:.72rem; font-weight:800; letter-spacing:.01em; }}
.pill-green {{ background:#E5F6F1; color:#116B59; }} .pill-amber {{ background:#FFF3D6; color:#895C00; }} .pill-red {{ background:#FCE8EC; color:#AB263B; }} .pill-blue {{ background:#E7EFFD; color:#305DA8; }}
.small-note {{ font-size:.77rem; color:{SLATE}; line-height:1.45; }}
div[data-testid="stExpander"] {{ background:white; border:1px solid {BORDER}; border-radius:12px; }}
footer {{ visibility:hidden; }}
@media (max-width: 900px) {{ .hero {{ padding:22px; }} .hero h1 {{ font-size:1.55rem; }} }}
</style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def cached_tables():
    t = load_demo_tables()
    # Cache only UI-ready dataframes.
    return {k: v for k, v in t.items()}


@st.cache_data(show_spinner=False, ttl=900)
def cached_result(horizon: int, service_level: float):
    return build_demo_result(horizon=horizon, service_level=service_level)


def make_dataset_zip() -> bytes:
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for name in ["sample_sales.csv", "sample_inventory.csv", "sample_products.csv", "sample_suppliers.csv"]:
            zf.write(DATA_DIR / name, arcname=name)
        zf.write(ROOT / "outputs" / "model_metrics.json", arcname="model_metrics.json")
    return buffer.getvalue()


def section(title: str, subtitle: str = ""):
    st.markdown(f'<div class="section-head"><h3>{title}</h3><span>{subtitle}</span></div>', unsafe_allow_html=True)


def action_pill(action: str) -> str:
    style = {"Reorder": "pill-red", "Promote": "pill-amber", "Hold": "pill-green"}.get(action, "pill-blue")
    return f'<span class="pill {style}">{action.upper()}</span>'


def load_upload(upload):
    suffix = Path(upload.name).suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(upload)
    return pd.read_csv(upload)


def show_hero(result: dict, data_mode: str):
    if data_mode == "Synthetic demo":
        chip_text = f"{result.get('sales_rows', 10000):,} synthetic sales records"
        end = result.get("source_data_end_date", "latest date")
        chips = f'<span class="hero-chip">● Pre-trained ML model</span><span class="hero-chip">{chip_text}</span><span class="hero-chip">History through {end}</span>'
        desc = "Forecast demand, protect service levels, and spot excess stock before it ties up cash. Explore a fully prepared retail scenario powered by a model trained on 10,000 synthetic sales observations."
    else:
        chips = '<span class="hero-chip">● Custom data mode</span><span class="hero-chip">CSV / Excel inputs</span><span class="hero-chip">Human-approved decisions</span>'
        desc = "Turn your retail sales history and stock snapshot into transparent inventory actions. Upload data, inspect the evidence, and export recommendations for review."
    st.markdown(f"""
    <div class="hero">
      <div class="hero-eyebrow">Retail operations · Decision intelligence</div>
      <h1>Inventory decisions, <span style="color:#86DFD4">made smarter.</span></h1>
      <p>{desc}</p>
      <div class="hero-bottom">{chips}</div>
    </div>
    """, unsafe_allow_html=True)


def create_forecast_figure(tables: dict, result: dict, pid: str):
    sales = tables["sales"].copy()
    sales["date"] = pd.to_datetime(sales["date"])
    hist = sales[sales.product_id.astype(str) == str(pid)].sort_values("date").tail(90)
    forecast = result.get("forecast_series", {}).get(str(pid), {})
    fig = go.Figure()
    if not hist.empty:
        fig.add_trace(go.Scatter(
            x=hist.date, y=hist.units_sold, mode="lines", name="Observed demand",
            line=dict(color=TEAL, width=2.3), fill="tozeroy", fillcolor="rgba(15,139,141,0.08)"
        ))
    if forecast:
        fig.add_trace(go.Scatter(
            x=pd.to_datetime(forecast.get("dates", [])), y=forecast.get("daily_forecast", []),
            mode="lines+markers", name="ML forecast", line=dict(color=ORANGE, width=2.8, dash="dash"),
            marker=dict(size=5),
        ))
    fig.update_layout(
        template="plotly_white", height=365, margin=dict(l=8, r=8, t=18, b=8),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        xaxis_title=None, yaxis_title="Units sold", hovermode="x unified",
        font=dict(family="DM Sans", color=INK), paper_bgcolor="white", plot_bgcolor="white",
    )
    fig.update_xaxes(showgrid=False, linecolor=BORDER)
    fig.update_yaxes(gridcolor="#EDF0F5", zeroline=False)
    return fig


def render_model_metrics(metrics: dict):
    if not metrics:
        st.info("Model evaluation metrics are not available. Run `python train_demo_model.py` to train the model and write the evaluation report.")
        return
    section("Model evaluation", "28-day rolling-origin holdout · synthetic data")
    a, b, c, d = st.columns(4)
    a.metric("Holdout MAE", f"{metrics.get('model_mae_units_per_day', 0):.2f}", help="Mean absolute error in units per SKU-day.")
    b.metric("Holdout WAPE", f"{metrics.get('model_wape_percent', 0):.1f}%", help="Weighted absolute percentage error on the synthetic holdout.")
    c.metric("Seasonal baseline MAE", f"{metrics.get('seasonal_naive_mae_units_per_day', 0):.2f}", help="Baseline repeats demand from seven days earlier.")
    diff = metrics.get("seasonal_naive_mae_units_per_day", 0) - metrics.get("model_mae_units_per_day", 0)
    d.metric("MAE improvement vs baseline", f"{diff:+.2f}", help="Baseline MAE minus model MAE. Positive indicates lower MAE than baseline.")
    st.caption(f"Final-model rows: {metrics.get('final_model_training_rows', metrics.get('training_rows', 0)):,} · Holdout observations: {metrics.get('holdout_observations', 0):,} · Model: {metrics.get('model', 'Random Forest')}")
    st.warning(metrics.get("note", "Synthetic holdout metrics are not a guarantee of performance on real-world sales."))


def render_demo_dashboard(result: dict, tables: dict):
    recs = result.get("inventory_recommendations", [])
    rec_df = pd.DataFrame(recs)
    summary = result.get("summary", {})
    metrics = result.get("model_metrics", {})
    show_hero(result, "Synthetic demo")

    # Main KPIs
    section("Executive snapshot", "Live from the pre-trained demo model")
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("SKUs monitored", f"{result.get('sku_count', len(recs)):,}", help="Products across the synthetic retail assortment.")
    k2.metric("Sales observations", f"{result.get('sales_rows', 10000):,}", help="Daily SKU-level sales records used to fit the model.")
    k3.metric("Reorders to review", f"{summary.get('reorder_count', 0):,}", help="Products whose inventory position is at or below the reorder point.")
    k4.metric("Excess-stock reviews", f"{summary.get('promote_count', 0):,}", help=f"Estimated excess stock: {summary.get('projected_excess_units', 0):,.0f} units.")
    k5.metric("Forecast horizon", f"{result.get('horizon_days', 30)} days", help=f"Target service level: {result.get('service_level', 0.95):.0%}.")

    # Tabs
    overview, recommendations_tab, forecasts_tab, model_tab, data_tab = st.tabs([
        "◈  Command center", "☷  Recommendations", "⌁  Demand forecasts", "✳  Model lab", "▤  Data explorer"
    ])

    with overview:
        section("Inventory command center", "Risk signals and recommended next moves")
        left, right = st.columns([0.9, 1.35], gap="large")
        with left:
            with st.container(border=True):
                st.markdown("#### Recommended action mix")
                st.caption("How the assortment is distributed by next step")
                action_counts = rec_df["recommendation_type"].value_counts().rename_axis("Action").reset_index(name="SKUs")
                action_colors = {"Reorder": RED, "Hold": GREEN, "Promote": ORANGE}
                fig = go.Figure(data=[go.Pie(
                    labels=action_counts["Action"], values=action_counts["SKUs"], hole=0.68,
                    marker=dict(colors=[action_colors.get(v, TEAL) for v in action_counts["Action"]], line=dict(color="white", width=3)),
                    textinfo="label+value", textfont=dict(size=12), hovertemplate="%{label}<br>%{value} SKUs<extra></extra>"
                )])
                fig.update_layout(height=295, margin=dict(l=4, r=4, t=6, b=4), showlegend=False,
                                  annotations=[dict(text=f"{len(recs)}<br><span style='font-size:10px'>SKUs</span>", x=0.5, y=0.5, showarrow=False, font=dict(size=23, color=INK))],
                                  paper_bgcolor="white", font=dict(family="DM Sans", color=INK))
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        with right:
            with st.container(border=True):
                st.markdown("#### Stock position vs policy target")
                st.caption("Available stock compared with replenishment and target levels")
                bar_rows = []
                for rec in recs:
                    m = rec["metrics"]
                    bar_rows.extend([
                        {"Product": rec["product_id"], "Stock metric": "Available", "Units": m["available_stock"]},
                        {"Product": rec["product_id"], "Stock metric": "Reorder point", "Units": m["reorder_point"]},
                        {"Product": rec["product_id"], "Stock metric": "Target stock", "Units": m["target_stock"]},
                    ])
                chart_df = pd.DataFrame(bar_rows)
                fig = px.bar(chart_df, x="Product", y="Units", color="Stock metric", barmode="group",
                             color_discrete_map={"Available": TEAL, "Reorder point": RED, "Target stock": "#AAB5C5"})
                fig.update_layout(template="plotly_white", height=295, margin=dict(l=0, r=0, t=4, b=4),
                                  legend_title_text="", xaxis_title=None, yaxis_title="Units",
                                  font=dict(family="DM Sans", color=INK), paper_bgcolor="white", plot_bgcolor="white")
                fig.update_xaxes(showgrid=False, tickangle=-35)
                fig.update_yaxes(gridcolor="#EDF0F5", zeroline=False)
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        section("Priority queue", "Highest-impact inventory actions first")
        priorities = rec_df.copy()
        order = {"Reorder": 0, "Promote": 1, "Hold": 2}
        priorities["_priority"] = priorities.recommendation_type.map(order)
        priorities["_units"] = priorities["metrics"].map(lambda x: abs(x.get("projected_excess", 0)))
        priorities = priorities.sort_values(["_priority", "_units"], ascending=[True, False]).head(8)
        show_cols = ["product_name", "product_id", "category", "recommendation_type", "quantity", "timing", "forecast_demand_units"]
        st.dataframe(priorities[show_cols].rename(columns={
            "product_name": "Product", "product_id": "SKU", "category": "Category",
            "recommendation_type": "Action", "quantity": "Suggested units (+ order / − clear)",
            "timing": "Timing", "forecast_demand_units": f"Forecast ({result['horizon_days']}d)"
        }), use_container_width=True, hide_index=True, column_config={
            "Suggested units (+ order / − clear)": st.column_config.NumberColumn(format="%d"),
            f"Forecast ({result['horizon_days']}d)": st.column_config.NumberColumn(format="%.1f"),
        })
        st.caption("Demo data is synthetic. ‘Promote’ quantities are excess units to review for markdowns, bundles, or transfers—not a disposal instruction.")

    with recommendations_tab:
        section("Recommendation workbench", "Filter, investigate and export actionable decisions")
        c1, c2, c3 = st.columns([1.05, 1.1, 1.4])
        with c1:
            selected_actions = st.multiselect("Action type", ["Reorder", "Hold", "Promote"], default=["Reorder", "Hold", "Promote"])
        with c2:
            categories = sorted(rec_df.category.dropna().unique().tolist())
            selected_categories = st.multiselect("Category", categories, default=categories)
        with c3:
            search = st.text_input("Search SKU or product", placeholder="e.g. SKU-1001 or coffee")
        filtered = rec_df[rec_df.recommendation_type.isin(selected_actions) & rec_df.category.isin(selected_categories)].copy()
        if search:
            q = search.lower()
            filtered = filtered[filtered.apply(lambda row: q in str(row.get("product_id", "")).lower() or q in str(row.get("product_name", "")).lower(), axis=1)]
        table = pd.DataFrame({
            "SKU": filtered.product_id, "Product": filtered.product_name, "Category": filtered.category,
            "Warehouse": filtered.warehouse_id, "Action": filtered.recommendation_type,
            "Qty (+ order / − clear)": filtered.quantity, "When": filtered.timing,
            "Forecast units": filtered.forecast_demand_units,
            "Stock cover (days)": filtered.metrics.map(lambda m: m.get("days_of_cover")),
            "Lead time (days)": filtered.supplier_lead_time_days,
            "Reasoning": filtered.reasoning,
        })
        st.dataframe(table, use_container_width=True, hide_index=True, height=475, column_config={
            "Qty (+ order / − clear)": st.column_config.NumberColumn(format="%d"),
            "Forecast units": st.column_config.NumberColumn(format="%.1f"),
            "Stock cover (days)": st.column_config.NumberColumn(format="%.1f"),
        })
        st.markdown(f"**{len(filtered)}** recommendations match your filters.")
        csv_bytes = table.to_csv(index=False).encode("utf-8")
        st.download_button("Download filtered recommendations CSV", data=csv_bytes, file_name="inventory_recommendations.csv", mime="text/csv", type="primary")
        st.markdown("#### Investigate a recommendation")
        selected_pid = st.selectbox("Select SKU", rec_df.product_id.tolist(), format_func=lambda x: f"{x} · {rec_df.loc[rec_df.product_id == x, 'product_name'].iloc[0]}", key="inspect_sku")
        r = rec_df[rec_df.product_id == selected_pid].iloc[0]
        st.markdown(f"{action_pill(r.recommendation_type)} &nbsp; **{r.product_name}**", unsafe_allow_html=True)
        st.write(r.reasoning)
        m = r.metrics
        a, b, c, d = st.columns(4)
        a.metric("Available stock", f"{m['available_stock']:,.0f}")
        b.metric("Inventory position", f"{m['inventory_position']:,.0f}")
        c.metric("Reorder point", f"{m['reorder_point']:,.1f}")
        d.metric("Target stock", f"{m['target_stock']:,.1f}")

    with forecasts_tab:
        section("Demand forecast studio", "Observed demand meets the model's forward view")
        sku_names = {row.product_id: f"{row.product_id} · {row.product_name}" for row in tables["products"].itertuples(index=False)}
        pids = rec_df.product_id.tolist()
        default_idx = pids.index("SKU-1001") if "SKU-1001" in pids else 0
        selected_pid = st.selectbox("Product series", pids, index=default_idx, format_func=lambda x: sku_names.get(x, x), key="forecast_sku")
        rec = rec_df[rec_df.product_id == selected_pid].iloc[0]
        left, right = st.columns([1.6, 0.9], gap="large")
        with left:
            with st.container(border=True):
                st.markdown("#### Daily demand · recent 90 days + forecast")
                st.caption("Forecast generated recursively from the pre-trained global model")
                st.plotly_chart(create_forecast_figure(tables, result, selected_pid), use_container_width=True, config={"displayModeBar": False})
        with right:
            with st.container(border=True):
                st.markdown("#### SKU profile")
                st.caption("Inventory policy and model context")
                st.markdown(f"**{rec.product_name}**  \n`{rec.product_id}` · {rec.category}")
                st.metric("Forecast demand", f"{rec.forecast_demand_units:,.1f} units", help=f"Next {result['horizon_days']} days")
                st.metric("Days of cover", "—" if pd.isna(rec.metrics["days_of_cover"]) else f"{rec.metrics['days_of_cover']:.1f} days")
                st.metric("Supplier lead time", f"{rec.supplier_lead_time_days} days")
                st.markdown(f"Next step: {action_pill(rec.recommendation_type)}", unsafe_allow_html=True)
                st.markdown(f'<div class="small-note">{rec.reasoning}</div>', unsafe_allow_html=True)
        section("Demand by category", "Forecast volume over the selected horizon")
        forecast_by_cat = rec_df.groupby("category", as_index=False).forecast_demand_units.sum().sort_values("forecast_demand_units", ascending=False)
        fig = px.bar(forecast_by_cat, x="category", y="forecast_demand_units", text_auto=".0f",
                     color="forecast_demand_units", color_continuous_scale=["#B9E4DF", TEAL])
        fig.update_layout(template="plotly_white", height=280, margin=dict(l=0,r=0,t=12,b=0),
                          xaxis_title=None, yaxis_title="Forecast units", coloraxis_showscale=False,
                          paper_bgcolor="white", plot_bgcolor="white", font=dict(family="DM Sans", color=INK))
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(gridcolor="#EDF0F5")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with model_tab:
        st.markdown("### Model laboratory")
        st.caption("A global machine-learning model learns shared retail patterns across all 20 SKUs.")
        render_model_metrics(metrics)
        left, right = st.columns([1.25, 1], gap="large")
        with left:
            section("Feature importance", "Relative contribution inside the fitted Random Forest")
            importance = pd.DataFrame(metrics.get("feature_importance", []))
            if not importance.empty:
                importance = importance.head(12).sort_values("importance")
                fig = px.bar(importance, x="importance", y="feature", orientation="h", text_auto=".3f",
                             color="importance", color_continuous_scale=["#B9E4DF", TEAL])
                fig.update_layout(template="plotly_white", height=390, margin=dict(l=8,r=5,t=5,b=5),
                                  xaxis_title="Relative importance", yaxis_title=None,
                                  coloraxis_showscale=False, paper_bgcolor="white", plot_bgcolor="white",
                                  font=dict(family="DM Sans", color=INK))
                fig.update_xaxes(gridcolor="#EDF0F5")
                fig.update_yaxes(showgrid=False)
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        with right:
            section("Training snapshot", "Reproducible training artifact")
            st.markdown(f"""
            <div style="background:#F7F9FC;border:1px solid {BORDER};border-radius:14px;padding:17px 18px;">
              <div style="font-size:.73rem;color:{SLATE};font-weight:700;text-transform:uppercase;letter-spacing:.08em;">Algorithm</div>
              <div style="font-family:Manrope;font-weight:800;font-size:1.12rem;margin:4px 0 16px;">Random Forest Regressor</div>
              <div style="font-size:.73rem;color:{SLATE};font-weight:700;text-transform:uppercase;letter-spacing:.08em;">Training rows</div>
              <div style="font-family:Manrope;font-weight:800;font-size:1.6rem;margin:4px 0 16px;">{metrics.get('final_model_training_rows', metrics.get('training_rows', 0)):,}</div>
              <div style="font-size:.73rem;color:{SLATE};font-weight:700;text-transform:uppercase;letter-spacing:.08em;">Features</div>
              <div style="font-family:Manrope;font-weight:800;font-size:1.6rem;margin:4px 0 16px;">{len(metrics.get('features', []))}</div>
              <div style="font-size:.73rem;color:{SLATE};font-weight:700;text-transform:uppercase;letter-spacing:.08em;">Trained through</div>
              <div style="font-family:Manrope;font-weight:800;font-size:1.02rem;margin:4px 0 2px;">{metrics.get('history_end', '—')}</div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("**Included signals**")
            st.write("Lagged demand (1/7/14/28 days), rolling means and variability, SKU/store/category identity, price, promotion calendar, weekday, month, and annual seasonality.")
        st.markdown("#### Retrain the model")
        st.write("Retraining regenerates the saved model artifact from the bundled 10,000-row synthetic dataset and refreshes the 28-day time-based holdout metrics.")
        if st.button("⟳  Retrain on synthetic dataset", type="primary", key="retrain_model"):
            with st.spinner("Training global model and evaluating the final 28 days..."):
                new_metrics = train_and_save(DATA_DIR, ROOT / "models", METRICS_PATH, holdout_days=28)
                st.cache_data.clear()
                st.success(f"Model retrained. Holdout MAE: {new_metrics['model_mae_units_per_day']:.2f} units/day; WAPE: {new_metrics['model_wape_percent']:.1f}%.")
                st.rerun()

    with data_tab:
        section("Synthetic data explorer", "A fully reproducible dataset — no Kaggle account or external download required")
        st.markdown(f"""
        <div style="background:#EAF6F4;border:1px solid #C7E8E2;border-radius:14px;padding:14px 16px;color:#135D59;margin-bottom:14px;">
          <strong>Demo data · clearly labeled synthetic</strong><br>
          10,000 daily SKU-level demand observations across 20 products and 4 warehouse zones, covering {result.get('source_data_start_date')} to {result.get('source_data_end_date')}. The generated history includes weekday effects, annual seasonality, promo lift, calendar events, gradual trend, intermittent demand, price variation and occasional demand spikes.
        </div>
        """, unsafe_allow_html=True)
        st.download_button("Download all demo data (ZIP)", data=make_dataset_zip(), file_name="stockwise_synthetic_retail_dataset.zip", mime="application/zip", type="primary")
        data1, data2, data3, data4 = st.tabs(["Sales history", "Inventory snapshot", "Product catalog", "Supplier terms"])
        with data1:
            sales = tables["sales"].copy()
            c1, c2 = st.columns(2)
            with c1: category_filter = st.selectbox("Filter sales by category", ["All"] + sorted(tables["products"].category.unique().tolist()), key="sales_category")
            with c2: last_days = st.select_slider("Sales history window", options=[30, 60, 90, 180, 365, 500], value=90)
            if category_filter != "All":
                keep = tables["products"].loc[tables["products"].category == category_filter, "product_id"]
                sales = sales[sales.product_id.isin(keep)]
            max_dt = sales.date.max()
            sales = sales[sales.date >= max_dt - pd.Timedelta(days=last_days-1)]
            st.dataframe(sales, use_container_width=True, hide_index=True, height=365)
            st.caption(f"Showing {len(sales):,} rows in the selected time window.")
        with data2:
            st.dataframe(tables["inventory"].merge(tables["products"][["product_id","product_name","category"]], on="product_id", how="left"), use_container_width=True, hide_index=True)
        with data3:
            st.dataframe(tables["products"], use_container_width=True, hide_index=True)
        with data4:
            st.dataframe(tables["suppliers"], use_container_width=True, hide_index=True)
        st.markdown("#### Downloadable demo artifacts")
        sales_csv = tables["sales"].to_csv(index=False).encode("utf-8")
        st.download_button("Sales history CSV (10,000 rows)", data=sales_csv, file_name="sample_sales.csv", mime="text/csv")
        st.download_button("Product catalog CSV", data=tables["products"].to_csv(index=False).encode("utf-8"), file_name="sample_products.csv", mime="text/csv")
        st.download_button("Inventory snapshot CSV", data=tables["inventory"].to_csv(index=False).encode("utf-8"), file_name="sample_inventory.csv", mime="text/csv")

    # Downloads in a persistent report section at the bottom of the page.
    st.divider()
    section("Take the analysis with you", "Files are generated from the current dashboard settings")
    d1, d2, d3 = st.columns(3)
    with d1:
        st.download_button("↓  Recommendations JSON", data=json.dumps(result, indent=2, default=str).encode("utf-8"),
                           file_name="stockwise_inventory_recommendations.json", mime="application/json", use_container_width=True)
    with d2:
        st.download_button("↓  Recommendations CSV", data=rec_df.drop(columns=["metrics"], errors="ignore").to_csv(index=False).encode("utf-8"),
                           file_name="stockwise_inventory_recommendations.csv", mime="text/csv", use_container_width=True)
    with d3:
        pdf = build_pdf_report(result)
        st.download_button("↓  Executive PDF report", data=pdf, file_name="stockwise_inventory_report.pdf", mime="application/pdf", use_container_width=True, type="primary")
    st.markdown('<div class="small-note">Stockwise is a demonstration decision-support tool. Synthetic inventory levels, purchase quantities and supplier lead times are illustrative only; a human planner should validate actions before ordering or markdown decisions.</div>', unsafe_allow_html=True)


def render_uploaded_dashboard(uploaded_inputs: dict, horizon: int, service_level: float, minimum_history: int):
    show_hero({"sales_rows": len(uploaded_inputs["sales"]), "source_data_end_date": "uploaded data"}, "Upload")
    st.warning("Custom-upload mode trains SKU-level forecasts from the provided history; validate input units, stockout-censored demand, order multiples and supplier lead times before acting.")
    with st.expander("Preview uploaded data", expanded=False):
        tabs = st.tabs(["Sales", "Inventory", "Products", "Suppliers"])
        for tab, key in zip(tabs, ["sales", "inventory", "products", "suppliers"]):
            with tab:
                st.dataframe(uploaded_inputs[key].head(20), use_container_width=True, hide_index=True)
    if st.button("Run inventory intelligence on uploaded data", type="primary"):
        with st.spinner("Validating uploads, forecasting demand, and generating recommendations..."):
            try:
                result = generate_recommendations(
                    uploaded_inputs["sales"], uploaded_inputs["inventory"], uploaded_inputs["products"], uploaded_inputs["suppliers"],
                    horizon=horizon, service_level=service_level, minimum_history_days=minimum_history
                )
                st.session_state["custom_result"] = result
                st.success("Recommendations generated.")
            except (DataValidationError, ValueError) as exc:
                st.error(f"Data validation failed: {exc}")
            except Exception as exc:
                st.error(f"Analysis failed: {type(exc).__name__}: {exc}")
    result = st.session_state.get("custom_result")
    if result:
        recs = result.get("inventory_recommendations", [])
        summary = result.get("summary", {})
        section("Results snapshot", "Based on your uploaded data")
        a, b, c, d = st.columns(4)
        a.metric("SKUs evaluated", summary.get("products_evaluated", 0))
        b.metric("Reorder", summary.get("reorder_count", 0))
        c.metric("Hold", summary.get("hold_count", 0))
        d.metric("Promote", summary.get("promote_count", 0))
        frame = pd.json_normalize(recs)
        cols = [c for c in ["product_id", "warehouse_id", "recommendation_type", "quantity", "timing", "forecast_demand_units", "supplier_lead_time_days", "reasoning"] if c in frame]
        st.dataframe(frame[cols], use_container_width=True, hide_index=True)
        x, y, z = st.columns(3)
        x.download_button("Download JSON", data=json.dumps(result, indent=2, default=str).encode(), file_name="inventory_recommendations.json", mime="application/json")
        y.download_button("Download CSV", data=frame.to_csv(index=False).encode(), file_name="inventory_recommendations.csv", mime="text/csv")
        z.download_button("Download PDF", data=build_pdf_report(result), file_name="inventory_report.pdf", mime="application/pdf")


# ---- Sidebar controls --------------------------------------------------------
with st.sidebar:
    st.markdown("<div style='font-family:Manrope;font-weight:800;font-size:1.35rem;letter-spacing:-.04em;'>stockwise<span style='color:#86DFD4'>.</span></div><div style='font-size:.74rem;color:#9FAEC4;margin-top:2px;'>INVENTORY INTELLIGENCE STUDIO</div>", unsafe_allow_html=True)
    st.divider()
    mode = st.radio("Workspace", ["Synthetic demo", "Upload your data"], index=0)
    st.markdown("#### Decision settings")
    horizon = st.slider("Forecast horizon", min_value=7, max_value=60, value=30, step=1, help="Forecast days used in stock-cover and inventory-policy calculations.")
    service_level = st.slider("Target service level", min_value=0.80, max_value=0.99, value=0.95, step=0.01)
    minimum_history = st.slider("Minimum history for ML", min_value=30, max_value=180, value=60, step=15)
    st.caption("Higher service levels usually increase safety stock. Forecasts are estimates, not purchase orders.")
    st.divider()
    st.markdown("#### Demo status")
    st.markdown('<span style="display:inline-block;width:8px;height:8px;border-radius:100%;background:#43D9A3;margin-right:7px;"></span><span style="font-size:.82rem;font-weight:700;">Model artifact ready</span>', unsafe_allow_html=True)
    st.caption("Global Random Forest · 19 engineered features")
    st.markdown("[Project README](README.md) · [Technical deep dive](docs/technical_deep_dive.md)")

if mode == "Synthetic demo":
    with st.spinner("Loading trained model and scoring the synthetic retail assortment..."):
        try:
            result = cached_result(horizon, service_level)
            tables = cached_tables()
            render_demo_dashboard(result, tables)
        except Exception as exc:
            st.error(f"Could not load the demo: {type(exc).__name__}: {exc}")
            st.code("python data/generate_demo_data.py\npython train_demo_model.py", language="bash")
else:
    st.markdown('<div class="hero"><div class="hero-eyebrow">Retail operations · Bring your own data</div><h1>Connect your inventory data.</h1><p>Upload four clean data files using the documented schemas. The system validates your data and produces explainable replenishment recommendations.</p><div class="hero-bottom"><span class="hero-chip">CSV / Excel</span><span class="hero-chip">Four data sources</span><span class="hero-chip">JSON · CSV · PDF exports</span></div></div>', unsafe_allow_html=True)
    with st.expander("Required data files and formats", expanded=True):
        st.markdown("**Sales:** `date, product_id, units_sold` · optional `warehouse_id, promotion_flag, holiday_flag, unit_price`  \n**Inventory:** `product_id, on_hand` · optional `warehouse_id, on_order, reserved`  \n**Products:** `product_id` · optional category, cost, price, pack size and minimum order quantity  \n**Suppliers:** `product_id, lead_time_days` · optional supplier ID and review period. See `docs/data_dictionary.md`.")
    c1, c2 = st.columns(2)
    with c1:
        sales_u = st.file_uploader("01 · Sales history", type=["csv", "xlsx", "xls"], key="upload_sales")
        inventory_u = st.file_uploader("02 · Inventory snapshot", type=["csv", "xlsx", "xls"], key="upload_inventory")
    with c2:
        products_u = st.file_uploader("03 · Product master", type=["csv", "xlsx", "xls"], key="upload_products")
        suppliers_u = st.file_uploader("04 · Supplier lead times", type=["csv", "xlsx", "xls"], key="upload_suppliers")
    if all([sales_u, inventory_u, products_u, suppliers_u]):
        try:
            payload = {"sales": load_upload(sales_u), "inventory": load_upload(inventory_u),
                       "products": load_upload(products_u), "suppliers": load_upload(suppliers_u)}
            render_uploaded_dashboard(payload, horizon, service_level, minimum_history)
        except Exception as exc:
            st.error(f"Unable to read files: {type(exc).__name__}: {exc}")
    else:
        st.info("Upload all four data files to enable the analysis button.")
