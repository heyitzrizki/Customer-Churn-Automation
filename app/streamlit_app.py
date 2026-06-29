from pathlib import Path
from html import escape

import pandas as pd
import plotly.express as px
import streamlit as st


BASE_DIR = Path(__file__).resolve().parents[1]
CUSTOMERS_SCORED_PATH = BASE_DIR / "data" / "processed" / "customers_scored.csv"
SEGMENTS_PATH = BASE_DIR / "data" / "processed" / "customer_segments.csv"
N8N_WORKFLOW_IMAGE = BASE_DIR / "app" / "assets" / "n8n_workflow.png"


st.set_page_config(page_title="Customer Retention Intelligence", layout="wide")

st.markdown(
    """
    <style>
    :root {
        --app-bg: #0f1117;
        --panel: #171b22;
        --panel-soft: #1f242d;
        --border: #303744;
        --text: #f6f7fb;
        --muted: #aab2c0;
        --teal: #20c7b5;
        --amber: #f5b642;
        --rose: #ff5f6d;
        --blue: #65a8ff;
    }

    .block-container {
        max-width: 1540px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    h1 {
        font-size: 2.05rem !important;
        line-height: 1.15 !important;
        margin-bottom: 0.25rem !important;
    }

    h2, h3 {
        letter-spacing: 0 !important;
    }

    section[data-testid="stSidebar"] {
        background: #22242d;
        border-right: 1px solid #333846;
    }

    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] label {
        color: var(--text) !important;
    }

    .section-panel {
        border: 1px solid var(--border);
        background: var(--panel);
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0 1rem;
    }

    .metric-card {
        border: 1px solid var(--border);
        border-left: 4px solid var(--accent);
        border-radius: 8px;
        background: linear-gradient(180deg, #1a1f27 0%, #13171e 100%);
        padding: 1rem 1rem 0.9rem;
        min-height: 116px;
        box-shadow: 0 8px 22px rgba(0, 0, 0, 0.18);
    }

    .metric-label {
        color: var(--muted);
        font-size: 0.78rem;
        font-weight: 700;
        line-height: 1.25;
        min-height: 2.1rem;
    }

    .metric-value {
        color: var(--text);
        font-size: 1.9rem;
        font-weight: 800;
        line-height: 1.1;
        margin-top: 0.35rem;
    }

    .metric-note {
        color: var(--muted);
        font-size: 0.78rem;
        line-height: 1.25;
        margin-top: 0.35rem;
    }

    .business-note {
        border: 1px solid var(--border);
        border-left: 4px solid var(--teal);
        border-radius: 8px;
        background: #151a20;
        padding: 0.8rem 1rem;
        color: var(--muted);
        margin: 0.5rem 0 1rem;
    }

    div[data-testid="stExpander"] {
        border-color: var(--border) !important;
        background: #13171d !important;
    }

    div[data-baseweb="select"] > div {
        background: #1f242d !important;
        border-color: #3a4250 !important;
        border-radius: 8px !important;
        min-height: 42px;
    }

    div[data-baseweb="select"] span {
        color: var(--text) !important;
    }

    .stDataFrame {
        border: 1px solid var(--border);
        border-radius: 8px;
        overflow: hidden;
    }

    div[data-testid="stDownloadButton"] button {
        border-radius: 8px;
        border: 1px solid #25b7a7;
        background: #12645e;
        color: white;
        font-weight: 700;
    }

    div[data-testid="stDownloadButton"] button:hover {
        border-color: #35d8c6;
        background: #16766f;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

COLOR_SEQUENCE = ["#20c7b5", "#f5b642", "#ff5f6d", "#65a8ff", "#9b8cff", "#7ad37a"]


def metric_card(label, value, note="", accent="#20c7b5"):
    st.markdown(
        f"""
        <div class="metric-card" style="--accent:{accent}">
            <div class="metric-label">{escape(str(label))}</div>
            <div class="metric-value">{escape(str(value))}</div>
            <div class="metric-note">{escape(str(note))}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def business_note(text):
    st.markdown(f'<div class="business-note">{escape(text)}</div>', unsafe_allow_html=True)


@st.cache_data
def load_data():
    if not CUSTOMERS_SCORED_PATH.exists() or not SEGMENTS_PATH.exists():
        return None, None

    scored = pd.read_csv(CUSTOMERS_SCORED_PATH)
    segments = pd.read_csv(SEGMENTS_PATH)
    merged = scored.merge(
        segments[["customer_id", "segment_description", "suggested_strategy", "usage_minutes"]],
        on="customer_id",
        how="left",
    )
    return merged, segments


def format_percent(value):
    return f"{value * 100:.1f}%"


def select_filter(label, values):
    options = ["All"] + sorted([value for value in values.dropna().unique()])
    return st.selectbox(label, options)


def apply_single_filter(data, column, value):
    if value == "All":
        return data
    return data[data[column] == value]


def missing_data_message():
    st.title("Customer Retention Intelligence")
    st.warning("Processed customer files are not available yet.")
    st.code(
        "python src/train_models.py\npython src/build_segments.py\npython src/score_customers.py",
        language="bash",
    )


def segment_summary(scored):
    return (
        scored.groupby("segment")
        .agg(
            Customers=("customer_id", "count"),
            Avg_Customer_Value=("customer_value", "mean"),
            Avg_Usage=("usage_minutes", "mean"),
            Complaint_Rate=("main_reason", lambda values: values.str.contains("Complaint recorded", na=False).mean()),
            Avg_Churn_Risk=("churn_probability", "mean"),
            Suggested_Strategy=("suggested_strategy", "first"),
            Segment_Description=("segment_description", "first"),
        )
        .reset_index()
        .rename(
            columns={
                "segment": "Segment",
                "Avg_Customer_Value": "Avg Customer Value",
                "Avg_Usage": "Avg Usage",
                "Complaint_Rate": "Complaint Rate",
                "Avg_Churn_Risk": "Avg Churn Risk",
                "Suggested_Strategy": "Suggested Strategy",
                "Segment_Description": "Segment Description",
            }
        )
    )


def customer_segmentation_page(scored):
    st.title("Customer Segmentation")
    st.caption("What types of customers do we have, and how should each segment be managed?")

    summary = segment_summary(scored)

    card_columns = st.columns(min(4, len(summary)))
    for index, row in summary.head(4).iterrows():
        with card_columns[index % len(card_columns)]:
            metric_card(
                row["Segment"],
                f"{int(row['Customers']):,}",
                f"Avg risk {format_percent(row['Avg Churn Risk'])}",
                COLOR_SEQUENCE[index % len(COLOR_SEQUENCE)],
            )

    left, right = st.columns([1, 1])
    with left:
        count_chart = px.bar(
            summary,
            x="Segment",
            y="Customers",
            color="Segment",
            text="Customers",
            title="Customer Count by Segment",
            color_discrete_sequence=COLOR_SEQUENCE,
            template="plotly_dark",
        )
        count_chart.update_layout(
            showlegend=False,
            xaxis_title="",
            yaxis_title="Customers",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10, r=10, t=52, b=10),
        )
        count_chart.update_traces(textposition="outside", marker_line_width=0)
        st.plotly_chart(count_chart, width="stretch")

    with right:
        st.dataframe(
            summary[
                [
                    "Segment",
                    "Customers",
                    "Avg Customer Value",
                    "Avg Usage",
                    "Complaint Rate",
                    "Avg Churn Risk",
                    "Suggested Strategy",
                ]
            ].style.format(
                {
                    "Avg Customer Value": "{:,.1f}",
                    "Avg Usage": "{:,.1f}",
                    "Complaint Rate": "{:.1%}",
                    "Avg Churn Risk": "{:.1%}",
                }
            ),
            width="stretch",
            hide_index=True,
            height=360,
        )

    selected_segment = st.selectbox("Segment detail", summary["Segment"].tolist())
    selected = summary.loc[summary["Segment"] == selected_segment].iloc[0]
    detail_columns = st.columns(4)
    with detail_columns[0]:
        metric_card("Customers", f"{int(selected['Customers']):,}", "Segment size", "#20c7b5")
    with detail_columns[1]:
        metric_card("Avg Churn Risk", format_percent(selected["Avg Churn Risk"]), "Segment average", "#ff5f6d")
    with detail_columns[2]:
        metric_card("Avg Customer Value", f"{selected['Avg Customer Value']:,.1f}", "Value proxy", "#f5b642")
    with detail_columns[3]:
        metric_card("Complaint Rate", format_percent(selected["Complaint Rate"]), "Service issue signal", "#65a8ff")

    st.subheader("Business View")
    st.write(f"**Who they are:** {selected['Segment Description']}")
    st.write(f"**Why they matter:** This group represents {int(selected['Customers']):,} customers with an average value of {selected['Avg Customer Value']:,.1f}.")
    st.write(f"**Main risk pattern:** Average churn risk is {format_percent(selected['Avg Churn Risk'])}, with a complaint rate of {format_percent(selected['Complaint Rate'])}.")
    st.write(f"**Recommended business action:** {selected['Suggested Strategy']}")

    st.subheader("Segmentation Map")
    scatter = px.scatter(
        scored,
        x="customer_value",
        y="churn_probability",
        color="segment",
        hover_data=["customer_id", "risk_level", "recommended_action"],
        labels={
            "customer_value": "Customer Value",
            "churn_probability": "Churn Risk Score",
            "segment": "Customer Segment",
        },
        color_discrete_sequence=COLOR_SEQUENCE,
        template="plotly_dark",
    )
    scatter.add_hline(y=0.65, line_dash="dash", line_color="#d85f5f")
    scatter.add_vline(x=scored["customer_value"].median(), line_dash="dash", line_color="#7b8794")
    scatter.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend_title_text="Customer Segment",
        margin=dict(l=10, r=10, t=20, b=10),
    )
    st.plotly_chart(scatter, width="stretch")
    business_note(
        "High value + high risk: priority retention. High value + low risk: loyalty maintain. "
        "Low value + high risk: automated campaign. Low value + low risk: low priority."
    )


def churn_risk_scoring_page(scored):
    st.title("Churn Risk Scoring")
    st.caption("Which customers should we act on, why, and what should we do next?")

    high_risk = scored[scored["risk_level"] == "High"]
    high_value_high_risk = high_risk[high_risk["customer_value_tier"] == "High"]
    revenue_at_risk = high_risk["customer_value"].sum()
    first_target_size = len(scored[scored["priority"].isin(["P1", "P2"])])

    cards = st.columns(5)
    with cards[0]:
        metric_card("Total Customers Scored", f"{len(scored):,}", "Full scored base", "#20c7b5")
    with cards[1]:
        metric_card("High-Risk Customers", f"{len(high_risk):,}", "Churn risk >= 65%", "#ff5f6d")
    with cards[2]:
        metric_card("High-Value High-Risk", f"{len(high_value_high_risk):,}", "Best first calls", "#f5b642")
    with cards[3]:
        metric_card("Estimated Value at Risk", f"{revenue_at_risk:,.0f}", "Value proxy", "#65a8ff")
    with cards[4]:
        metric_card("Recommended First Target", f"{first_target_size:,}", "P1 and P2 customers", "#9b8cff")
    business_note("Estimated value at risk is a customer value proxy, not proven lost revenue.")

    with st.expander("Filters", expanded=True):
        filter_columns = st.columns(4)
        with filter_columns[0]:
            risk_filter = select_filter("Risk level", scored["risk_level"])
        with filter_columns[1]:
            segment_filter = select_filter("Customer segment", scored["segment"])
        with filter_columns[2]:
            action_filter = select_filter("Recommended action", scored["recommended_action"])
        with filter_columns[3]:
            value_filter = select_filter("Customer value tier", scored["customer_value_tier"])

    filtered = scored.copy()
    filtered = apply_single_filter(filtered, "risk_level", risk_filter)
    filtered = apply_single_filter(filtered, "segment", segment_filter)
    filtered = apply_single_filter(filtered, "recommended_action", action_filter)
    filtered = apply_single_filter(filtered, "customer_value_tier", value_filter)

    st.caption(f"Showing {len(filtered):,} of {len(scored):,} customers.")

    chart_left, chart_right = st.columns([1.15, 1])
    with chart_left:
        risk_chart_data = (
            filtered.groupby("risk_level")
            .agg(Customers=("customer_id", "count"))
            .reindex(["High", "Medium", "Low"])
            .dropna()
            .reset_index()
        )
        risk_chart = px.bar(
            risk_chart_data,
            x="risk_level",
            y="Customers",
            color="risk_level",
            text="Customers",
            title="Customers by Risk Level",
            color_discrete_map={"High": "#ff5f6d", "Medium": "#f5b642", "Low": "#20c7b5"},
            template="plotly_dark",
            labels={"risk_level": "Risk Level"},
        )
        risk_chart.update_layout(
            showlegend=False,
            xaxis_title="",
            yaxis_title="Customers",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10, r=10, t=52, b=10),
        )
        risk_chart.update_traces(textposition="outside")
        st.plotly_chart(risk_chart, width="stretch")

    with chart_right:
        action_chart_data = (
            filtered.groupby("recommended_action")
            .agg(Customers=("customer_id", "count"))
            .sort_values("Customers", ascending=False)
            .head(6)
            .reset_index()
        )
        action_chart = px.bar(
            action_chart_data,
            x="Customers",
            y="recommended_action",
            orientation="h",
            color="recommended_action",
            title="Top Recommended Actions",
            color_discrete_sequence=COLOR_SEQUENCE,
            template="plotly_dark",
            labels={"recommended_action": "Recommended Action"},
        )
        action_chart.update_layout(
            showlegend=False,
            xaxis_title="Customers",
            yaxis_title="",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10, r=10, t=52, b=10),
        )
        st.plotly_chart(action_chart, width="stretch")

    table = filtered[
        [
            "customer_id",
            "segment",
            "churn_probability",
            "risk_level",
            "retention_priority_score",
            "customer_value",
            "main_reason",
            "recommended_action",
            "priority",
        ]
    ].rename(
        columns={
            "customer_id": "Customer ID",
            "segment": "Segment",
            "churn_probability": "Churn Risk",
            "risk_level": "Risk Level",
            "retention_priority_score": "Retention Priority Score",
            "customer_value": "Customer Value",
            "main_reason": "Main Reason",
            "recommended_action": "Recommended Action",
            "priority": "Priority",
        }
    )
    table["Churn Risk"] = table["Churn Risk"] * 100
    st.dataframe(
        table,
        width="stretch",
        hide_index=True,
        height=420,
        column_config={
            "Customer ID": st.column_config.TextColumn("Customer ID", width="small"),
            "Segment": st.column_config.TextColumn("Segment", width="medium"),
            "Churn Risk": st.column_config.ProgressColumn(
                "Churn Risk",
                format="%.1f%%",
                min_value=0,
                max_value=100,
                width="small",
            ),
            "Risk Level": st.column_config.TextColumn("Risk Level", width="small"),
            "Retention Priority Score": st.column_config.ProgressColumn(
                "Retention Priority Score",
                format="%d",
                min_value=0,
                max_value=100,
                width="medium",
            ),
            "Customer Value": st.column_config.NumberColumn("Customer Value", format="%.1f", width="small"),
            "Main Reason": st.column_config.TextColumn("Main Reason", width="large"),
            "Recommended Action": st.column_config.TextColumn("Recommended Action", width="medium"),
            "Priority": st.column_config.TextColumn("Priority", width="small"),
        },
    )

    st.download_button(
        "Download retention target list CSV",
        data=filtered.to_csv(index=False),
        file_name="retention_target_list.csv",
        mime="text/csv",
    )

    if not filtered.empty:
        selected_customer_id = st.selectbox("Customer detail", filtered["customer_id"].tolist())
        customer = filtered.loc[filtered["customer_id"] == selected_customer_id].iloc[0]

        detail_columns = st.columns(4)
        with detail_columns[0]:
            metric_card("Churn Risk Score", format_percent(customer["churn_probability"]), customer["risk_level"], "#ff5f6d")
        with detail_columns[1]:
            metric_card("Customer Value Tier", customer["customer_value_tier"], f"{customer['customer_value']:,.1f}", "#f5b642")
        with detail_columns[2]:
            metric_card("Retention Priority Score", int(customer["retention_priority_score"]), "0 to 100", "#20c7b5")
        with detail_columns[3]:
            metric_card("Priority", customer["priority"], customer["recommended_action"], "#9b8cff")

        st.markdown(
            f"""
            <div class="section-panel">
                <strong>Segment</strong><br>{escape(str(customer['segment']))}<br><br>
                <strong>Main risk drivers</strong><br>{escape(str(customer['main_reason']))}<br><br>
                <strong>Recommended action</strong><br>{escape(str(customer['recommended_action']))}
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.subheader("Automation Workflow")
    st.write(
        "The churn model is connected to n8n so high-risk customers can trigger Telegram alerts "
        "and HubSpot CRM updates with the recommended retention action."
    )
    if N8N_WORKFLOW_IMAGE.exists():
        st.image(str(N8N_WORKFLOW_IMAGE), caption="n8n retention automation workflow")
    else:
        st.info("Add `app/assets/n8n_workflow.png` to show the n8n automation workflow here.")


scored_data, segment_data = load_data()
if scored_data is None:
    missing_data_message()
else:
    st.sidebar.title("Retention Intelligence")
    page = st.sidebar.radio("Page", ["Customer Segmentation", "Churn Risk Scoring"])

    if page == "Customer Segmentation":
        customer_segmentation_page(scored_data)
    else:
        churn_risk_scoring_page(scored_data)
