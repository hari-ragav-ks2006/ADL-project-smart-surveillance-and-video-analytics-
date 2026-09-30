"""
Streamlit Security & Surveillance Analytics Dashboard.
Standalone post-incident review, alert analytics, and evidence gallery.
"""

from datetime import datetime
from pathlib import Path
import json
import os
import pandas as pd
from PIL import Image
import streamlit as st

from database import AlertRecord, DatabaseManager, SystemLogRecord

st.set_page_config(
    page_title="SentinelEye Surveillance Analytics",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main { background-color: #0B0F19; }
    .stMetric {
        background-color: #111827;
        padding: 16px;
        border-radius: 10px;
        border: 1px solid rgba(255,255,255,0.08);
    }
    .badge-critical { color: #EF4444; font-weight: bold; }
    .badge-warning { color: #F59E0B; font-weight: bold; }
    .badge-safe { color: #10B981; font-weight: bold; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_db():
    return DatabaseManager("logs/surveillance.db")


db_manager = get_db()


def load_alert_data() -> pd.DataFrame:
    with db_manager.get_session() as session:
        alerts = session.query(AlertRecord).order_by(AlertRecord.timestamp.desc()).all()
        if not alerts:
            return pd.DataFrame()
        return pd.DataFrame([a.to_dict() for a in alerts])


def load_system_logs() -> pd.DataFrame:
    with db_manager.get_session() as session:
        logs = session.query(SystemLogRecord).order_by(SystemLogRecord.timestamp.desc()).limit(100).all()
        if not logs:
            return pd.DataFrame()
        return pd.DataFrame([l.to_dict() for l in logs])


# Sidebar
st.sidebar.title("🛡️ SentinelEye Analytics")
st.sidebar.caption("Applied Deep Learning Capstone Project")

menu = st.sidebar.radio(
    "Navigation",
    ["📊 Analytics & KPIs", "📋 Incident Records", "🖼️ Evidence Gallery", "⚙️ System Logs"],
)

st.sidebar.markdown("---")
if st.sidebar.button("🔄 Refresh Data"):
    st.cache_data.clear()

df_alerts = load_alert_data()

# 1. Analytics & KPIs
if menu == "📊 Analytics & KPIs":
    st.title("🛡️ Surveillance Analytics & Key Metrics")
    st.caption("Aggregated analytics from YOLOv8 + ByteTrack surveillance events")

    if df_alerts.empty:
        st.info("No alert records logged yet. Run surveillance pipeline (`python main.py`) to generate events.")
    else:
        # Top KPI row
        k1, k2, k3, k4 = st.columns(4)
        total_alerts = len(df_alerts)
        critical_alerts = len(df_alerts[df_alerts["severity"] == "CRITICAL"])
        active_alerts = len(df_alerts[df_alerts["resolution_status"] == "ACTIVE"])
        resolved_alerts = len(df_alerts[df_alerts["resolution_status"] == "RESOLVED"])

        k1.metric("Total Logged Alerts", total_alerts)
        k2.metric("Critical / Blackout Alerts", critical_alerts, delta_color="inverse")
        k3.metric("Active Alerts", active_alerts)
        k4.metric("Resolved Incidents", resolved_alerts)

        st.markdown("---")

        c1, c2 = st.columns(2)

        with c1:
            st.subheader("Alerts Breakdown by Type")
            type_counts = df_alerts["alert_type"].value_counts()
            st.bar_chart(type_counts)

        with c2:
            st.subheader("Alert Distribution by Severity")
            sev_counts = df_alerts["severity"].value_counts()
            st.bar_chart(sev_counts)

        st.subheader("Recent Alert Timeline")
        df_display = df_alerts[["id", "timestamp", "alert_type", "severity", "module_name", "message", "resolution_status"]].head(10)
        st.dataframe(df_display, use_container_width=True)


# 2. Incident Records
elif menu == "📋 Incident Records":
    st.title("📋 Incident Records & Full Audit Log")

    if df_alerts.empty:
        st.info("No records to display.")
    else:
        f1, f2, f3 = st.columns(3)
        with f1:
            sev_filter = st.multiselect("Filter by Severity", options=df_alerts["severity"].unique().tolist())
        with f2:
            type_filter = st.multiselect("Filter by Alert Type", options=df_alerts["alert_type"].unique().tolist())
        with f3:
            status_filter = st.multiselect("Filter by Status", options=df_alerts["resolution_status"].unique().tolist())

        filtered_df = df_alerts.copy()
        if sev_filter:
            filtered_df = filtered_df[filtered_df["severity"].isin(sev_filter)]
        if type_filter:
            filtered_df = filtered_df[filtered_df["alert_type"].isin(type_filter)]
        if status_filter:
            filtered_df = filtered_df[filtered_df["resolution_status"].isin(status_filter)]

        st.dataframe(filtered_df, use_container_width=True)

        # Excel & CSV Download
        excel_path = "logs/surveillance_alerts.xlsx"
        if os.path.exists(excel_path):
            with open(excel_path, "rb") as f:
                st.download_button(
                    label="📥 Download Full Excel Report (.xlsx)",
                    data=f,
                    file_name="surveillance_alerts.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )


# 3. Evidence Gallery
elif menu == "🖼️ Evidence Gallery":
    st.title("🖼️ Incident Screenshot Evidence Gallery")

    if df_alerts.empty or "screenshot_path" not in df_alerts.columns:
        st.info("No evidence screenshots found.")
    else:
        shots_df = df_alerts[df_alerts["screenshot_path"].notna() & (df_alerts["screenshot_path"] != "")]
        if shots_df.empty:
            st.info("No screenshots captured yet.")
        else:
            cols = st.columns(3)
            for idx, (_, row) in enumerate(shots_df.iterrows()):
                col = cols[idx % 3]
                img_path = Path(row["screenshot_path"])
                if img_path.exists():
                    try:
                        img = Image.open(img_path)
                        with col:
                            st.image(
                                img,
                                caption=f"#{row['id']} - {row['alert_type']} ({row['severity']})",
                                use_container_width=True,
                            )
                            st.caption(f"Time: {row['timestamp']}")
                    except Exception:
                        pass


# 4. System Logs
elif menu == "⚙️ System Logs":
    st.title("⚙️ Operational & System Event Logs")
    df_logs = load_system_logs()
    if df_logs.empty:
        st.info("No system events logged.")
    else:
        st.dataframe(df_logs, use_container_width=True)
