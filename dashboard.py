
# ============================================================
# 4.6 DASHBOARD — Real-time Decision Support Visualization
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pymongo import MongoClient
import time, os
from PIL import Image
from datetime import datetime, timezone

# ── PAGE CONFIG ──────────────────────────────────────────────────────────

st.set_page_config(
    page_title="X-Ray AI Dashboard",
    page_icon="🫁",
    layout="wide",
)

# ── STYLING ──────────────────────────────────────────────────────────────

st.markdown('''
<style>
.stApp { background:#FAFAFA; color:#212121; }

.kpi-card {
    background:#fff; border-radius:12px; padding:18px 22px;
    border-left:5px solid #1565C0;
    box-shadow:0 2px 8px rgba(0,0,0,.08); margin:6px 0;
}

.alert-box {
    background:#FFF3E0; border-left:5px solid #E53935;
    border-radius:8px; padding:10px 14px; margin:4px 0;
}

.section-title {
    font-size:1.1rem; font-weight:700; color:#1565C0;
    border-bottom:2px solid #BBDEFB;
    padding-bottom:4px; margin:14px 0 10px;
}

.success-box {
    background:#E8F5E9; border-left:5px solid #43A047;
    border-radius:8px; padding:10px 14px; margin:4px 0;
}
</style>
''', unsafe_allow_html=True)

# ── MONGODB CONNECTION ────────────────────────────────────────────────────

MONGO_URI = "mongodb+srv://tuyettrinh3525:Trinh3005@clusterbigdata.ubhpjjc.mongodb.net/?appName=ClusterBigData"
MONGO_DB = "chestxray_db"
IMG_DIR = "/content/images-224/images-224/"

@st.cache_resource
def get_db():
    c = MongoClient(MONGO_URI, serverSelectionTimeoutMS=8000,
                    tls=True, tlsAllowInvalidCertificates=True)
    return c[MONGO_DB]

db = get_db()

# ── DATA LOADING FUNCTIONS ────────────────────────────────────────────────

@st.cache_data(ttl=15)
def load_predictions_by_date(target_date):
    start = datetime.combine(target_date, datetime.min.time(), tzinfo=timezone.utc)
    end = datetime.combine(target_date, datetime.max.time(), tzinfo=timezone.utc)

    docs = list(db["predictions"].find(
        {"timestamp": {"$gte": start, "$lte": end}},
        {"_id": 0}
    ).sort("timestamp", -1))

    return pd.DataFrame(docs) if docs else pd.DataFrame()

@st.cache_data(ttl=30)
def load_batch_stats_by_date(target_date):
    date_str = target_date.strftime("%Y-%m-%d")
    return db["batch_stats"].find_one(
        {"date": date_str},
        {"_id": 0, "computed_at": 0}
    )

@st.cache_data(ttl=60)
def get_available_dates():
    dates = list(db["batch_stats"].find({}, {"date": 1, "_id": 0}).sort("date", -1))
    return [d["date"] for d in dates]

# ── HEADER ────────────────────────────────────────────────────────────────

st.markdown(
    '<h1 style="text-align:center;color:#1565C0;font-size:1.8rem">🫁 X-Ray AI Dashboard</h1>',
    unsafe_allow_html=True
)
st.markdown(
    '<p style="text-align:center;color:#757575;font-size:.9rem">DenseNet121 · AUC=0.8380 · Lambda Architecture</p>',
    unsafe_allow_html=True
)

# ── SIDEBAR ───────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### ⚙️ Điều khiển")

    available_dates = get_available_dates()

    if available_dates:
        available_dates = [datetime.fromisoformat(d).date() for d in available_dates]
        default_date = available_dates[0] if available_dates else datetime.now().date()

        selected_date = st.date_input(
            "📅 Chọn ngày:",
            value=default_date,
            min_value=min(available_dates),
            max_value=max(available_dates)
        )
    else:
        selected_date = st.date_input("📅 Chọn ngày:", value=datetime.now().date())

    col1, col2 = st.columns(2)
    if col1.button("🔄 Refresh"):
        st.cache_data.clear()
        st.rerun()

    if col2.button("🏠 Home"):
        selected_date = datetime.now().date()
        st.rerun()

    st.divider()

    pri_filter = st.multiselect(
        "🎯 Priority Filter",
        [1, 2, 3],
        default=[1, 2, 3]
    )
    st.markdown("**🔴 Priority 3** = Khẩn cấp  \n**🟡 Priority 2** = Nặng  \n**🟢 Priority 1** = Thường")

    st.divider()
    st.markdown("**ℹ️ Info**\n- ⚡ Speed: predictions\n- 📦 Batch: batch_stats")

# ── LOAD DATA ────────────────────────────────────────────────────────────────

df = load_predictions_by_date(selected_date)
bs = load_batch_stats_by_date(selected_date)

st.markdown(
    f'<p style="text-align:center;color:#1565C0;font-size:1.2rem;font-weight:bold">📅 Ngày: {selected_date.strftime("%Y-%m-%d")}</p>',
    unsafe_allow_html=True
)

if df.empty:
    st.warning(f"⚠️ Không có dữ liệu ngày {selected_date}")
    st.info("💡 Chọn ngày khác từ sidebar")
    st.stop()

if pri_filter:
    df_filtered = df[df["priority"].isin(pri_filter)] if "priority" in df.columns else df
else:
    df_filtered = df

# ── KPI CARDS ──────────────────────────────────────────────────────────────

k1, k2, k3, k4, k5 = st.columns(5)

n_total = len(df_filtered)
n_emg = int(df_filtered["requires_emergency"].sum()) if "requires_emergency" in df_filtered.columns else 0
n_pri3 = int((df_filtered["priority"]==3).sum()) if "priority" in df_filtered.columns else 0
n_abn = int(df_filtered["is_abnormal"].sum()) if "is_abnormal" in df_filtered.columns else 0
ai_filter_pct = round(n_abn/max(1, n_total)*100, 1)

k1.metric("📊 Tổng ca", f"{n_total:,}")
k2.metric("🔴 Khẩn cấp", f"{n_emg:,}", delta_color="inverse")
k3.metric("⚠️ Priority 3", f"{n_pri3:,}", delta_color="inverse")
k4.metric("🏥 Bất thường", f"{n_abn:,}")
k5.metric("✨ AI lọc", f"{ai_filter_pct:.1f}%")

st.divider()

# ── TABS ───────────────────────────────────────────────────────────────────

T1, T2, T3, T4 = st.tabs([
    "🚨 Cảnh Báo",
    "📊 Tổng Quan",
    "🏥 Bệnh Lý",
    "👤 Bệnh Nhân"
])

with T1:
    st.markdown('### 🚨 Ca Khẩn Cấp')

    df3 = df_filtered[df_filtered["priority"]==3].head(10) if "priority" in df_filtered.columns else pd.DataFrame()

    if df3.empty:
        st.markdown(
            '<div class="success-box"><b>✅ Không có ca khẩn cấp!</b></div>',
            unsafe_allow_html=True
        )
    else:
        for idx, (_, r) in enumerate(df3.iterrows(), 1):
            diseases_str = ", ".join(r.get("diseases", []) or ["No Finding"])
            st.markdown(f'''
            <div class="alert-box">
              <b>🔴 #{idx} Patient {r.get("patient_id","")} — {diseases_str}</b><br>
              <small>Severity: <b>{r.get("severity","")}</b> | Age: {r.get("age","")} ({r.get("age_group","")})  </small>
            </div>
            ''', unsafe_allow_html=True)

with T2:
    st.markdown('### 📊 Thống Kê Tổng Quan')

    c1, c2 = st.columns(2)

    with c1:
        if "priority" in df_filtered.columns:
            pri_cnt = df_filtered["priority"].value_counts().reset_index()
            pri_cnt.columns = ["Priority", "Count"]
            fig = px.pie(pri_cnt, values="Count", names="Priority", title="Priority")
            st.plotly_chart(fig, use_container_width=True)

    with c2:
        if "age_group" in df_filtered.columns:
            age_cnt = df_filtered["age_group"].value_counts().reset_index()
            age_cnt.columns = ["Age Group", "Count"]
            fig = px.bar(age_cnt, x="Age Group", y="Count", title="Age Group")
            st.plotly_chart(fig, use_container_width=True)

    if bs:
        st.markdown(f"### 📦 Batch Stats")
        bc1, bc2, bc3 = st.columns(3)
        bc1.metric("Tổng ca", f"{bs.get('total_cases', 0):,}")
        bc2.metric("Khẩn cấp", f"{bs.get('emergency_cases', 0):,}")
        bc3.metric("Tỷ lệ", f"{bs.get('emergency_rate_pct', 0):.2f}%")

with T3:
    st.markdown('### Phân Tích Bệnh')

    if "diseases" in df_filtered.columns:
        all_d = []
        for d_list in df_filtered["diseases"]:
            if isinstance(d_list, list):
                all_d.extend(d_list)

        if all_d:
            cnt = pd.Series(all_d).value_counts().reset_index()
            cnt.columns = ["Disease", "Count"]
            fig = px.bar(cnt, y="Disease", x="Count", orientation="h", title="Disease Distribution")
            st.plotly_chart(fig, use_container_width=True)

with T4:
    st.markdown('### 👤 Thống Kê Bệnh Nhân')

    if "disease_count" in df_filtered.columns:
        dc_cnt = df_filtered["disease_count"].value_counts().sort_index().reset_index()
        dc_cnt.columns = ["Số bệnh", "Số ca"]
        fig = px.bar(dc_cnt, x="Số bệnh", y="Số ca", title="Workload")
        st.plotly_chart(fig, use_container_width=True)

# ── ECONOMIC IMPACT ────────────────────────────────────────────────────────

st.divider()
st.markdown('### 💰 Tác Động Kinh Tế')

n_all = len(df_filtered)
n_abn_all = int(df_filtered["is_abnormal"].sum()) if "is_abnormal" in df_filtered.columns else 0
ai_clear = n_all - n_abn_all
saved_vnd = ai_clear * 150_000

e1, e2, e3, e4 = st.columns(4)
e1.metric("Tổng ca", f"{n_all:,}")
e2.metric("AI lọc", f"{ai_clear:,}")
e3.metric("Cần bác sĩ", f"{n_abn_all:,}")
e4.metric("Tiết kiệm", f"{saved_vnd/1e6:.1f}M VNĐ")

st.caption("💡 150k/ca đọc thủ công")

# ── FOOTER ─────────────────────────────────────────────────────────────────

st.divider()
st.markdown(
    '<p style="text-align:center;color:#9E9E9E;font-size:.8rem">λ Lambda Architecture · DenseNet121 · MongoDB</p>',
    unsafe_allow_html=True
)
