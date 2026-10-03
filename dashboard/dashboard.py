import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent

# ======================================================================
# PAGE CONFIG & GLOBAL STYLE
# ======================================================================
st.set_page_config(
    page_title="Olist E-Commerce Dashboard",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Consistent color palette used across every chart in this dashboard
PALETTE = {
    "primary": "#4C72B0",
    "secondary": "#DD8452",
    "good": "#55A868",
    "warn": "#E8A33D",
    "bad": "#C44E52",
    "neutral": "#8C8C8C",
}
DELAY_COLORS = {
    "Lebih cepat >7 hari": "#2b8cbe",
    "Tepat waktu": "#55A868",
    "Terlambat 1-7 hari": "#E8A33D",
    "Terlambat >7 hari": "#C44E52",
}
TIER_COLORS = {"Small Seller": "#fcbba1", "Medium Seller": "#fb6a4a", "Large Seller": "#a50f15"}
SEGMENT_PALETTE = "viridis"

sns.set_theme(style="whitegrid", rc={"axes.edgecolor": "#D0D0D0"})
plt.rcParams.update({
    "font.size": 11,
    "axes.titleweight": "bold",
    "axes.titlesize": 13,
    "axes.labelsize": 11,
})

CUSTOM_CSS = """
<style>
    .main > div {padding-top: 1.2rem;}
    .block-container {padding-top: 1.5rem;}

    .dashboard-header {
        padding: 1.4rem 1.8rem;
        border-radius: 14px;
        background: linear-gradient(135deg, #4C72B0 0%, #6a8fc7 100%);
        color: white;
        margin-bottom: 1.2rem;
    }
    .dashboard-header h1 {
        margin: 0;
        font-size: 1.8rem;
        font-weight: 700;
        color: white;
    }
    .dashboard-header p {
        margin: 0.3rem 0 0 0;
        opacity: 0.9;
        font-size: 0.95rem;
    }

    div[data-testid="stMetric"] {
        background: #FFFFFF;
        border: 1px solid #ECECEC;
        border-left: 5px solid #4C72B0;
        border-radius: 10px;
        padding: 0.9rem 1rem 0.6rem 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    div[data-testid="stMetricLabel"] {font-weight: 600; color: #555;}

    .section-note {
        color: #6B6B6B;
        font-size: 0.88rem;
        margin-top: -0.3rem;
        margin-bottom: 0.8rem;
    }
    .insight-box {
        background: #F7F9FC;
        border-left: 4px solid #4C72B0;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        margin-top: 0.6rem;
        font-size: 0.92rem;
    }
    .tier-badge {
        display: inline-block;
        padding: 0.15rem 0.6rem;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 700;
        color: white;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ======================================================================
# DATA LOADING
# ======================================================================
@st.cache_data
def load_data():
    df = pd.read_csv(BASE_DIR / "main_data.csv")
    for col in ["order_purchase_timestamp", "order_delivered_customer_date", "order_estimated_delivery_date"]:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    df["order_month"] = df["order_purchase_timestamp"].dt.to_period("M").astype(str)
    return df


@st.cache_data
def compute_rfm(df):
    orders_customer = df.drop_duplicates(subset="order_id")[
        ["order_id", "customer_unique_id", "order_purchase_timestamp", "payment_value"]
    ]
    snapshot_date = orders_customer["order_purchase_timestamp"].max() + pd.Timedelta(days=1)
    rfm = (orders_customer.groupby("customer_unique_id")
           .agg(recency=("order_purchase_timestamp", lambda x: (snapshot_date - x.max()).days),
                frequency=("order_id", "nunique"),
                monetary=("payment_value", "sum"))
           .reset_index())
    rfm["r_score"] = pd.qcut(rfm["recency"], q=4, labels=[4, 3, 2, 1]).astype(int)
    rfm["m_score"] = pd.qcut(rfm["monetary"].rank(method="first"), q=4, labels=[1, 2, 3, 4]).astype(int)

    def freq_score(f):
        if f == 1:
            return 1
        elif f == 2:
            return 2
        return 3

    rfm["f_score"] = rfm["frequency"].apply(freq_score)

    def segment_customer(row):
        if row["f_score"] >= 2 and row["r_score"] >= 3:
            return "Loyal Customers"
        elif row["f_score"] >= 2 and row["r_score"] < 3:
            return "At Risk"
        elif row["f_score"] == 1 and row["r_score"] >= 3 and row["m_score"] >= 3:
            return "Potential Loyalist"
        elif row["f_score"] == 1 and row["r_score"] >= 3:
            return "New/One-time Customers"
        return "Hibernating/Lost Customers"

    rfm["segment"] = rfm.apply(segment_customer, axis=1)
    return rfm


@st.cache_data
def compute_seller_tiers(df):
    seller_perf = (df.groupby("seller_id")
                   .agg(total_revenue=("price", "sum"), total_orders=("order_id", "nunique"))
                   .reset_index())

    def tier(n):
        if n < 10:
            return "Small Seller"
        elif n < 50:
            return "Medium Seller"
        return "Large Seller"

    seller_perf["tier"] = seller_perf["total_orders"].apply(tier)
    return seller_perf


def delay_bucket(days):
    if pd.isna(days):
        return np.nan
    if days <= -7:
        return "Lebih cepat >7 hari"
    elif days <= 0:
        return "Tepat waktu"
    elif days <= 7:
        return "Terlambat 1-7 hari"
    return "Terlambat >7 hari"


df = load_data()

# ======================================================================
# HEADER
# ======================================================================
st.markdown("""
<div class="dashboard-header">
    <h1>🛍️ Olist E-Commerce Analytics Dashboard</h1>
    <p>Dashboard interaktif hasil analisis data E-Commerce Public Dataset — revenue, pengiriman, segmentasi pelanggan (RFM), dan segmentasi seller.</p>
</div>
""", unsafe_allow_html=True)

# ======================================================================
# SIDEBAR FILTERS
# ======================================================================
st.sidebar.markdown("### 🔎 Filter Data")
min_date = df["order_purchase_timestamp"].min().date()
max_date = df["order_purchase_timestamp"].max().date()
date_range = st.sidebar.date_input("Rentang tanggal order", value=(min_date, max_date),
                                    min_value=min_date, max_value=max_date)

all_states = sorted(df["customer_state"].dropna().unique().tolist())
selected_states = st.sidebar.multiselect("Provinsi (state) pelanggan", options=all_states, default=[])

all_categories = sorted(df["product_category_name_english"].dropna().unique().tolist())
selected_categories = st.sidebar.multiselect("Kategori produk", options=all_categories, default=[])

filtered = df.copy()
if len(date_range) == 2:
    start_date, end_date = date_range
    filtered = filtered[(filtered["order_purchase_timestamp"].dt.date >= start_date) &
                         (filtered["order_purchase_timestamp"].dt.date <= end_date)]
if selected_states:
    filtered = filtered[filtered["customer_state"].isin(selected_states)]
if selected_categories:
    filtered = filtered[filtered["product_category_name_english"].isin(selected_categories)]

st.sidebar.markdown(f"**📊 Baris data setelah filter:** {len(filtered):,}")
st.sidebar.divider()
st.sidebar.caption("Dataset: E-Commerce Public Dataset (Olist Brazilian E-Commerce) · Dibuat untuk submission Proyek Analisis Data - Dicoding")

# ======================================================================
# KPI ROW
# ======================================================================
col1, col2, col3, col4 = st.columns(4)
col1.metric("💰 Total Revenue", f"R$ {filtered['price'].sum():,.0f}")
col2.metric("📦 Total Order", f"{filtered['order_id'].nunique():,}")
avg_review = filtered["review_score"].mean()
col3.metric("⭐ Rata-rata Review", f"{avg_review:.2f}" if pd.notna(avg_review) else "N/A")
col4.metric("👥 Pelanggan Unik", f"{filtered['customer_unique_id'].nunique():,}")

st.divider()

tab1, tab2, tab3, tab4 = st.tabs([
    "📈  Revenue & Kategori", "🚚  Pengiriman & Review", "👥  RFM Segmentation", "🏪  Seller Clustering"
])

# ======================================================================
# TAB 1 — Revenue per category
# ======================================================================
with tab1:
    st.subheader("Kategori produk apa yang memberikan kontribusi revenue tertinggi & terendah?")
    st.markdown('<p class="section-note">Pertanyaan Bisnis 1 — revenue per kategori & tren bulanannya</p>', unsafe_allow_html=True)

    category_summary = (filtered.groupby("product_category_name_english")
                         .agg(revenue=("price", "sum"), total_orders=("order_id", "nunique"))
                         .sort_values("revenue", ascending=False))

    n_show = st.slider("Jumlah kategori ditampilkan", 5, 20, 10, key="n_cat")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**Top {n_show} kategori (revenue tertinggi)**")
        top_n = category_summary.head(n_show).sort_values("revenue")
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.barh(top_n.index, top_n["revenue"], color=sns.color_palette("Blues_r", len(top_n)))
        ax.set_xlabel("Revenue (BRL)")
        sns.despine(left=True, bottom=True)
        st.pyplot(fig)
    with c2:
        st.markdown(f"**Bottom {n_show} kategori (revenue terendah)**")
        bottom_n = category_summary.tail(n_show).sort_values("revenue")
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.barh(bottom_n.index, bottom_n["revenue"], color=sns.color_palette("Reds", len(bottom_n)))
        ax.set_xlabel("Revenue (BRL)")
        sns.despine(left=True, bottom=True)
        st.pyplot(fig)

    st.markdown("**Tren revenue bulanan — 5 kategori teratas**")
    top5 = category_summary.head(5).index.tolist()
    monthly = (filtered[filtered["product_category_name_english"].isin(top5)]
               .groupby(["order_month", "product_category_name_english"])["price"].sum().reset_index())
    if not monthly.empty:
        fig, ax = plt.subplots(figsize=(11, 5))
        sns.lineplot(data=monthly, x="order_month", y="price", hue="product_category_name_english", marker="o", ax=ax)
        ax.set_xlabel("Bulan"); ax.set_ylabel("Revenue (BRL)")
        plt.xticks(rotation=45)
        ax.legend(title="Kategori", bbox_to_anchor=(1.02, 1), loc="upper left")
        sns.despine()
        st.pyplot(fig)

        leader_per_month = monthly.loc[monthly.groupby("order_month")["price"].idxmax()]
        leader_counts = leader_per_month["product_category_name_english"].value_counts()
        st.markdown(
            f'<div class="insight-box">📌 <b>Insight:</b> Kepemimpinan revenue bulanan berpindah-pindah antar kategori — '
            f'<b>{leader_counts.index[0]}</b> paling sering menjadi #1 ({leader_counts.iloc[0]} dari {len(leader_per_month)} bulan pada data terfilter).</div>',
            unsafe_allow_html=True,
        )
    else:
        st.info("Tidak ada data untuk filter yang dipilih.")

# ======================================================================
# TAB 2 — Delivery delay vs review
# ======================================================================
with tab2:
    st.subheader("Apakah keterlambatan pengiriman berpengaruh terhadap review score?")
    st.markdown('<p class="section-note">Pertanyaan Bisnis 2 — hubungan delay pengiriman vs kepuasan pelanggan</p>', unsafe_allow_html=True)

    delivered = filtered[(filtered["order_status"] == "delivered") &
                          (filtered["order_delivered_customer_date"].notna())].copy()
    delivered["delivery_delay_days"] = (delivered["order_delivered_customer_date"] -
                                         delivered["order_estimated_delivery_date"]).dt.days
    delivered["delay_category"] = delivered["delivery_delay_days"].apply(delay_bucket)

    order_cats = ["Lebih cepat >7 hari", "Tepat waktu", "Terlambat 1-7 hari", "Terlambat >7 hari"]
    review_by_delay = (delivered.dropna(subset=["review_score"])
                        .groupby("delay_category")["review_score"]
                        .agg(avg_review="mean", jumlah_order="count")
                        .reindex(order_cats))

    if review_by_delay["jumlah_order"].sum() > 0:
        c1, c2 = st.columns([3, 2])
        with c1:
            fig, ax = plt.subplots(figsize=(8, 5))
            bars = ax.bar(review_by_delay.index, review_by_delay["avg_review"],
                           color=[DELAY_COLORS[i] for i in review_by_delay.index])
            ax.set_ylim(0, 5)
            ax.set_ylabel("Rata-rata Review Score")
            for bar, n in zip(bars, review_by_delay["jumlah_order"]):
                if not pd.isna(n):
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.08,
                            f"n={int(n):,}", ha="center", fontsize=9)
            plt.xticks(rotation=10)
            sns.despine()
            st.pyplot(fig)
        with c2:
            st.markdown("**Ringkasan per kategori**")
            st.dataframe(review_by_delay.style.format({"avg_review": "{:.2f}", "jumlah_order": "{:,.0f}"})
                         .background_gradient(cmap="RdYlGn", subset=["avg_review"]))

        drop = review_by_delay["avg_review"].iloc[1] - review_by_delay["avg_review"].iloc[2]
        st.markdown(
            f'<div class="insight-box">📌 <b>Insight:</b> Review score turun paling tajam (-{drop:.2f} poin) begitu order melewati '
            f'tanggal estimasi pengiriman — dari rata-rata <b>{review_by_delay["avg_review"].iloc[1]:.2f}</b> (tepat waktu) '
            f'menjadi <b>{review_by_delay["avg_review"].iloc[2]:.2f}</b> (terlambat 1-7 hari).</div>',
            unsafe_allow_html=True,
        )
    else:
        st.info("Tidak ada data pengiriman yang cukup untuk filter yang dipilih.")

# ======================================================================
# TAB 3 — RFM Segmentation
# ======================================================================
with tab3:
    st.subheader("Segmentasi Pelanggan dengan RFM Analysis")
    st.markdown(
        '<p class="section-note">Teknik binning manual (bukan machine learning) pada Recency, Frequency, Monetary — '
        'dihitung dari keseluruhan dataset agar histori pelanggan tetap utuh, tidak mengikuti filter sidebar.</p>',
        unsafe_allow_html=True,
    )
    rfm = compute_rfm(df)
    segment_summary = (rfm.groupby("segment")
                        .agg(jumlah_pelanggan=("customer_unique_id", "count"),
                             rata_rata_recency=("recency", "mean"),
                             rata_rata_frequency=("frequency", "mean"),
                             rata_rata_monetary=("monetary", "mean"))
                        .sort_values("jumlah_pelanggan", ascending=False))
    segment_summary["pct"] = segment_summary["jumlah_pelanggan"] / segment_summary["jumlah_pelanggan"].sum() * 100

    c1, c2 = st.columns([2, 3])
    with c1:
        st.dataframe(segment_summary.style.format({
            "jumlah_pelanggan": "{:,.0f}", "rata_rata_recency": "{:.0f}",
            "rata_rata_frequency": "{:.2f}", "rata_rata_monetary": "R$ {:.2f}", "pct": "{:.1f}%"
        }))
    with c2:
        fig, ax = plt.subplots(figsize=(7, 5))
        seg_order = segment_summary.index
        sns.barplot(x=segment_summary["jumlah_pelanggan"], y=seg_order, hue=seg_order,
                    palette=SEGMENT_PALETTE, legend=False, ax=ax)
        ax.set_xlabel("Jumlah Pelanggan"); ax.set_ylabel("")
        for i, v in enumerate(segment_summary["jumlah_pelanggan"]):
            ax.text(v, i, f"  {v:,} ({segment_summary['pct'].iloc[i]:.1f}%)", va="center", fontsize=9)
        sns.despine()
        st.pyplot(fig)

    onetime_pct = segment_summary.loc[segment_summary.index.isin(
        ["Hibernating/Lost Customers", "New/One-time Customers", "Potential Loyalist"]), "pct"].sum()
    st.markdown(
        f'<div class="insight-box">📌 <b>Insight:</b> Sekitar <b>{onetime_pct:.0f}%</b> pelanggan hanya bertransaksi satu kali. '
        f'Namun segmen <b>Loyal Customers</b> & <b>At Risk</b> (yang pernah repeat order) justru memiliki rata-rata monetary '
        f'tertinggi — kandidat utama program retensi.</div>',
        unsafe_allow_html=True,
    )

# ======================================================================
# TAB 4 — Seller Clustering (Manual Grouping)
# ======================================================================
with tab4:
    st.subheader("Segmentasi Performa Seller (Clustering — Manual Grouping)")
    st.markdown(
        '<p class="section-note">Pengelompokan seller ke 3 tier berdasarkan jumlah order (aturan bisnis / manual grouping, '
        'bukan algoritma machine learning): Small (&lt;10 order), Medium (10-49), Large (&ge;50).</p>',
        unsafe_allow_html=True,
    )

    seller_perf = compute_seller_tiers(df)
    tier_summary = (seller_perf.groupby("tier")
                    .agg(jumlah_seller=("seller_id", "count"),
                         total_revenue=("total_revenue", "sum"),
                         total_orders=("total_orders", "sum"))
                    .reindex(["Small Seller", "Medium Seller", "Large Seller"]))
    tier_summary["pct_seller"] = tier_summary["jumlah_seller"] / tier_summary["jumlah_seller"].sum() * 100
    tier_summary["pct_revenue"] = tier_summary["total_revenue"] / tier_summary["total_revenue"].sum() * 100

    badge_html = " ".join(
        f'<span class="tier-badge" style="background:{TIER_COLORS[t]};">{t}: {tier_summary.loc[t,"pct_seller"]:.0f}% seller '
        f'/ {tier_summary.loc[t,"pct_revenue"]:.0f}% revenue</span>'
        for t in tier_summary.index
    )
    st.markdown(badge_html, unsafe_allow_html=True)
    st.write("")

    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(6, 5))
        colors = [TIER_COLORS[t] for t in tier_summary.index]
        ax.bar(tier_summary.index, tier_summary["jumlah_seller"], color=colors)
        ax.set_title("Jumlah Seller per Tier")
        ax.set_ylabel("Jumlah Seller")
        for i, v in enumerate(tier_summary["jumlah_seller"]):
            ax.text(i, v, f"{v:,}\n({tier_summary['pct_seller'].iloc[i]:.1f}%)", ha="center", va="bottom")
        sns.despine()
        st.pyplot(fig)
    with c2:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.bar(tier_summary.index, tier_summary["total_revenue"], color=colors)
        ax.set_title("Total Revenue per Tier")
        ax.set_ylabel("Revenue (BRL)")
        for i, v in enumerate(tier_summary["total_revenue"]):
            ax.text(i, v, f"R$ {v:,.0f}\n({tier_summary['pct_revenue'].iloc[i]:.1f}%)", ha="center", va="bottom")
        sns.despine()
        st.pyplot(fig)

    st.markdown("**Scatter: Revenue vs Jumlah Order per Seller**")
    fig, ax = plt.subplots(figsize=(10, 5))
    for t in ["Small Seller", "Medium Seller", "Large Seller"]:
        sub = seller_perf[seller_perf["tier"] == t]
        ax.scatter(sub["total_orders"], sub["total_revenue"], label=t, color=TIER_COLORS[t], alpha=0.6, s=25)
    ax.set_xlabel("Total Order"); ax.set_ylabel("Total Revenue (BRL)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.legend(title="Tier")
    sns.despine()
    st.pyplot(fig)

    st.markdown(
        f'<div class="insight-box">📌 <b>Insight:</b> <b>{tier_summary.loc["Small Seller","pct_seller"]:.0f}%</b> seller tergolong '
        f'Small Seller namun hanya menyumbang <b>{tier_summary.loc["Small Seller","pct_revenue"]:.0f}%</b> revenue, sementara '
        f'<b>Large Seller</b> ({tier_summary.loc["Large Seller","pct_seller"]:.0f}% populasi) menopang '
        f'<b>{tier_summary.loc["Large Seller","pct_revenue"]:.0f}%</b> dari total revenue marketplace — pola konsentrasi 80/20.</div>',
        unsafe_allow_html=True,
    )

st.divider()
st.caption("Dibuat untuk submission Proyek Analisis Data - Dicoding | Dataset: E-Commerce Public Dataset (Olist)")
