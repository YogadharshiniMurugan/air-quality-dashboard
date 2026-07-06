"""
Air Quality Trend Dashboard — Streamlit App
=============================================
Author : Yogadharshini M
Skills : Python, Pandas, Matplotlib, Seaborn, Streamlit, Data Analytics
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="Air Quality Dashboard",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Space+Grotesk:wght@500;700&display=swap');

  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  .main { background: #0F1117; }

  .hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.6rem; font-weight: 700;
    background: linear-gradient(135deg, #6C63FF 0%, #00D4AA 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 0.25rem;
  }
  .hero-sub { color: #8B8FA8; font-size: 1rem; margin-bottom: 1.5rem; }

  .kpi-box {
    background: #1A1D2E; border: 1px solid #2A2D45;
    border-radius: 14px; padding: 1.2rem 1.4rem;
    text-align: center;
  }
  .kpi-label { font-size: 0.72rem; text-transform: uppercase;
    letter-spacing: 0.08em; color: #8B8FA8; margin-bottom: 6px; }
  .kpi-value { font-family: 'Space Grotesk', sans-serif;
    font-size: 2rem; font-weight: 700; color: #E8E9F3; }
  .kpi-sub { font-size: 0.75rem; color: #8B8FA8; margin-top: 4px; }

  .section-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.1rem; font-weight: 600;
    color: #E8E9F3; margin: 1.5rem 0 0.75rem;
    border-bottom: 1px solid #2A2D45; padding-bottom: 6px;
  }

  div[data-testid="stSelectbox"] label,
  div[data-testid="stMultiSelect"] label,
  div[data-testid="stSlider"] label { color: #8B8FA8 !important; font-size: 0.82rem; }

  .stMetric { background: #1A1D2E; border-radius: 12px; padding: 0.8rem; }

  footer { visibility: hidden; }
  #MainMenu { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Palette ───────────────────────────────────────────────────────────
BG       = "#0F1117"
SURFACE  = "#1A1D2E"
CARD     = "#222640"
TEXT     = "#E8E9F3"
MUTED    = "#8B8FA8"

CITY_COLORS = {
    "Chennai":  "#6C63FF",
    "Mumbai":   "#00D4AA",
    "Delhi":    "#FF6B6B",
    "Kolkata":  "#FFB347",
    "Bengaluru":"#4ECDC4",
}
CITIES    = list(CITY_COLORS.keys())
POLLUTANTS = ["PM2.5","PM10","NO2","SO2","CO","O3"]

AQI_BANDS = [
    (0,50,"#00E400","Good"),
    (51,100,"#9ACD32","Satisfactory"),
    (101,200,"#FF7E00","Moderate"),
    (201,300,"#FF0000","Poor"),
    (301,400,"#8F3F97","Very Poor"),
    (401,500,"#7E0023","Severe"),
]

def aqi_category(v):
    for lo,hi,c,l in AQI_BANDS:
        if lo <= v <= hi: return l, c
    return "Severe","#7E0023"

def get_season(m):
    if m in [12,1,2]:  return "Winter"
    if m in [3,4,5]:   return "Summer"
    if m in [6,7,8,9]: return "Monsoon"
    return "Post-monsoon"

# ── Data generation ───────────────────────────────────────────────────
@st.cache_data
def load_data():
    np.random.seed(42)
    records = []
    start = datetime(2022,1,1)
    profiles = {
        "Chennai":   {"base":85,  "wb":20, "md":35},
        "Mumbai":    {"base":110, "wb":40, "md":50},
        "Delhi":     {"base":180, "wb":120,"md":60},
        "Kolkata":   {"base":130, "wb":70, "md":45},
        "Bengaluru": {"base":70,  "wb":15, "md":25},
    }
    festivals = {(10,24):60,(10,25):80,(11,1):50,(1,26):30,(8,15):20}
    for d in range(730):
        date = start + timedelta(days=d)
        m, md = date.month, date.day
        for city, p in profiles.items():
            aqi = p["base"]
            if m in [11,12,1,2]: aqi += p["wb"]*(1+np.random.uniform(-0.2,0.3))
            if m in [6,7,8,9]:   aqi -= p["md"]*(1+np.random.uniform(-0.1,0.2))
            aqi += festivals.get((m,md),0) + np.random.normal(0,12)
            aqi = max(10, min(500, aqi))
            records.append({
                "date": date, "city": city,
                "month": m, "year": date.year,
                "month_name": date.strftime("%b"),
                "season": get_season(m),
                "aqi": round(aqi,1),
                "PM2.5": max(0,round(aqi*0.45+np.random.normal(0,5),1)),
                "PM10":  max(0,round(aqi*0.70+np.random.normal(0,8),1)),
                "NO2":   max(0,round(aqi*0.12+np.random.normal(0,3),1)),
                "SO2":   max(0,round(aqi*0.06+np.random.normal(0,2),1)),
                "CO":    max(0,round(aqi*0.03+np.random.normal(0,1),1)),
                "O3":    max(0,round(aqi*0.09+np.random.normal(0,3),1)),
            })
    df = pd.DataFrame(records)
    Q1,Q3 = df["aqi"].quantile(0.25), df["aqi"].quantile(0.75)
    df["is_spike"] = df["aqi"] > Q3 + 1.5*(Q3-Q1)
    return df

# ── Chart style ───────────────────────────────────────────────────────
def chart_style():
    plt.rcParams.update({
        "figure.facecolor": BG, "axes.facecolor": SURFACE,
        "axes.edgecolor": CARD, "axes.labelcolor": MUTED,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "text.color": TEXT, "grid.color": CARD,
        "grid.linewidth": 0.5, "font.family": "DejaVu Sans",
        "axes.titlesize": 12, "axes.titleweight": "bold",
        "axes.titlecolor": TEXT,
    })

# ── Sidebar ───────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌫️ AQI Dashboard")
    st.markdown("<p style='color:#8B8FA8;font-size:0.82rem'>India · 2022–2024 · CPCB format</p>", unsafe_allow_html=True)
    st.divider()

    selected_cities = st.multiselect(
        "Select cities", CITIES, default=CITIES,
    )
    year_filter = st.selectbox("Year", ["All","2022","2023"])

# ── Load & filter ─────────────────────────────────────────────────────
df_all = load_data()
df = df_all[df_all["city"].isin(selected_cities)] if selected_cities else df_all
if year_filter != "All":
    df = df[df["year"] == int(year_filter)]

if df.empty:
    st.warning("No data — please select at least one city.")
    st.stop()

# ── Hero ──────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">Air Quality Trend Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Two-year AQI analysis across major Indian cities — seasonal patterns, spikes & pollutant breakdown</div>', unsafe_allow_html=True)

# ── KPI row ───────────────────────────────────────────────────────────
summary = df.groupby("city")["aqi"].mean()
worst_city = summary.idxmax() if not summary.empty else "—"
best_city  = summary.idxmin() if not summary.empty else "—"
spikes     = int(df["is_spike"].sum())
avg_aqi    = round(df["aqi"].mean(),1)
cat_label, cat_color = aqi_category(avg_aqi)

k1,k2,k3,k4,k5 = st.columns(5)
for col, label, value, sub in [
    (k1,"Overall avg AQI", avg_aqi, cat_label),
    (k2,"Most polluted", worst_city, "Highest avg AQI"),
    (k3,"Cleanest city", best_city, "Lowest avg AQI"),
    (k4,"Spike days", spikes, "AQI > Q3+1.5×IQR"),
    (k5,"Records", f"{len(df):,}", "Sensor readings"),
]:
    col.markdown(f"""
    <div class="kpi-box">
      <div class="kpi-label">{label}</div>
      <div class="kpi-value">{value}</div>
      <div class="kpi-sub">{sub}</div>
    </div>""", unsafe_allow_html=True)

# ── Tab layout ────────────────────────────────────────────────────────
chart_style()
tab1, tab2, tab3, tab4 = st.tabs(["📈 Trends","🗓 Heatmap","🌦 Seasonal","🧪 Pollutants"])

# ── Tab 1: Time series ────────────────────────────────────────────────
with tab1:
    st.markdown('<div class="section-title">Monthly AQI trend</div>', unsafe_allow_html=True)
    monthly = (df.groupby(["city","year","month"])["aqi"]
               .mean().reset_index().sort_values(["year","month"]))
    fig, ax = plt.subplots(figsize=(12,4))
    for city in selected_cities:
        sub = monthly[monthly["city"]==city]
        ax.plot(range(len(sub)), sub["aqi"],
                color=CITY_COLORS.get(city,"#888"), linewidth=2,
                label=city, alpha=0.9)
    ax.axhspan(0,100,alpha=0.04,color="#00E400")
    ax.axhspan(100,200,alpha=0.04,color="#FF7E00")
    ax.axhspan(200,500,alpha=0.04,color="#FF0000")
    ax.set_xlabel("Month index"); ax.set_ylabel("AQI")
    ax.legend(loc="upper right",framealpha=0.2,labelcolor=TEXT,fontsize=9)
    ax.grid(True,axis="y",alpha=0.4)
    fig.tight_layout()
    st.pyplot(fig); plt.close()

    # City summary table
    st.markdown('<div class="section-title">City summary</div>', unsafe_allow_html=True)
    tbl = (df.groupby("city")["aqi"]
           .agg(avg="mean",max="max",min="min",
                spikes=lambda x:(x>200).sum())
           .round(1).reset_index())
    tbl["category"] = tbl["avg"].apply(lambda v: aqi_category(v)[0])
    st.dataframe(
        tbl.rename(columns={"city":"City","avg":"Avg AQI",
                             "max":"Max","min":"Min",
                             "spikes":"Poor days","category":"Status"}),
        use_container_width=True, hide_index=True,
    )

# ── Tab 2: Heatmap ────────────────────────────────────────────────────
with tab2:
    st.markdown('<div class="section-title">Avg AQI — city × month</div>', unsafe_allow_html=True)
    month_order = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    pivot = df.pivot_table(values="aqi",index="city",
                           columns="month_name",aggfunc="mean").round(1)
    pivot = pivot[[c for c in month_order if c in pivot.columns]]
    fig, ax = plt.subplots(figsize=(13,4))
    cmap = sns.diverging_palette(120,10,s=80,l=40,as_cmap=True)
    sns.heatmap(pivot,ax=ax,cmap=cmap,annot=True,fmt=".0f",
                linewidths=0.4,linecolor=BG,
                annot_kws={"size":9,"color":TEXT},
                cbar_kws={"shrink":0.7})
    ax.tick_params(colors=TEXT)
    ax.collections[0].colorbar.ax.tick_params(colors=TEXT)
    fig.tight_layout()
    st.pyplot(fig); plt.close()

    # Spike chart
    st.markdown('<div class="section-title">Pollution spike days per city</div>', unsafe_allow_html=True)
    spike_df = (df[df["is_spike"]].groupby("city").size()
                .reset_index(name="spikes"))
    fig, ax = plt.subplots(figsize=(8,3.5))
    colors = [CITY_COLORS.get(c,"#888") for c in spike_df["city"]]
    bars = ax.bar(spike_df["city"],spike_df["spikes"],color=colors,width=0.5,alpha=0.9)
    for b,v in zip(bars,spike_df["spikes"]):
        ax.text(b.get_x()+b.get_width()/2,b.get_height()+0.3,str(v),
                ha="center",fontsize=10,color=TEXT)
    ax.set_ylabel("Spike days"); ax.grid(True,axis="y",alpha=0.4)
    fig.tight_layout()
    st.pyplot(fig); plt.close()

# ── Tab 3: Seasonal ───────────────────────────────────────────────────
with tab3:
    st.markdown('<div class="section-title">Seasonal AQI breakdown</div>', unsafe_allow_html=True)
    seasons = ["Winter","Summer","Monsoon","Post-monsoon"]
    seasonal = df.groupby(["city","season"])["aqi"].mean().round(1).reset_index()
    fig, axes = plt.subplots(1,4,figsize=(14,4),sharey=True)
    for ax, season in zip(axes,seasons):
        sub = seasonal[seasonal["season"]==season].sort_values("aqi")
        colors = [CITY_COLORS.get(c,"#888") for c in sub["city"]]
        bars = ax.barh(sub["city"],sub["aqi"],color=colors,height=0.5,alpha=0.9)
        for b,v in zip(bars,sub["aqi"]):
            ax.text(b.get_width()+1,b.get_y()+b.get_height()/2,
                    f"{v:.0f}",va="center",fontsize=9,color=TEXT)
        ax.set_title(season,fontsize=11); ax.set_xlabel("AQI")
        ax.grid(True,axis="x",alpha=0.4)
    fig.tight_layout()
    st.pyplot(fig); plt.close()

    # Seasonal insight
    worst_s = seasonal.loc[seasonal["aqi"].idxmax()]
    best_s  = seasonal.loc[seasonal["aqi"].idxmin()]
    c1,c2 = st.columns(2)
    c1.info(f"**Worst:** {worst_s['season']} in {worst_s['city']} — avg AQI {worst_s['aqi']:.0f}")
    c2.success(f"**Best:** {best_s['season']} in {best_s['city']} — avg AQI {best_s['aqi']:.0f}")

# ── Tab 4: Pollutants ─────────────────────────────────────────────────
with tab4:
    st.markdown('<div class="section-title">Average pollutant levels (µg/m³)</div>', unsafe_allow_html=True)
    poll_df = df.groupby("city")[POLLUTANTS].mean().round(1).reset_index()

    selected_poll = st.selectbox("Select pollutant", POLLUTANTS)
    fig, ax = plt.subplots(figsize=(9,4))
    sub = poll_df[["city",selected_poll]].sort_values(selected_poll,ascending=False)
    colors = [CITY_COLORS.get(c,"#888") for c in sub["city"]]
    bars = ax.bar(sub["city"],sub[selected_poll],color=colors,width=0.5,alpha=0.9)
    for b,v in zip(bars,sub[selected_poll]):
        ax.text(b.get_x()+b.get_width()/2,b.get_height()+0.3,f"{v:.1f}",
                ha="center",fontsize=10,color=TEXT)
    ax.set_ylabel("µg/m³"); ax.set_title(f"{selected_poll} levels by city")
    ax.grid(True,axis="y",alpha=0.4)
    fig.tight_layout()
    st.pyplot(fig); plt.close()

    # All pollutants grid
    st.markdown('<div class="section-title">All pollutants comparison</div>', unsafe_allow_html=True)
    fig, axes = plt.subplots(2,3,figsize=(14,7))
    axes = axes.flatten()
    for i, poll in enumerate(POLLUTANTS):
        ax = axes[i]
        sub = poll_df[["city",poll]].sort_values(poll,ascending=False)
        colors = [CITY_COLORS.get(c,"#888") for c in sub["city"]]
        ax.bar(sub["city"],sub[poll],color=colors,alpha=0.85,width=0.55)
        ax.set_title(poll,fontsize=11)
        ax.set_ylabel("µg/m³")
        ax.tick_params(axis="x",labelsize=8,rotation=15)
        ax.grid(True,axis="y",alpha=0.4)
    fig.tight_layout()
    st.pyplot(fig); plt.close()

# ── Footer ────────────────────────────────────────────────────────────
st.divider()
st.markdown("""
<p style='text-align:center;color:#8B8FA8;font-size:0.78rem'>
Air Quality Trend Dashboard · Python · Pandas · Matplotlib · Seaborn · Streamlit ·
Data simulated from CPCB sensor format · Built by Yogadharshini M
</p>""", unsafe_allow_html=True)
