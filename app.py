import pandas as pd
import streamlit as st
import pathlib
import glob
import numpy as np
import plotly.express as px

st.set_page_config(page_title="Dashboard | بستن تقویم", layout="wide")

# --- Persian font (works well on Streamlit Cloud) ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Vazirmatn:wght@300;400;600;700&display=swap');

html, body, [class*="css"], .stApp {
  font-family: 'Vazirmatn', sans-serif !important;
}
</style>
""", unsafe_allow_html=True)

PERSIAN_FONT = "Vazirmatn"

# Prefer any xlsx inside data/
_candidates = sorted(glob.glob("data/*.xlsx"))
DEFAULT_XLSX_PATH = _candidates[0] if _candidates else "data/analysis.xlsx"

OUTSIDE_REASON = "رزرو خارج از جاباما (کانال‌های دیگر)"

@st.cache_data
def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    # Reason column (survey question)
    reason_cols = [c for c in df.columns if "دلیل" in c]
    if not reason_cols:
        raise ValueError("ستونِ دلیل در دیتاست پیدا نشد.")
    reason_col = reason_cols[0]

    # Selling method (optional)
    sell_cols = [c for c in df.columns if "روش فروش" in c]
    sell_col = sell_cols[0] if sell_cols else None

    def normalize_reason(x):
        if pd.isna(x):
            return "نامشخص"
        s = str(x).strip()
        if s.startswith("سایر"):
            return "سایر (متن آزاد)"
        return s

    out = df.copy()
    out["Reason_norm"] = out[reason_col].apply(normalize_reason)

    if sell_col:
        out["SellMethod"] = out[sell_col].fillna("نامشخص").astype(str).str.strip()
        out.loc[out["SellMethod"].eq("nan"), "SellMethod"] = "نامشخص"
    else:
        out["SellMethod"] = "نامشخص"

    # Numeric columns (if exist)
    for col in ["Orate", "TO", "TFO", "RateScore", "RateCount", "Capacity", "AIV", "NMV"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    return out

@st.cache_data
def load_data_from_path(path: str) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name="Merged_data")
    return _normalize(df)

@st.cache_data
def load_data_from_upload(uploaded_file) -> pd.DataFrame:
    df = pd.read_excel(uploaded_file, sheet_name="Merged_data")
    return _normalize(df)

def _apply_plotly_font(fig):
    fig.update_layout(
        font=dict(family=PERSIAN_FONT, size=14),
        legend=dict(font=dict(family=PERSIAN_FONT, size=12)),
        title=dict(font=dict(family=PERSIAN_FONT, size=16)),
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig

def pie_chart(count_df: pd.DataFrame, names_col: str, values_col: str, title: str):
    fig = px.pie(count_df, names=names_col, values=values_col, title=title, hole=0.35)
    fig.update_traces(textposition="inside", textinfo="percent+label")
    fig = _apply_plotly_font(fig)
    st.plotly_chart(fig, use_container_width=True)

def bar_chart(count_df: pd.DataFrame, x: str, y: str, title: str):
    fig = px.bar(count_df, x=x, y=y, title=title)
    fig = _apply_plotly_font(fig)
    st.plotly_chart(fig, use_container_width=True)

def top_n_with_other(series: pd.Series, top_n: int = 10, other_label: str = "سایر"):
    vc = series.value_counts(dropna=False)
    if len(vc) <= top_n:
        return vc.reset_index().rename(columns={"index":"name", series.name or "value":"count"})
    top = vc.head(top_n)
    other = vc.iloc[top_n:].sum()
    out = pd.concat([top, pd.Series({other_label: other})])
    return out.reset_index().rename(columns={"index":"name", 0:"count"})

st.title("داشبورد تحلیل بستن تقویم (تعطیلات ۲۱ تا ۲۴ بهمن)")

with st.sidebar:
    st.header("منبع داده")
    st.caption("فایل‌های داخل پوشه data/ که اپ می‌بیند:")
    st.write(sorted(glob.glob("data/*.xlsx")))

    mode = st.radio(
        "داده‌ها از کجا خوانده شود؟",
        ["از فایل داخل ریپو (پیشنهادی)", "آپلود دستی فایل اکسل"],
        index=0
    )

    if mode == "از فایل داخل ریپو (پیشنهادی)":
        xlsx_path = st.text_input("مسیر فایل اکسل", value=DEFAULT_XLSX_PATH)
        load_btn = st.button("بارگذاری داده")
    else:
        uploaded = st.file_uploader("آپلود فایل analysis.xlsx", type=["xlsx"])

df = None
if mode == "از فایل داخل ریپو (پیشنهادی)":
    if load_btn:
        try:
            df = load_data_from_path(xlsx_path)
        except Exception as e:
            st.error(f"خطا در خواندن فایل: {e}")
    else:
        try:
            if pathlib.Path(xlsx_path).exists():
                df = load_data_from_path(xlsx_path)
        except Exception:
            pass
else:
    if uploaded is not None:
        try:
            df = load_data_from_upload(uploaded)
        except Exception as e:
            st.error(f"خطا در خواندن فایل: {e}")

if df is None:
    st.info("داده‌ها هنوز بارگذاری نشده‌اند. اگر فایل داخل ریپو هست، گزینه «از فایل داخل ریپو» را انتخاب کن.")
    st.stop()

# ---------------- Filters ----------------
st.subheader("فیلترها")
colf1, colf2, colf3, colf4 = st.columns(4)

zones = sorted([z for z in df.get("Zone", pd.Series(dtype=object)).dropna().unique()])
cities = sorted([c for c in df.get("City", pd.Series(dtype=object)).dropna().unique()])
types = sorted([t for t in df.get("TypeMPO", pd.Series(dtype=object)).dropna().unique()])
reasons = sorted([r for r in df.get("Reason_norm", pd.Series(dtype=object)).dropna().unique()])

zone_sel = colf1.multiselect("زون (Zone)", zones, default=[])
city_sel = colf2.multiselect("شهر (City)", cities, default=[])
type_sel = colf3.multiselect("نوع (TypeMPO)", types, default=[])
reason_sel = colf4.multiselect("ریزن (Reason)", reasons, default=[])

f = df.copy()
if zone_sel and "Zone" in f.columns:
    f = f[f["Zone"].isin(zone_sel)]
if city_sel and "City" in f.columns:
    f = f[f["City"].isin(city_sel)]
if type_sel and "TypeMPO" in f.columns:
    f = f[f["TypeMPO"].isin(type_sel)]
if reason_sel:
    f = f[f["Reason_norm"].isin(reason_sel)]

# ---------------- KPI ----------------
k1, k2, k3, k4, k5 = st.columns(5)
total = len(f)
hosts = f["Host"].nunique() if "Host" in f.columns else 0
share_outside = (f["Reason_norm"].eq(OUTSIDE_REASON).mean() * 100) if total else 0
median_orate = float(np.nanmedian(f["Orate"])) if ("Orate" in f.columns and total) else np.nan
avg_to = float(np.nanmean(f["TO"])) if ("TO" in f.columns and total) else np.nan

k1.metric("تعداد پاسخ", f"{total:,}")
k2.metric("تعداد Host یکتا", f"{hosts:,}")
k3.metric("سهم رزرو خارج از جاباما", f"{share_outside:.1f}%")
k4.metric("میانه Orate", f"{median_orate:.3f}" if not np.isnan(median_orate) else "-")
k5.metric("میانگین TO (Orders)", f"{avg_to:.2f}" if not np.isnan(avg_to) else "-")

st.divider()

# ============================================================
# 1) سهم ریزن ها
# ============================================================
st.header("۱) سهم ریزن‌ها")
rc = f["Reason_norm"].value_counts().reset_index()
rc.columns = ["ریزن", "تعداد"]
rc = rc.sort_values("تعداد", ascending=False)

c1, c2 = st.columns([1, 1])
with c1:
    bar_chart(rc, x="ریزن", y="تعداد", title="تعداد پاسخ به تفکیک ریزن (نزولی)")
with c2:
    pie_df = rc.head(10).copy()
    if len(rc) > 10:
        pie_df = pd.concat([pie_df, pd.DataFrame([{"ریزن":"سایر", "تعداد": rc.iloc[10:]["تعداد"].sum()}])], ignore_index=True)
    pie_chart(pie_df, names_col="ریزن", values_col="تعداد", title="پای‌چارت سهم ریزن‌ها")

st.divider()

# ============================================================
# 2) سهم ریزن‌های فروش خارج از جاباما
# ============================================================
st.header("۲) سهم ریزن‌های فروش خارج از جاباما (روش‌های فروش)")
fo = f[f["Reason_norm"].eq(OUTSIDE_REASON)].copy()
if len(fo) == 0:
    st.info("با فیلترهای فعلی، ریزن «رزرو خارج از جاباما» وجود ندارد.")
else:
    sm = fo["SellMethod"].value_counts().reset_index()
    sm.columns = ["روش فروش", "تعداد"]
    sm = sm.sort_values("تعداد", ascending=False)

    c3, c4 = st.columns([1, 1])
    with c3:
        bar_chart(sm, x="روش فروش", y="تعداد", title="تعداد روش‌های فروش (فقط خارج از جاباما) - نزولی")
    with c4:
        pie_sm = sm.head(10).copy()
        if len(sm) > 10:
            pie_sm = pd.concat([pie_sm, pd.DataFrame([{"روش فروش":"سایر", "تعداد": sm.iloc[10:]["تعداد"].sum()}])], ignore_index=True)
        pie_chart(pie_sm, names_col="روش فروش", values_col="تعداد", title="پای‌چارت روش‌های فروش (فقط خارج از جاباما)")

st.divider()

# ============================================================
# 3) سهم هر دو (Reason و Outside-SellMethod) به تفکیک Zone
# ============================================================
st.header("۳) سهم‌ها به تفکیک زون")

if "Zone" not in f.columns or f["Zone"].dropna().empty:
    st.info("ستون Zone موجود نیست یا بعد از فیلتر خالی است.")
else:
    # 3a) Zone x Reason share
    st.subheader("۳-الف) سهم ریزن‌ها در هر زون (درصدی)")
    zr_cnt = pd.crosstab(f["Zone"], f["Reason_norm"])
    zr_share = (zr_cnt.div(zr_cnt.sum(axis=1), axis=0) * 100).round(1)
    # order zones by total responses
    zr_share = zr_share.loc[zr_cnt.sum(axis=1).sort_values(ascending=False).index]
    st.dataframe(zr_share, use_container_width=True)

    # 3b) Zone x SellMethod share (only Outside reason)
    st.subheader("۳-ب) سهم روش‌های فروش خارج از جاباما در هر زون (درصدی)")
    if len(fo) == 0 or "Zone" not in fo.columns or fo["Zone"].dropna().empty:
        st.info("داده‌ای برای خارج از جاباما در این فیلترها وجود ندارد.")
    else:
        zsm_cnt = pd.crosstab(fo["Zone"], fo["SellMethod"])
        zsm_share = (zsm_cnt.div(zsm_cnt.sum(axis=1), axis=0) * 100).round(1)
        zsm_share = zsm_share.loc[zsm_cnt.sum(axis=1).sort_values(ascending=False).index]
        st.dataframe(zsm_share, use_container_width=True)

st.divider()

# ============================================================
# 4) Average number of orders (TO) per Reason per Zone
# ============================================================
st.header("۴) میانگین تعداد سفارش (TO) در هر ریزن به تفکیک زون")

if "TO" not in f.columns or f["TO"].dropna().empty:
    st.info("ستون TO موجود نیست یا داده‌ای ندارد.")
elif "Zone" not in f.columns or f["Zone"].dropna().empty:
    st.info("ستون Zone موجود نیست یا بعد از فیلتر خالی است.")
else:
    to_table = (
        f.groupby(["Zone", "Reason_norm"])["TO"]
        .mean()
        .reset_index()
        .rename(columns={"Reason_norm":"ریزن", "TO":"میانگین TO"})
    )
    # Keep top zones by volume to avoid huge tables
    top_zones = f["Zone"].value_counts().head(12).index.tolist()
    to_table = to_table[to_table["Zone"].isin(top_zones)]
    # Pivot for readability
    to_pivot = to_table.pivot_table(index="Zone", columns="ریزن", values="میانگین TO")
    to_pivot = to_pivot.loc[f["Zone"].value_counts().loc[to_pivot.index].sort_values(ascending=False).index]
    st.caption("Zoneهای نمایش داده شده = 12 زون برتر از نظر تعداد پاسخ در فیلتر فعلی")
    st.dataframe(to_pivot.round(2), use_container_width=True)

st.divider()

# ============================================================
# 5) For each Reason: top 3 cities by count
# ============================================================
st.header("۵) در هر ریزن: ۳ شهر با بیشترین انتخاب")

if "City" not in f.columns or f["City"].dropna().empty:
    st.info("ستون City موجود نیست یا بعد از فیلتر خالی است.")
else:
    # Build a compact table
    rows = []
    for r in f["Reason_norm"].dropna().unique():
        sub = f[f["Reason_norm"] == r]
        top3 = sub["City"].value_counts().head(3)
        for city, cnt in top3.items():
            rows.append({"ریزن": r, "شهر": city, "تعداد": int(cnt)})
    top_city_df = pd.DataFrame(rows)
    if top_city_df.empty:
        st.info("داده‌ای برای نمایش وجود ندارد.")
    else:
        top_city_df = top_city_df.sort_values(["ریزن", "تعداد"], ascending=[True, False])
        st.dataframe(top_city_df, use_container_width=True)

st.divider()

# ============================================================
# 6) Orate analysis vs survey reasons
# ============================================================
st.header("۶) تحلیل Orate در کنار نتیجه سروِی")

if "Orate" not in f.columns or f["Orate"].dropna().empty:
    st.info("ستون Orate موجود نیست یا داده‌ای ندارد.")
else:
    c5, c6 = st.columns(2)

    with c5:
        # Median Orate by reason
        med = f.groupby("Reason_norm")["Orate"].median().sort_values(ascending=False).reset_index()
        med.columns = ["ریزن", "میانه Orate"]
        st.dataframe(med.round(4), use_container_width=True)

    with c6:
        # Orate bins and reason mix
        try:
            bins = pd.qcut(f["Orate"].fillna(0), q=4, duplicates="drop")
            labels = [f"Bin {i+1}" for i in range(len(bins.cat.categories))]
            tmp = f.copy()
            tmp["Orate_bin"] = pd.qcut(tmp["Orate"].fillna(0), q=4, duplicates="drop", labels=labels)
            mix = (pd.crosstab(tmp["Orate_bin"], tmp["Reason_norm"], normalize="index") * 100).round(1)
            st.caption("ترکیب ریزن‌ها داخل هر Bin از Orate (درصدی)")
            st.dataframe(mix, use_container_width=True)
        except Exception as e:
            st.warning(f"باین‌بندی Orate انجام نشد: {e}")

st.caption("اگر هنوز در نمودارها مشکل فونت/RTL دیدی، بگو تا نسخه‌ی کامل‌تر RTL را هم اعمال کنم.")
