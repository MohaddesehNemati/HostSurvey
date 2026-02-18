import pandas as pd
import streamlit as st
import pathlib
import glob
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Dashboard | Calendar Closure", layout="wide")

# Prefer any xlsx inside data/ (so we don't depend on a specific filename)
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

    # Optional: selling method column
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

    # A clean selling method (optional)
    if sell_col:
        out["SellMethod"] = out[sell_col].fillna("نامشخص").astype(str).str.strip()
        out.loc[out["SellMethod"].eq("nan"), "SellMethod"] = "نامشخص"
    else:
        out["SellMethod"] = "نامشخص"

    # Ensure numeric Orate if exists
    if "Orate" in out.columns:
        out["Orate"] = pd.to_numeric(out["Orate"], errors="coerce")

    return out

@st.cache_data
def load_data_from_path(path: str) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name="Merged_data")
    return _normalize(df)

@st.cache_data
def load_data_from_upload(uploaded_file) -> pd.DataFrame:
    df = pd.read_excel(uploaded_file, sheet_name="Merged_data")
    return _normalize(df)

def pie_from_series(s: pd.Series, title: str, top_n: int = 8):
    counts = s.value_counts(dropna=False)
    if len(counts) > top_n:
        top = counts.head(top_n)
        other = counts.iloc[top_n:].sum()
        counts = pd.concat([top, pd.Series({"سایر": other})])
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.pie(counts.values, labels=counts.index, autopct="%1.1f%%", startangle=90)
    ax.set_title(title)
    st.pyplot(fig, clear_figure=True)

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

zone_sel = colf1.multiselect("Zone", zones, default=[])
city_sel = colf2.multiselect("City", cities, default=[])
type_sel = colf3.multiselect("TypeMPO", types, default=[])
reason_sel = colf4.multiselect("Reason", reasons, default=[])

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
share_repairs = (f["Reason_norm"].eq("تعمیرات یا در دسترس نبودن اقامتگاه").mean() * 100) if total else 0
median_orate = float(np.nanmedian(f["Orate"])) if ("Orate" in f.columns and total) else np.nan

k1.metric("تعداد پاسخ", f"{total:,}")
k2.metric("تعداد Host یکتا", f"{hosts:,}")
k3.metric("سهم رزرو خارج از جاباما", f"{share_outside:.1f}%")
k4.metric("سهم تعمیرات/عدم دسترسی", f"{share_repairs:.1f}%")
k5.metric("میانه Orate", f"{median_orate:.3f}" if not np.isnan(median_orate) else "-")

st.divider()

# ---------------- Charts: Reasons ----------------
st.subheader("تحلیل دلایل (Survey Reasons)")

c1, c2 = st.columns([1, 1])
with c1:
    # Bar chart (descending)
    reason_counts = f["Reason_norm"].value_counts().sort_values(ascending=False)
    st.bar_chart(reason_counts)
with c2:
    # Pie chart: share of each reason
    pie_from_series(f["Reason_norm"], "پای‌چارت سهم هر دلیل", top_n=8)

# Pie: selling methods for Outside Jabama reason
st.markdown("### فروش خارج از جاباما: سهم روش‌های فروش (برای ریزن «رزرو خارج از جاباما»)")
fo = f[f["Reason_norm"].eq(OUTSIDE_REASON)]
if len(fo) == 0:
    st.info("با فیلترهای فعلی ریزن «رزرو خارج از جاباما» وجود ندارد.")
else:
    pie_from_series(fo["SellMethod"], "پای‌چارت روش فروش (فقط ریزن خارج از جاباما)", top_n=8)

st.divider()

# ---------------- Zone x Reason Shares ----------------
st.subheader("سهم هر دلیل در هر Zone")

if "Zone" in f.columns and f["Zone"].notna().any():
    zone_reason_counts = pd.crosstab(f["Zone"], f["Reason_norm"])
    zone_totals = zone_reason_counts.sum(axis=1)
    zone_reason_share = (zone_reason_counts.div(zone_totals, axis=0) * 100).round(1)

    # Order zones by total responses (descending)
    zone_reason_share = zone_reason_share.loc[zone_totals.sort_values(ascending=False).index]

    st.caption("اعداد = درصد سهم هر دلیل در داخل همان Zone (جمع هر ردیف ~100%)")
    st.dataframe(zone_reason_share, use_container_width=True)

    # Optional: pick a zone and show pie
    z_pick = st.selectbox("برای مشاهده پای‌چارت، یک Zone انتخاب کن:", ["-"] + list(zone_reason_share.index))
    if z_pick != "-":
        pie_from_series(f.loc[f["Zone"] == z_pick, "Reason_norm"], f"پای‌چارت دلایل در Zone: {z_pick}", top_n=8)
else:
    st.info("ستون Zone در داده موجود نیست یا بعد از فیلتر خالی است.")

st.divider()

# ---------------- Orate Analysis ----------------
st.subheader("تحلیل Orate و ارتباط با دلایل")

if "Orate" not in f.columns or f["Orate"].dropna().empty:
    st.info("ستون Orate موجود نیست یا بعد از فیلتر داده‌ای ندارد.")
else:
    ocol1, ocol2 = st.columns([1, 1])

    with ocol1:
        st.markdown("**توزیع Orate**")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(f["Orate"].dropna().values, bins=25)
        ax.set_xlabel("Orate")
        ax.set_ylabel("Count")
        ax.set_title("Histogram of Orate")
        st.pyplot(fig, clear_figure=True)

    with ocol2:
        st.markdown("**میانه Orate به تفکیک دلیل**")
        med_by_reason = f.groupby("Reason_norm")["Orate"].median().sort_values(ascending=False).round(4)
        st.dataframe(med_by_reason.reset_index().rename(columns={"Reason_norm":"دلیل","Orate":"میانه Orate"}), use_container_width=True)

    # Orate bins -> reason mix
    st.markdown("**ترکیب دلایل در سطوح مختلف Orate (Bin)**")
    try:
        bins = pd.qcut(f["Orate"].fillna(0), q=4, duplicates="drop")
        # qcut may produce 3 bins if duplicates
        labels = [f"Bin {i+1}" for i in range(len(bins.cat.categories))]
        f2 = f.copy()
        f2["Orate_bin"] = pd.qcut(f2["Orate"].fillna(0), q=4, duplicates="drop", labels=labels)

        mix = pd.crosstab(f2["Orate_bin"], f2["Reason_norm"], normalize="index") * 100
        mix = mix.round(1)
        st.caption("اعداد = درصد سهم هر دلیل داخل هر Bin از Orate (جمع هر ردیف ~100%)")
        st.dataframe(mix, use_container_width=True)
    except Exception as e:
        st.warning(f"به دلیل توزیع خاص Orate، باین‌بندی انجام نشد: {e}")

st.divider()

# ---------------- More complete summary tables (instead of sample 200 rows) ----------------
st.subheader("خلاصه‌های تکمیلی (به‌جای جدول نمونه 200 ردیف)")

t1, t2 = st.columns(2)
with t1:
    st.markdown("**Top Zone (Count)**")
    if "Zone" in f.columns:
        tz = f["Zone"].value_counts().reset_index()
        tz.columns = ["Zone", "Count"]
        st.dataframe(tz.head(30), use_container_width=True)
    else:
        st.info("Zone موجود نیست.")

with t2:
    st.markdown("**Top City (Count)**")
    if "City" in f.columns:
        tc = f["City"].value_counts().reset_index()
        tc.columns = ["City", "Count"]
        st.dataframe(tc.head(30), use_container_width=True)
    else:
        st.info("City موجود نیست.")

st.caption("نکته: جدول «نمونه 200 ردیف» حذف شد. اگر جدول کامل لازم داری، می‌تونیم دانلود CSV/Excel اضافه کنیم.")
