import pandas as pd
import streamlit as st
import pathlib
import glob

st.set_page_config(page_title="Dashboard | Calendar Closure", layout="wide")

# Prefer any xlsx inside data/ (so we don't depend on a specific filename)
_candidates = sorted(glob.glob("data/*.xlsx"))
DEFAULT_XLSX_PATH = _candidates[0] if _candidates else "data/analysis.xlsx"

@st.cache_data
def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    reason_cols = [c for c in df.columns if "دلیل" in c]
    if not reason_cols:
        raise ValueError("ستونِ دلیل در دیتاست پیدا نشد.")
    reason_col = reason_cols[0]

    def normalize_reason(x):
        if pd.isna(x):
            return "نامشخص"
        s = str(x).strip()
        if s.startswith("سایر"):
            return "سایر (متن آزاد)"
        return s

    out = df.copy()
    out["Reason_norm"] = out[reason_col].apply(normalize_reason)
    return out

@st.cache_data
def load_data_from_path(path: str) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name="Merged_data")
    return _normalize(df)

@st.cache_data
def load_data_from_upload(uploaded_file) -> pd.DataFrame:
    df = pd.read_excel(uploaded_file, sheet_name="Merged_data")
    return _normalize(df)

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

# Filters
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

# KPIs
k1, k2, k3, k4 = st.columns(4)
total = len(f)
hosts = f["Host"].nunique() if "Host" in f.columns else 0
share_outside = (f["Reason_norm"].eq("رزرو خارج از جاباما (کانال‌های دیگر)").mean() * 100) if total else 0
share_repairs = (f["Reason_norm"].eq("تعمیرات یا در دسترس نبودن اقامتگاه").mean() * 100) if total else 0

k1.metric("تعداد پاسخ", f"{total:,}")
k2.metric("تعداد Host یکتا", f"{hosts:,}")
k3.metric("سهم رزرو خارج از جاباما", f"{share_outside:.1f}%")
k4.metric("سهم تعمیرات/عدم دسترسی", f"{share_repairs:.1f}%")

st.divider()

left, right = st.columns([1.15, 0.85])

with left:
    st.subheader("دلایل اصلی (Count)")
    reason_counts = f["Reason_norm"].value_counts().reset_index()
    reason_counts.columns = ["Reason", "Count"]
    st.bar_chart(reason_counts.set_index("Reason"))

with right:
    st.subheader("دلایل به تفکیک Zone (Top 12)")
    if "Zone" in f.columns:
        pivot = pd.pivot_table(
            f, index="Zone", columns="Reason_norm", values="Host",
            aggfunc="count", fill_value=0
        )
        pivot["Total"] = pivot.sum(axis=1)
        pivot = pivot.sort_values("Total", ascending=False).head(12).drop(columns=["Total"])
        st.dataframe(pivot, use_container_width=True)
    else:
        st.info("ستون Zone در داده موجود نیست.")

st.divider()

c1, c2 = st.columns(2)
with c1:
    st.subheader("Top City (براساس تعداد پاسخ)")
    if "City" in f.columns:
        city_tab = f.groupby("City").size().sort_values(ascending=False).head(20).reset_index()
        city_tab.columns = ["City", "Count"]
        st.dataframe(city_tab, use_container_width=True)
    else:
        st.info("ستون City در داده موجود نیست.")

with c2:
    st.subheader("جدول دیتای فیلترشده (نمونه 200 ردیف)")
    cols_show = ["Host","Zone","City","TypeMPO","Reason_norm","Orate","RateScore","RateCount","Capacity","AIV","NMV"]
    cols_show = [c for c in cols_show if c in f.columns]
    st.dataframe(f[cols_show].head(200), use_container_width=True)

st.caption("اگر فایل‌ها داخل data/ باشند، داشبورد بدون آپلود دستی اجرا می‌شود.")
