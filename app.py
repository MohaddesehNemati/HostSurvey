import pandas as pd
import streamlit as st
import pathlib
import glob
import numpy as np
import plotly.express as px
import re

st.set_page_config(page_title="Dashboard | بستن تقویم", layout="wide")

# --- Persian font (works well on Streamlit Cloud) ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Vazirmatn:wght@300;400;600;700&display=swap');
html, body, [class*="css"], .stApp { font-family: 'Vazirmatn', sans-serif !important; }
</style>
""", unsafe_allow_html=True)

PERSIAN_FONT = "Vazirmatn"

_candidates = sorted(glob.glob("data/*.xlsx"))
DEFAULT_XLSX_PATH = _candidates[0] if _candidates else "data/analysis.xlsx"

OUTSIDE_REASON = "رزرو خارج از جاباما (کانال‌های دیگر)"

def _apply_plotly_font(fig):
    fig.update_layout(
        font=dict(family=PERSIAN_FONT, size=14),
        legend=dict(font=dict(family=PERSIAN_FONT, size=12)),
        title=dict(font=dict(family=PERSIAN_FONT, size=16)),
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig

def pie_chart(df: pd.DataFrame, names_col: str, values_col: str, title: str):
    fig = px.pie(df, names=names_col, values=values_col, title=title, hole=0.35)
    fig.update_traces(textposition="inside", textinfo="percent+label")
    st.plotly_chart(_apply_plotly_font(fig), use_container_width=True)

def bar_chart(df: pd.DataFrame, x: str, y: str, title: str):
    fig = px.bar(df, x=x, y=y, title=title)
    st.plotly_chart(_apply_plotly_font(fig), use_container_width=True)

def _top_with_other(value_counts: pd.Series, top_n: int = 10, other_label: str = "سایر"):
    vc = value_counts.copy()
    if len(vc) <= top_n:
        out = vc.reset_index()
        out.columns = ["name", "count"]
        return out
    top = vc.head(top_n)
    other = vc.iloc[top_n:].sum()
    out = pd.concat([top, pd.Series({other_label: other})]).reset_index()
    out.columns = ["name", "count"]
    return out

# ---------------- "سایر" Tagging rules ----------------
def tag_other_reason(text: str) -> str:
    if not text:
        return "سایر/نامشخص"
    t = str(text)

    if re.search(r"تقویم\s*باز|باز\s*بوده|بسته\s*نبود|رزرو\s*توسط\s*شما|توسط\s*شما|اصلا\s*باز", t):
        return "ابهام/تناقض در داده یا وضعیت تقویم"
    if re.search(r"اپلیکیشن|سیستم|باگ|خطا|بدون\s*اطلاع|خودکار|رزرو\s*آنی", t):
        return "مشکل/باگ سیستم یا اپ"
    if re.search(r"قیمت|قیمتگذاری|قیمت\s*گذاری|پایین|بالا|گرون|ارزون", t):
        return "قیمت‌گذاری/نارضایتی قیمت"
    if re.search(r"فول|تکمیل|ظرفیت|تمامی\s*اتاق|پر\s*بود", t):
        return "تکمیل ظرفیت/فول"
    if re.search(r"خصوصی|ثابت|محلی|آشنا|خانواده|فامیل|مشتری\s*محلی", t):
        return "مهمان خصوصی/ثابت/محلی"
    if re.search(r"تمدید|قرارداد|اجاره|رهن|واگذاری", t):
        return "قرارداد/تمدید/اجاره"
    if re.search(r"بازسازی|نقاشی|تعمیر|تاسیسات|بازسازي", t):
        return "بازسازی/تعمیرات"
    if re.search(r"نمیدونستم|نمی\s*دانستم|اطلاع\s*نداشتم|فرصت\s*نکرد", t):
        return "عدم اطلاع/فرصت نکردن"
    return "سایر/متفرقه"

def tag_other_sellmethod(text: str) -> str:
    if not text:
        return "سایر/نامشخص"
    t = str(text)

    if re.search(r"دیوار|شیپور", t):
        return "آگهی (دیوار/شیپور)"
    if re.search(r"اینستاگرام|تلگرام|واتساپ|شبکه", t):
        return "شبکه‌های اجتماعی/پیام‌رسان"
    if re.search(r"سایت|وبسایت|وب\s*سایت|رزرو\s*مستقیم|تماس|شماره", t):
        return "مستقیم/وبسایت/تماس"
    if re.search(r"پلتفرم|سایر\s*پلتفرم|شب|جاجی|هتل|booking|airbnb|اقامت24|جاباما", t, re.IGNORECASE):
        return "پلتفرم‌های رزرو آنلاین"
    if re.search(r"آژانس|تور|کارگزار", t):
        return "آژانس/تور"
    if re.search(r"سازمانی|اداری|شرکت", t):
        return "سازمانی"
    return "سایر/متفرقه"

@st.cache_data
def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    # Columns
    reason_cols = [c for c in df.columns if "دلیل" in c]
    if not reason_cols:
        raise ValueError("ستونِ دلیل در دیتاست پیدا نشد.")
    reason_col = reason_cols[0]

    sell_cols = [c for c in df.columns if "روش فروش" in c]
    sell_col = sell_cols[0] if sell_cols else None

    out = df.copy()
    out["Reason_raw"] = out[reason_col].astype(str)

    def normalize_reason(x):
        if pd.isna(x):
            return "نامشخص"
        s = str(x).strip()
        if s.startswith("سایر"):
            return "سایر (متن آزاد)"
        return s

    out["Reason_norm"] = out[reason_col].apply(normalize_reason)

    # Extract free text for "سایر - ..."
    out["Reason_other_text"] = ""
    mask_other = out[reason_col].astype(str).str.startswith("سایر")
    out.loc[mask_other, "Reason_other_text"] = (
        out.loc[mask_other, reason_col]
        .astype(str)
        .str.replace(r"^سایر\s*[-–:]\s*", "", regex=True)
        .str.strip()
    )
    out["Reason_other_tag"] = out["Reason_other_text"].apply(tag_other_reason)

    # Selling methods
    if sell_col:
        out["Sell_raw"] = out[sell_col].fillna("نامشخص").astype(str).str.strip()
    else:
        out["Sell_raw"] = "نامشخص"

    out["SellMethod"] = out["Sell_raw"].replace({"nan":"نامشخص"})
    out["Sell_other_text"] = ""
    mask_sell_other = out["Sell_raw"].astype(str).str.startswith("سایر")
    out.loc[mask_sell_other, "Sell_other_text"] = (
        out.loc[mask_sell_other, "Sell_raw"]
        .astype(str)
        .str.replace(r"^سایر\s*[-–:]\s*", "", regex=True)
        .str.strip()
    )
    out["Sell_other_tag"] = out["Sell_raw"].apply(tag_other_sellmethod)

    # Numeric columns
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
# 1) سهم ریزن ها + دسته‌بندی "سایر"
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

st.subheader("۱-الف) دسته‌بندی ریزن «سایر (متن آزاد)»")
other_rows = f[f["Reason_norm"].eq("سایر (متن آزاد)")].copy()
if len(other_rows) == 0:
    st.info("با فیلترهای فعلی ریزن «سایر (متن آزاد)» وجود ندارد.")
else:
    tag_counts = other_rows["Reason_other_tag"].value_counts().reset_index()
    tag_counts.columns = ["تگ", "تعداد"]
    tag_counts = tag_counts.sort_values("تعداد", ascending=False)
    c1a, c1b = st.columns([1,1])
    with c1a:
        bar_chart(tag_counts, x="تگ", y="تعداد", title="تعداد تگ‌ها در «سایر» (ریزن اصلی)")
    with c1b:
        pie_tag = tag_counts.head(10).copy()
        if len(tag_counts) > 10:
            pie_tag = pd.concat([pie_tag, pd.DataFrame([{"تگ":"سایر", "تعداد": tag_counts.iloc[10:]["تعداد"].sum()}])], ignore_index=True)
        pie_chart(pie_tag, names_col="تگ", values_col="تعداد", title="پای‌چارت تگ‌های «سایر» (ریزن اصلی)")

    st.caption("نمونه متن‌های «سایر» به تفکیک تگ (تا 5 نمونه برای هر تگ):")
    samples = (
        other_rows.loc[other_rows["Reason_other_text"].astype(str).str.len() > 0, ["Reason_other_tag", "Reason_other_text"]]
        .drop_duplicates()
        .groupby("Reason_other_tag")
        .head(5)
        .rename(columns={"Reason_other_tag":"تگ", "Reason_other_text":"متن"})
    )
    st.dataframe(samples, use_container_width=True)

st.divider()

# ============================================================
# 2) سهم روش‌های فروش خارج از جاباما + دسته‌بندی "سایر"
# ============================================================
st.header("۲) سهم روش‌های فروش خارج از جاباما (برای ریزن «رزرو خارج از جاباما»)")
fo = f[f["Reason_norm"].eq(OUTSIDE_REASON)].copy()
if len(fo) == 0:
    st.info("با فیلترهای فعلی، ریزن «رزرو خارج از جاباما» وجود ندارد.")
else:
    sm = fo["SellMethod"].value_counts().reset_index()
    sm.columns = ["روش فروش", "تعداد"]
    sm = sm.sort_values("تعداد", ascending=False)

    c3, c4 = st.columns([1, 1])
    with c3:
        bar_chart(sm, x="روش فروش", y="تعداد", title="تعداد روش‌های فروش (نزولی)")
    with c4:
        pie_sm = sm.head(10).copy()
        if len(sm) > 10:
            pie_sm = pd.concat([pie_sm, pd.DataFrame([{"روش فروش":"سایر", "تعداد": sm.iloc[10:]["تعداد"].sum()}])], ignore_index=True)
        pie_chart(pie_sm, names_col="روش فروش", values_col="تعداد", title="پای‌چارت روش‌های فروش (فقط خارج از جاباما)")

    st.subheader("۲-الف) دسته‌بندی «سایر» در روش فروش")
    sell_other = fo[fo["Sell_raw"].astype(str).str.startswith("سایر")].copy()
    if len(sell_other) == 0:
        st.info("در روش فروشِ خارج از جاباما، موردی با شروع «سایر ...» وجود ندارد (یا بعد از فیلتر حذف شده).")
    else:
        st.caption("تگ‌ها بر اساس کلمات کلیدی (دیوار/شبکه اجتماعی/پلتفرم رزرو/...) ساخته شده‌اند.")
        sell_tag_counts = sell_other["Sell_other_tag"].value_counts().reset_index()
        sell_tag_counts.columns = ["تگ", "تعداد"]
        sell_tag_counts = sell_tag_counts.sort_values("تعداد", ascending=False)

        c2a, c2b = st.columns([1,1])
        with c2a:
            bar_chart(sell_tag_counts, x="تگ", y="تعداد", title="تعداد تگ‌ها در «سایر» (روش فروش)")
        with c2b:
            pie_sell_tag = sell_tag_counts.head(10).copy()
            if len(sell_tag_counts) > 10:
                pie_sell_tag = pd.concat([pie_sell_tag, pd.DataFrame([{"تگ":"سایر", "تعداد": sell_tag_counts.iloc[10:]["تعداد"].sum()}])], ignore_index=True)
            pie_chart(pie_sell_tag, names_col="تگ", values_col="تعداد", title="پای‌چارت تگ‌های «سایر» (روش فروش)")

        samples2 = (
            sell_other.loc[sell_other["Sell_other_text"].astype(str).str.len() > 0, ["Sell_other_tag", "Sell_other_text"]]
            .drop_duplicates()
            .groupby("Sell_other_tag")
            .head(5)
            .rename(columns={"Sell_other_tag":"تگ", "Sell_other_text":"متن"})
        )
        st.caption("نمونه متن‌های «سایر» در روش فروش (تا 5 نمونه برای هر تگ):")
        st.dataframe(samples2, use_container_width=True)

st.divider()

# ============================================================
# 3) سهم‌ها به تفکیک زون
# ============================================================
st.header("۳) سهم‌ها به تفکیک زون")

if "Zone" not in f.columns or f["Zone"].dropna().empty:
    st.info("ستون Zone موجود نیست یا بعد از فیلتر خالی است.")
else:
    st.subheader("۳-الف) سهم ریزن‌ها در هر زون (درصدی)")
    zr_cnt = pd.crosstab(f["Zone"], f["Reason_norm"])
    zr_share = (zr_cnt.div(zr_cnt.sum(axis=1), axis=0) * 100).round(1)
    zr_share = zr_share.loc[zr_cnt.sum(axis=1).sort_values(ascending=False).index]
    st.dataframe(zr_share, use_container_width=True)

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
# 4) Average TO per Reason per Zone
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
    top_zones = f["Zone"].value_counts().head(12).index.tolist()
    to_table = to_table[to_table["Zone"].isin(top_zones)]
    to_pivot = to_table.pivot_table(index="Zone", columns="ریزن", values="میانگین TO")
    to_pivot = to_pivot.loc[f["Zone"].value_counts().loc[to_pivot.index].sort_values(ascending=False).index]
    st.caption("Zoneهای نمایش داده شده = 12 زون برتر از نظر تعداد پاسخ در فیلتر فعلی")
    st.dataframe(to_pivot.round(2), use_container_width=True)

st.divider()

# ============================================================
# 5) Top 3 cities per reason
# ============================================================
st.header("۵) در هر ریزن: ۳ شهر با بیشترین انتخاب")
if "City" not in f.columns or f["City"].dropna().empty:
    st.info("ستون City موجود نیست یا بعد از فیلتر خالی است.")
else:
    rows = []
    for r in sorted(f["Reason_norm"].dropna().unique()):
        sub = f[f["Reason_norm"] == r]
        top3 = sub["City"].value_counts().head(3)
        for city, cnt in top3.items():
            rows.append({"ریزن": r, "شهر": city, "تعداد": int(cnt)})
    top_city_df = pd.DataFrame(rows)
    if top_city_df.empty:
        st.info("داده‌ای برای نمایش وجود ندارد.")
    else:
        st.dataframe(top_city_df.sort_values(["ریزن","تعداد"], ascending=[True, False]), use_container_width=True)

st.divider()

# ============================================================
# 6) Orate analysis title fixed
# ============================================================
st.header("۶) ارتباط Orate با دلایل بستن تقویم (نتیجه سروِی)")

if "Orate" not in f.columns or f["Orate"].dropna().empty:
    st.info("ستون Orate موجود نیست یا داده‌ای ندارد.")
else:
    c5, c6 = st.columns(2)

    with c5:
        med = f.groupby("Reason_norm")["Orate"].median().sort_values(ascending=False).reset_index()
        med.columns = ["ریزن", "میانه Orate"]
        st.dataframe(med.round(4), use_container_width=True)

    with c6:
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
