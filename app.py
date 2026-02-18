import pandas as pd
import streamlit as st
import pathlib
import glob
import numpy as np
import plotly.express as px
import re

st.set_page_config(page_title="Dashboard | بستن تقویم", layout="wide")

# --- Persian font ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Vazirmatn:wght@300;400;600;700&display=swap');
html, body, [class*="css"], .stApp { font-family: 'Vazirmatn', sans-serif !important; }
</style>
""", unsafe_allow_html=True)

PERSIAN_FONT = "Vazirmatn"
APP_VERSION = "v6"  # cache-busting key

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
    if df.empty:
        st.info("داده‌ای برای نمایش وجود ندارد.")
        return
    fig = px.pie(df, names=names_col, values=values_col, title=title, hole=0.35)
    fig.update_traces(textposition="inside", textinfo="percent+label")
    st.plotly_chart(_apply_plotly_font(fig), use_container_width=True)

def bar_chart(df: pd.DataFrame, x: str, y: str, title: str):
    if df.empty:
        st.info("داده‌ای برای نمایش وجود ندارد.")
        return
    fig = px.bar(df, x=x, y=y, title=title)
    st.plotly_chart(_apply_plotly_font(fig), use_container_width=True)

# ---------------- "سایر" Tagging rules ----------------
def tag_other_reason(text: str) -> str:
    if text is None:
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
    if text is None:
        return "سایر/نامشخص"
    t = str(text)

    if re.search(r"دیوار|شیپور", t):
        return "آگهی (دیوار/شیپور)"
    if re.search(r"اینستاگرام|تلگرام|واتساپ|شبکه", t):
        return "شبکه‌های اجتماعی/پیام‌رسان"
    if re.search(r"سایت|وبسایت|وب\s*سایت|رزرو\s*مستقیم|تماس|شماره", t):
        return "مستقیم/وبسایت/تماس"
    if re.search(r"پلتفرم|سایر\s*پلتفرم|شب|جاجی|booking|airbnb|اقامت24|هتل", t, re.IGNORECASE):
        return "پلتفرم‌های رزرو آنلاین"
    if re.search(r"آژانس|تور|کارگزار", t):
        return "آژانس/تور"
    if re.search(r"سازمانی|اداری|شرکت", t):
        return "سازمانی"
    return "سایر/متفرقه"

def _ensure_other_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    reason_cols = [c for c in out.columns if "دلیل" in c]
    reason_col = reason_cols[0] if reason_cols else None
    sell_cols = [c for c in out.columns if "روش فروش" in c]
    sell_col = sell_cols[0] if sell_cols else None

    if "Reason_norm" not in out.columns and reason_col:
        def normalize_reason(x):
            if pd.isna(x): return "نامشخص"
            s = str(x).strip()
            if s.startswith("سایر"): return "سایر (متن آزاد)"
            return s
        out["Reason_norm"] = out[reason_col].apply(normalize_reason)

    if "Reason_other_text" not in out.columns:
        out["Reason_other_text"] = ""
        if reason_col:
            mask_other = out[reason_col].astype(str).str.startswith("سایر")
            out.loc[mask_other, "Reason_other_text"] = (
                out.loc[mask_other, reason_col].astype(str)
                .str.replace(r"^سایر\s*[-–:]\s*", "", regex=True).str.strip()
            )

    if "Reason_other_tag" not in out.columns:
        out["Reason_other_tag"] = out["Reason_other_text"].apply(tag_other_reason)

    if "Sell_raw" not in out.columns:
        if sell_col:
            out["Sell_raw"] = out[sell_col].fillna("نامشخص").astype(str).str.strip()
        else:
            out["Sell_raw"] = "نامشخص"

    if "SellMethod" not in out.columns:
        out["SellMethod"] = out["Sell_raw"].replace({"nan":"نامشخص"})

    if "Sell_other_text" not in out.columns:
        out["Sell_other_text"] = ""
        mask_sell_other = out["Sell_raw"].astype(str).str.startswith("سایر")
        out.loc[mask_sell_other, "Sell_other_text"] = (
            out.loc[mask_sell_other, "Sell_raw"].astype(str)
            .str.replace(r"^سایر\s*[-–:]\s*", "", regex=True).str.strip()
        )

    if "Sell_other_tag" not in out.columns:
        out["Sell_other_tag"] = out["Sell_raw"].apply(tag_other_sellmethod)

    for col in ["Orate", "TO", "TFO", "RateScore", "RateCount", "Capacity", "AIV", "NMV", "AccommodationCountPerHost"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    return out

@st.cache_data
def load_data_from_path(path: str, _version: str) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name="Merged_data")
    return _ensure_other_columns(df)

@st.cache_data
def load_data_from_upload(uploaded_file, _version: str) -> pd.DataFrame:
    df = pd.read_excel(uploaded_file, sheet_name="Merged_data")
    return _ensure_other_columns(df)

st.title("داشبورد تحلیل بستن تقویم (تعطیلات ۲۱ تا ۲۴ بهمن)")

with st.sidebar:
    st.header("منبع داده")
    st.caption("فایل‌های داخل پوشه data/ که اپ می‌بیند:")
    st.write(sorted(glob.glob("data/*.xlsx")))

    if st.button("پاک‌کردن کش (Fix errors)"):
        st.cache_data.clear()
        st.success("کش پاک شد. صفحه را Refresh کن یا دوباره Run کن.")

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
            df = load_data_from_path(xlsx_path, APP_VERSION)
        except Exception as e:
            st.error(f"خطا در خواندن فایل: {e}")
    else:
        try:
            if pathlib.Path(xlsx_path).exists():
                df = load_data_from_path(xlsx_path, APP_VERSION)
        except Exception:
            pass
else:
    if uploaded is not None:
        try:
            df = load_data_from_upload(uploaded, APP_VERSION)
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

# Safety: ensure derived columns exist even after filtering
f = _ensure_other_columns(f)

# ---------------- Top Summary (requested) ----------------
# 1) number of responses
total = len(f)

# 2) number of hosts
hosts = f["Host"].nunique() if "Host" in f.columns else 0

# 3) accommodations per host (use provided column if exists, else compute as unique AccommodationCountPerHost mean, else fallback)
acc_per_host = np.nan
if "AccommodationCountPerHost" in f.columns and f["AccommodationCountPerHost"].notna().any():
    # avoid over-weighting hosts with multiple responses
    tmp = f.dropna(subset=["Host"]).drop_duplicates(subset=["Host"])
    acc_per_host = float(np.nanmean(tmp["AccommodationCountPerHost"]))
else:
    # fallback: if accommodation_id exists
    acc_id_cols = [c for c in f.columns if "Accommodation" in c and "Count" not in c]
    # can't reliably infer; keep nan
    acc_per_host = np.nan

# 4) top reason among reasons
top_reason = "-"
if "Reason_norm" in f.columns and total:
    top_reason = f["Reason_norm"].value_counts().idxmax()

# 5) top sell method within outside reason
top_outside_sell = "-"
fo = f[f["Reason_norm"].eq(OUTSIDE_REASON)].copy()
if len(fo) > 0:
    top_outside_sell = fo["SellMethod"].value_counts().idxmax()

st.subheader("خلاصه کلی (براساس فیلترهای بالا)")
s1, s2, s3, s4, s5 = st.columns(5)
s1.metric("تعداد پاسخ‌ها", f"{total:,}")
s2.metric("تعداد هاست‌ها", f"{hosts:,}")
s3.metric("تعداد اقامتگاه به ازای هر هاست", f"{acc_per_host:.2f}" if not np.isnan(acc_per_host) else "-")
s4.metric("بیشترین ریزن در دلایل", top_reason)
s5.metric("بیشترین ریزن در فروش خارج از جاباما", top_outside_sell)

st.divider()

# ============================================================
# ۱) سهم ریزن‌ها + دسته‌بندی "سایر"
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

st.divider()

# ============================================================
# ۲) سهم روش‌های فروش خارج از جاباما + دسته‌بندی "سایر"
# ============================================================
st.header("۲) سهم روش‌های فروش خارج از جاباما (برای ریزن «رزرو خارج از جاباما»)")
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
        sell_tag_counts = sell_other["Sell_other_tag"].value_counts().reset_index()
        sell_tag_counts.columns = ["تگ", "تعداد"]
        sell_tag_counts = sell_tag_counts.sort_values("تعداد", ascending=False)
        bar_chart(sell_tag_counts, x="تگ", y="تعداد", title="تعداد تگ‌ها در «سایر» (روش فروش)")

st.divider()

# ============================================================
# ۳) سهم‌ها به تفکیک زون
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
# ۴) میانگین TO در هر ریزن به تفکیک زون
# ============================================================
st.header("۴) میانگین تعداد سفارش (TO) در هر ریزن به تفکیک زون")
if "TO" not in f.columns or f["TO"].dropna().empty:
    st.info("ستون TO موجود نیست یا داده‌ای ندارد.")
elif "Zone" not in f.columns or f["Zone"].dropna().empty:
    st.info("ستون Zone موجود نیست یا بعد از فیلتر خالی است.")
else:
    to_table = (
        f.groupby(["Zone", "Reason_norm"])["TO"].mean().reset_index()
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
# ۵) در هر ریزن: ۳ شهر با بیشترین انتخاب
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
    st.dataframe(top_city_df.sort_values(["ریزن","تعداد"], ascending=[True, False]), use_container_width=True)

st.divider()

# ============================================================
# ۶) ارتباط Orate با دلایل بستن تقویم (نتیجه سروِی)
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
