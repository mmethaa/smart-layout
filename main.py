import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split

# -------------------- CONFIG --------------------
st.set_page_config(page_title="Smart Layout AI", page_icon="🎗️", layout="centered")

# -------------------- STYLING --------------------
st.markdown("""
    <style>
    html, body, [class*="css"]  {
        font-family: 'Segoe UI', sans-serif;
        background-color: #f0f4f8;
    }
    .stButton>button {
        background: linear-gradient(to right, #0f4c75, #3282b8);
        color: white;
        font-size: 18px;
        border-radius: 10px;
        padding: 0.6em 2em;
    }
    div[data-testid="metric-container"] {
        background-color: white;
        border-radius: 12px;
        padding: 1em;
        margin: 10px 0;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.08);
    }
    div[data-testid="metric-container"] > label, div[data-testid="metric-container"] > div {
        color: #1f2937 !important;
        font-weight: 600;
        font-size: 18px;
    }
    </style>
""", unsafe_allow_html=True)

# -------------------- HEADER --------------------
st.markdown("""
<div style="background: linear-gradient(to right, #0f4c75, #3282b8); padding: 2rem; border-radius: 12px; text-align: center;">
    <h1 style="color: white;">Smart Layout Predictor 🏢</h1>
    <p style="color: #d9ecf2; font-size: 18px;">ระบบพยากรณ์ผังโครงการอสังหาฯ สำหรับ Developer และ Builder</p>
</div>
""", unsafe_allow_html=True)

# -------------------- LOT SIZE OPTIMIZER --------------------
def optimize_lot_count(total_area, ratio, min_lot_size, max_lot_size, enforce_even=False):
    avg_lot_size = (min_lot_size + max_lot_size) / 2
    lot_area = total_area * ratio
    lot_count = int(round(lot_area / avg_lot_size))
    if enforce_even and lot_count % 2 != 0:
        lot_count += 1
    used_area = lot_count * avg_lot_size
    return lot_count, used_area

LOT_SIZES = {
    "ทาวน์โฮม": (20.08, 45.84),
    "บ้านแฝด": (47, 70),
    "บ้านเดี่ยว2ชั้น": (60, 160),
    "อาคารพาณิชย์": (20, 40)
}

# -------------------- LOAD DATA --------------------
df = pd.read_excel("data update.xlsx")
df.columns = df.columns.str.strip()

# ✅ คำนวณฟีเจอร์
df['หลังต่อซอย'] = df['จำนวนหลัง'] / df['จำนวนซอย'].replace(0, 1)
df['%บ้านเดี่ยว2ชั้น'] = df['บ้านเดี่ยว2ชั้น'] / df['จำนวนหลัง'].replace(0, 1)
if '%บ้านเดี่ยว' in df.columns:
    df = df.drop(columns=['%บ้านเดี่ยว'])
df['%บ้านแฝด'] = df['บ้านแฝด'] / df['จำนวนหลัง'].replace(0, 1)
df['%ทาวโฮม'] = df['ทาวโฮม'] / df['จำนวนหลัง'].replace(0, 1)
df['%พื้นที่ขาย'] = df['พื้นที่จัดจำหน่าย(ตรม)'] / df['พื้นที่โครงการ(ตรม)']
df['%พื้นที่สาธา'] = df['พื้นที่สาธา(ตรม)'] / df['พื้นที่โครงการ(ตรม)']

X_raw = df[['จังหวัด', 'เกรดโครงการ', 'พื้นที่โครงการ(ตรม)', 'รูปร่างที่ดิน']]
y_ratio = pd.DataFrame({
    'สัดส่วนพื้นที่สาธา': df['%พื้นที่สาธา'],
    'สัดส่วนพื้นที่จัดจำหน่าย': df['%พื้นที่ขาย'],
    'สัดส่วนพื้นที่สวน': df['พื้นที่สวน(5%ของพื้นที่จัดจำหน่าย)'] / df['พื้นที่โครงการ(ตรม)'],
    'จำนวนหลังต่อไร่': df['จำนวนหลัง'] / (df['พื้นที่โครงการ(ตรม)'] / 1600),
    'สัดส่วนทาวโฮม': df['%ทาวโฮม'],
    'สัดส่วนบ้านแฝด': df['%บ้านแฝด'],
    'สัดส่วนบ้านเดี่ยว2ชั้น': df['%บ้านเดี่ยว2ชั้น'],
    'สัดส่วนอาคารพาณิชย์': df['อาคารพาณิชย์'] / df['จำนวนหลัง'].replace(0, 1)
})

X = pd.get_dummies(X_raw, columns=['จังหวัด', 'เกรดโครงการ', 'รูปร่างที่ดิน'])
X_train, _, y_train, _ = train_test_split(X, y_ratio, test_size=0.2, random_state=42)
model = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42)).fit(X_train, y_train)
avg_ซอยต่อหลัง = df.groupby('เกรดโครงการ')['หลังต่อซอย'].mean().to_dict()

# -------------------- FORM --------------------
st.markdown("## 📋 กรอกข้อมูลโครงการ")
with st.form("input_form"):
    col1, col2 = st.columns(2)
    with col1:
        จังหวัด = st.selectbox("📍 จังหวัด", sorted(df['จังหวัด'].dropna().unique()))
        รูปร่าง = st.selectbox("🧱️ รูปร่างที่ดิน", sorted(df['รูปร่างที่ดิน'].dropna().unique()))
    with col2:
        เกรด = st.selectbox("🏧 เกรดโครงการ", sorted(df['เกรดโครงการ'].dropna().unique()))
        พื้นที่ = st.number_input("📀 พื้นที่โครงการ (ตร.ม.)", min_value=1000, value=30000, step=500)
    submitted = st.form_submit_button("🚀 เริ่มพยากรณ์")

if submitted:
    area = พื้นที่
    input_df = pd.DataFrame([{ 'จังหวัด': จังหวัด, 'เกรดโครงการ': เกรด, 'พื้นที่โครงการ(ตรม)': area, 'รูปร่างที่ดิน': รูปร่าง }])
    encoded = pd.get_dummies(input_df)
    for col in X.columns:
        if col not in encoded.columns:
            encoded[col] = 0
    encoded = encoded[X.columns]
    pred = model.predict(encoded)[0]

    พท_สาธา = pred[0] * area
    พท_ขาย = pred[1] * area
    พท_สวน = pred[2] * area
    พท_ขาย_twa = พท_ขาย / 4

    สัดส่วน = pred[4:8]
    รวม = sum(สัดส่วน) or 1
    ratio = [x / รวม for x in สัดส่วน]

    ทาวโฮม, _ = optimize_lot_count(พท_ขาย_twa, ratio[0], *LOT_SIZES["ทาวน์โฮม"])
    บ้านแฝด, _ = optimize_lot_count(พท_ขาย_twa, ratio[1], *LOT_SIZES["บ้านแฝด"], enforce_even=True)
    บ้านเดี่ยว2ชั้น, _ = optimize_lot_count(พท_ขาย_twa, ratio[2], *LOT_SIZES["บ้านเดี่ยว2ชั้น"])
    อาคารพาณิชย์, _ = optimize_lot_count(พท_ขาย_twa, ratio[3], *LOT_SIZES["อาคารพาณิชย์"])

    หลังรวม = ทาวโฮม + บ้านแฝด + บ้านเดี่ยว2ชั้น + อาคารพาณิชย์
    ซอย = หลังรวม / avg_ซอยต่อหลัง.get(เกรด, 12)

    st.markdown("---")
    st.markdown("## 🌟 ผลลัพพยากรณ์")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("พื้นที่สาธารณะ", f"{พท_สาธา / 4:,.0f} ตร.วา")
        st.metric("พื้นที่จัดจำหน่าย", f"{พท_ขาย / 4:,.0f} ตร.วา")
        st.metric("พื้นที่สวน", f"{พท_สวน / 4:,.0f} ตร.วา")
    with col2:
        st.metric("จำนวนแปลงรวม", f"{หลังรวม:,.0f} หลัง")
        st.metric("จำนวนซอย", f"{ซอย:,.0f} ซอย")

    st.markdown("### 🏡 แยกตามประเภศบ้าน")
    st.markdown(f"""
    - ทาวน์โฮม: **{ทาวโฮม:,}** หลัง
    - บ้านแฝด: **{บ้านแฝด:,}** หลัง
    - บ้านเดี่ยว 2 ชั้น: **{บ้านเดี่ยว2ชั้น:,}** หลัง
    - อาคารพาณิชย์: **{อาคารพาณิชย์:,}** หลัง
    """)

st.markdown("---")
st.caption("Developed by mmethaa | Smart Layout AI 🚀")
