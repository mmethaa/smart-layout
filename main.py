import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
import os

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

# -------------------- LOAD DATA --------------------
data_path = "data update (2).xlsx"
if not os.path.exists(data_path):
    st.error("ไม่พบไฟล์ข้อมูล 'data update (2).xlsx'")
    st.stop()

df = pd.read_excel(data_path)
df.columns = df.columns.str.strip()

required_cols = ["บ้านเดี่ยว3ชั้น", "จำนวนหลัง"]
missing_cols = [col for col in required_cols if col not in df.columns]
if missing_cols:
    st.error(f"คอลัมน์หายไป: {', '.join(missing_cols)}")
    st.stop()

# Fill missing with 0 for safety
df.fillna(0, inplace=True)

df['หลังต่อซอย'] = df['จำนวนหลัง'] / df['จำนวนซอย'].replace(0, 1)
df['%บ้านเดี่ยว2ชั้น'] = df.get('บ้านเดี่ยว2ชั้น', 0) / df['จำนวนหลัง'].replace(0, 1)
df['%บ้านเดี่ยว3ชั้น'] = df.get('บ้านเดี่ยว3ชั้น', 0) / df['จำนวนหลัง'].replace(0, 1)
df['%บ้านแฝด'] = df.get('บ้านแฝด', 0) / df['จำนวนหลัง'].replace(0, 1)
df['%ทาวโฮม'] = df.get('ทาวโฮม', 0) / df['จำนวนหลัง'].replace(0, 1)
df['%พื้นที่ขาย'] = df.get('พื้นที่จัดจำหน่าย(ตรม)', 0) / df['พื้นที่โครงการ(ตรม)']
df['%พื้นที่สาธา'] = df.get('พื้นที่สาธา(ตรม)', 0) / df['พื้นที่โครงการ(ตรม)']
df['%พื้นที่ถนน'] = df.get('พื้นที่ถนนรวม(ตร.ม.)', pd.NA) / df['พื้นที่โครงการ(ตรม)']
df['%พื้นที่ถนน'] = df['%พื้นที่ถนน'].fillna(df['%พื้นที่ถนน'].mean())

X_raw = df[['จังหวัด', 'เกรดโครงการ', 'พื้นที่โครงการ(ตรม)', 'รูปร่างที่ดิน']]
y_ratio = pd.DataFrame({
    'สัดส่วนพื้นที่สาธา': df['%พื้นที่สาธา'],
    'สัดส่วนพื้นที่จัดจำหน่าย': df['%พื้นที่ขาย'],
    'สัดส่วนพื้นที่สวน': df.get('พื้นที่สวน(5%ของพื้นที่จัดจำหน่าย)', 0) / df['พื้นที่โครงการ(ตรม)'],
    'จำนวนหลังต่อไร่': df['จำนวนหลัง'] / (df['พื้นที่โครงการ(ตรม)'] / 1600),
    'สัดส่วนทาวโฮม': df['%ทาวโฮม'],
    'สัดส่วนบ้านแฝด': df['%บ้านแฝด'],
    'สัดส่วนบ้านเดี่ยว2ชั้น': df['%บ้านเดี่ยว2ชั้น'],
    'สัดส่วนบ้านเดี่ยว3ชั้น': df['%บ้านเดี่ยว3ชั้น'],
    'สัดส่วนพื้นที่ถนน': df['%พื้นที่ถนน']
})

X = pd.get_dummies(X_raw, columns=['จังหวัด', 'เกรดโครงการ', 'รูปร่างที่ดิน'])
X_train, _, y_train, _ = train_test_split(X, y_ratio.fillna(0), test_size=0.2, random_state=42)
model = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42)).fit(X_train, y_train)
avg_ซอยต่อหลัง = df.groupby('เกรดโครงการ')['หลังต่อซอย'].mean().to_dict()

# -------------------- FORM --------------------
st.markdown("## 📋 กรอกข้อมูลโครงการ")
with st.form("input_form"):
    col1, col2 = st.columns(2)
    with col1:
        จังหวัด = st.selectbox("📍 จังหวัด", sorted(df['จังหวัด'].dropna().unique()))
        รูปร่าง = st.selectbox("🦡️ รูปร่างที่ดิน", sorted(df['รูปร่างที่ดิน'].dropna().unique()))
    with col2:
        เกรด = st.selectbox("🏮 เกรดโครงการ", sorted(df['เกรดโครงการ'].dropna().unique()))
        พื้นที่ = st.number_input("📀 พื้นที่โครงการ (ตร.ม.)", min_value=1000, value=30000, step=500)
    submitted = st.form_submit_button("🚀 เริ่มพยากรณ์")

if submitted:
    area = พื้นที่
    rai = area / 1600
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
    หลังรวม = pred[3] * rai
    พท_ถนน = pred[8] * area

    รวมสัดส่วน = sum(pred[4:8]) or 1
    ทาวโฮม = หลังรวม * pred[4] / รวมสัดส่วน
    บ้านแฝด = หลังรวม * pred[5] / รวมสัดส่วน
    บ้านเดี่ยว2 = หลังรวม * pred[6] / รวมสัดส่วน
    บ้านเดี่ยว3 = หลังรวม * pred[7] / รวมสัดส่วน

    if เกรด not in ["ELEGANCE", "GRANDESSENCE", "ESSENCE"]:
        บ้านเดี่ยว3 = 0
        บ้านเดี่ยว2 = หลังรวม - (ทาวโฮม + บ้านแฝด)

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
        st.metric("พื้นที่ถนน (คาดการณ์)", f"{พท_ถนน / 4:,.0f} ตร.วา")

    st.markdown("### 🏡 แยกตามประเภทบ้าน")
    st.markdown(f"""
    - ทาวน์โฮม: **{ทาวโฮม:,.0f}** หลัง
    - บ้านแฝด: **{บ้านแฝด:,.0f}** หลัง
    - บ้านเดี่ยว 2 ชั้น: **{บ้านเดี่ยว2:,.0f}** หลัง
    - บ้านเดี่ยว 3 ชั้น: **{บ้านเดี่ยว3:,.0f}** หลัง
    """)

st.markdown("---")
st.caption("Developed by mmethaa | Smart Layout AI 🚀")

