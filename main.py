import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
import joblib

# ---------- CONFIG ----------------
st.set_page_config(page_title="Smart Layout", page_icon="🏗️", layout="centered")

# ---------- STYLE ----------------
st.markdown("""
    <style>
    .main {
        background-color: white;
        padding: 2rem;
        border-radius: 8px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    }
    .stButton>button {
        background-color: #1f4e79;
        color: white;
        padding: 0.6em 1.2em;
        font-size: 1.1em;
        border-radius: 5px;
    }
    </style>
""", unsafe_allow_html=True)

# ---------- HEADER ----------------
st.markdown("<h1 style='color:#1f4e79;'>🏗️ Smart Layout Predictor</h1>", unsafe_allow_html=True)
st.markdown("ระบบพยากรณ์ผังโครงการจัดสรร สำหรับ Developer และ Builder")
st.markdown("---")

# ---------- LOAD DATA & MODEL ------------
df = pd.read_excel("project_data.xlsx")
df.columns = df.columns.str.strip()

# ฟีเจอร์ช่วยคำนวณ
df['หลังต่อซอย'] = df['จำนวนหลัง'] / df['จำนวนซอย'].replace(0, 1)
df['%บ้านเดี่ยว'] = df['บ้านเดี่ยว'] / df['จำนวนหลัง'].replace(0, 1)
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
    'สัดส่วนบ้านเดี่ยว': df['%บ้านเดี่ยว'],
    'สัดส่วนอาคารพาณิชย์': df['อาคารพาณิชย์'] / df['จำนวนหลัง'].replace(0, 1)
})

X = pd.get_dummies(X_raw, columns=['จังหวัด', 'เกรดโครงการ', 'รูปร่างที่ดิน'])
model = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42)).fit(X, y_ratio)

avg_ซอยต่อหลัง = df.groupby('เกรดโครงการ')['หลังต่อซอย'].mean().to_dict()

# ---------- FORM UI ----------------
col1, col2 = st.columns(2)
with col1:
    province = st.selectbox("📍 จังหวัด", sorted(df['จังหวัด'].dropna().unique()))
    shape = st.selectbox("🧱 รูปร่างที่ดิน", sorted(df['รูปร่างที่ดิน'].dropna().unique()))
with col2:
    grade = st.selectbox("🏷️ เกรดโครงการ", sorted(df['เกรดโครงการ'].dropna().unique()))
    area = st.number_input("📐 พื้นที่โครงการ (ตร.ม.)", min_value=1000, value=30000, step=1000)

# ---------- PREDICT BUTTON ----------------
if st.button("🚀 พยากรณ์"):
    rai = area / 1600

    new_df = pd.DataFrame([{
        'จังหวัด': province,
        'เกรดโครงการ': grade,
        'พื้นที่โครงการ(ตรม)': area,
        'รูปร่างที่ดิน': shape
    }])
    new_encoded = pd.get_dummies(new_df)
    for col in X.columns:
        if col not in new_encoded.columns:
            new_encoded[col] = 0
    new_encoded = new_encoded[X.columns]

    pred = model.predict(new_encoded)[0]
    พท_สาธา = pred[0] * area
    พท_ขาย = pred[1] * area
    พท_สวน = pred[2] * area
    หลังรวม = pred[3] * rai

    # Normalize
    total = sum(pred[4:8]) or 1
    ทาวโฮม, บ้านแฝด, บ้านเดี่ยว, อาคารพาณิชย์ = [หลังรวม * (r / total) for r in pred[4:8]]

    # Rules
    คำเตือน = ""
    if grade in ['ESSENCE', 'PRIME', 'GRANDESSENCE', 'GRAND', 'PRIMA VILLA']:
        ทาวโฮม = บ้านแฝด = อาคารพาณิชย์ = 0
        บ้านเดี่ยว = หลังรวม
        คำเตือน = "🏡 เกรดพรีเมียม: เฉพาะบ้านเดี่ยว"
    elif grade in ['BELLA', 'PRIMO']:
        if บ้านเดี่ยว > หลังรวม * 0.2:
            บ้านเดี่ยว = หลังรวม * 0.2
        if อาคารพาณิชย์ > หลังรวม * 0.1:
            อาคารพาณิชย์ = 0
        คำเตือน = "🏘️ เกรดประหยัด: เน้นทาวน์โฮม + บ้านแฝด"

    ซอย = หลังรวม / avg_ซอยต่อหลัง.get(grade, 12)

    # ---------- OUTPUT ----------------
    st.markdown("---")
    st.subheader("📊 ผลลัพธ์การพยากรณ์")
    if คำเตือน:
        st.warning(คำเตือน)

    st.success(f"""
- พื้นที่จัดจำหน่าย: **{พท_ขาย:,.0f} ตร.ม.**
- พื้นที่สาธารณะ: **{พท_สาธา:,.0f} ตร.ม.**
- พื้นที่สวน: **{พท_สวน:,.0f} ตร.ม.**
- จำนวนหลังรวม: **{หลังรวม:.0f}**
- จำนวนซอย: **{ซอย:.0f}** (เกรด {grade})
    
🏘️ Breakdown:
- ทาวน์โฮม: {ทาวโฮม:.0f}
- บ้านแฝด: {บ้านแฝด:.0f}
- บ้านเดี่ยว: {บ้านเดี่ยว:.0f}
- อาคารพาณิชย์: {อาคารพาณิชย์:.0f}
    """)

# ---------- FOOTER ----------------
st.markdown("---")
st.caption("© 2025 Smart Layout for Real Estate Projects | by mmethaa")
