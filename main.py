import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split

# -------------------- CONFIG --------------------
st.set_page_config(page_title="Smart Layout AI", page_icon="🏗️", layout="centered")

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
df = pd.read_excel("updatedata.xlsx")
df.columns = df.columns.str.strip()
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
    st.success("พยากรณ์เสร็จสิ้น กรุณาเลื่อนลงเพื่อดูผลการวิเคราะห์ความคาดเคลื่อน")

# -------------------- ERROR ANALYSIS SECTION --------------------
st.markdown("## 📉 วิเคราะเครื่องความคลาดเคลื่อนของโมเดล")

# เตรียมข้อมูลคำนวณ
calc_df = df.copy()
calc_df['ไร่'] = calc_df['พื้นที่โครงการ(ตรม)'] / 1600
calc_df['ตรว'] = calc_df['พื้นที่โครงการ(ตรม)'] / 4
X_input = pd.get_dummies(calc_df[['จังหวัด', 'เกรดโครงการ', 'พื้นที่โครงการ(ตรม)', 'รูปร่างที่ดิน']])
for col in X.columns:
    if col not in X_input.columns:
        X_input[col] = 0
X_input = X_input[X.columns]

predicted_df = pd.DataFrame(model.predict(X_input), columns=y_ratio.columns)

compare_df = pd.DataFrame()
compare_df['โครงการ'] = df['โครงการ']
compare_df['พื้นที่ขาย (จริง)(ตรว)'] = calc_df['พื้นที่จัดจำหน่าย(ตรม)'] / 4
compare_df['พื้นที่ขาย (predict)(ตรว)'] = predicted_df['สัดส่วนพื้นที่จัดจำหน่าย'] * calc_df['ตรว']
compare_df['%Error_พื้นที่ขาย'] = (compare_df['พื้นที่ขาย (predict)(ตรว)'] - compare_df['พื้นที่ขาย (จริง)(ตรว)']).abs() / compare_df['พื้นที่ขาย (จริง)(ตรว)'] * 100

compare_df['พื้นที่สาธารณะ (จริง)(ตรว)'] = calc_df['พื้นที่สาธา(ตรม)'] / 4
compare_df['พื้นที่สาธารณะ (predict)(ตรว)'] = predicted_df['สัดส่วนพื้นที่สาธา'] * calc_df['ตรว']
compare_df['%Error_พื้นที่สาธา'] = (compare_df['พื้นที่สาธารณะ (predict)(ตรว)'] - compare_df['พื้นที่สาธารณะ (จริง)(ตรว)']).abs() / compare_df['พื้นที่สาธารณะ (จริง)(ตรว)'] * 100

compare_df['จำนวนแปลงรวม (จริง)'] = calc_df['จำนวนหลัง']
compare_df['จำนวนแปลงรวม (predict)'] = predicted_df['จำนวนหลังต่อไร่'] * calc_df['ไร่']
compare_df['%Error_จำนวนแปลง'] = (compare_df['จำนวนแปลงรวม (predict)'] - compare_df['จำนวนแปลงรวม (จริง)']).abs() / compare_df['จำนวนแปลงรวม (จริง)'].replace(0, 1) * 100

compare_df['จำนวนซอย (จริง)'] = calc_df['จำนวนซอย']
compare_df['จำนวนซอย (predict)'] = compare_df['จำนวนแปลงรวม (predict)'] / calc_df['หลังต่อซอย'].replace(0, 1)
compare_df['%Error_จำนวนซอย'] = (compare_df['จำนวนซอย (predict)'] - compare_df['จำนวนซอย (จริง)']).abs() / compare_df['จำนวนซอย (จริง)'].replace(0, 1) * 100

st.dataframe(compare_df.round(2))

st.caption("Developed by mmethaa | Smart Layout AI 🚀")
