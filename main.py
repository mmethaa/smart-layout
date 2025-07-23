import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(page_title="Smart Layout AI", page_icon="🏗️", layout="centered")

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

# โหลดข้อมูล
sheet_name = "Sheet1 (2)"
df = pd.read_excel("layoutdata.xlsx", sheet_name=sheet_name)
df.columns = df.columns.str.strip()

# ตรวจสอบและแทนค่าขาดหาย
for col in ['ความกว้าง(ทาวโฮม)', 'ความยาว(ทาวโฮม)', 'ความกว้าง(บ้านแฝด)', 'ความยาว(บ้านแฝด)', 'ความกว้าง(บ้านเดี่ยว)', 'ความยาว(บ้านเดี่ยว)']:
    df[col] = df[col].fillna(0)

# Feature Engineering
for t in ['ทาวโฮม', 'บ้านแฝด', 'บ้านเดี่ยว']:
    df[f'พื้นที่เฉลี่ย({t})'] = df[f'ความกว้าง({t})'] * df[f'ความยาว({t})']

# Target ratios
df['หลังต่อซอย'] = df['จำนวนหลัง'] / df['จำนวนซอย'].replace(0, 1)
df['%บ้านเดี่ยว'] = df['บ้านเดี่ยว'] / df['จำนวนหลัง'].replace(0, 1)
df['%บ้านแฝด'] = df['บ้านแฝด'] / df['จำนวนหลัง'].replace(0, 1)
df['%ทาวโฮม'] = df['ทาวโฮม'] / df['จำนวนหลัง'].replace(0, 1)
df['%พื้นที่ขาย'] = df['พื้นที่จัดจำหน่าย(ตรม)'] / df['พื้นที่โครงการ(ตรม)']
df['%พื้นที่สาธา'] = df['พื้นที่สาธา(ตรม)'] / df['พื้นที่โครงการ(ตรม)']
df['%พื้นที่สวน'] = df['พื้นที่สวน(5%ของพื้นที่จัดจำหน่าย)'] / df['พื้นที่โครงการ(ตรม)']
df['%ถนนในสาธารณะ'] = (df['พื้นที่ถนนรวม'] / df['พื้นที่สาธา(ตรม)']).fillna(0)

ถนน_ต่อ_สาธารณะ_เฉลี่ย = df['%ถนนในสาธารณะ'].mean()

# Feature set
X_raw = df[[
    'จังหวัด', 'เกรดโครงการ', 'พื้นที่โครงการ(ตรม)', 'รูปร่างที่ดิน',
    'ความยาวถนน', 'ความกว้างถนนปกติ',
    'พื้นที่เฉลี่ย(ทาวโฮม)', 'พื้นที่เฉลี่ย(บ้านแฝด)', 'พื้นที่เฉลี่ย(บ้านเดี่ยว)']]

# Targets
y_ratio = pd.DataFrame({
    'สัดส่วนพื้นที่สาธา': df['%พื้นที่สาธา'],
    'สัดส่วนพื้นที่จัดจำหน่าย': df['%พื้นที่ขาย'],
    'สัดส่วนพื้นที่สวน': df['%พื้นที่สวน'],
    'จำนวนหลังต่อไร่': df['จำนวนหลัง'] / (df['พื้นที่โครงการ(ตรม)'] / 1600),
    'สัดส่วนทาวโฮม': df['%ทาวโฮม'],
    'สัดส่วนบ้านแฝด': df['%บ้านแฝด'],
    'สัดส่วนบ้านเดี่ยว': df['%บ้านเดี่ยว'],
    'สัดส่วนอาคารพาณิชย์': df['อาคารพาณิชย์'] / df['จำนวนหลัง'].replace(0, 1)
})

# Encoding
X = pd.get_dummies(X_raw, columns=['จังหวัด', 'เกรดโครงการ', 'รูปร่างที่ดิน'])
X_train, _, y_train, _ = train_test_split(X, y_ratio, test_size=0.2, random_state=42)

# Model Training
model = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42)).fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_train)
mae = mean_absolute_error(y_train, y_pred)
r2 = r2_score(y_train, y_pred)

st.markdown("## 📈 ความแม่นยำของโมเดล")
st.write(f"**MAE (Mean Absolute Error):** {mae:.4f}")
st.write(f"**R² Score:** {r2:.4f}")
