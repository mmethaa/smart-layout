import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error
import os

# -------------------- CONFIG --------------------
st.set_page_config(page_title="Smart Layout AI", page_icon="🎗️", layout="centered")

# -------------------- LOAD DATA --------------------
data_path = "data update (2).xlsx"
if not os.path.exists(data_path):
    st.error("ไม่พบไฟล์ข้อมูล 'data update (2).xlsx'")
    st.stop()

# Load dataset
st.markdown("## 📊 วิเคราะห์ความแม่นยำของโมเดล")
df = pd.read_excel(data_path)
df.columns = df.columns.str.strip()
df.fillna(0, inplace=True)

# Feature Engineering
df['หลังต่อซอย'] = df['จำนวนหลัง'] / df['จำนวนซอย'].replace(0, 1)
df['%บ้านเดี่ยว2ชั้น'] = df.get('บ้านเดี่ยว2ชั้น', 0) / df['จำนวนหลัง'].replace(0, 1)
df['%บ้านเดี่ยว3ชั้น'] = df.get('บ้านเดี่ยว3ชั้น', 0) / df['จำนวนหลัง'].replace(0, 1)
df['%บ้านแฝด'] = df.get('บ้านแฝด', 0) / df['จำนวนหลัง'].replace(0, 1)
df['%ทาวโฮม'] = df.get('ทาวโฮม', 0) / df['จำนวนหลัง'].replace(0, 1)
df['%พื้นที่ขาย'] = df.get('พื้นที่จัดจำหน่าย(ตรม)', 0) / df['พื้นที่โครงการ(ตรม)']
df['%พื้นที่สาธา'] = df.get('พื้นที่สาธา(ตรม)', 0) / df['พื้นที่โครงการ(ตรม)']

# พื้นที่ถนน / สาธารณะ
if 'พื้นที่ถนนรวม(ตร.ม.)' in df.columns and 'พื้นที่สาธา(ตรม)' in df.columns:
    ถนน_ต่อ_สาธา = df['พื้นที่ถนนรวม(ตร.ม.)'] / df['พื้นที่สาธา(ตรม)'].replace(0, 1)
    เฉลี่ย_ถนน_ต่อ_สาธา = ถนน_ต่อ_สาธา[(ถนน_ต่อ_สาธา > 0) & (ถนน_ต่อ_สาธา < 5)].mean()
else:
    เฉลี่ย_ถนน_ต่อ_สาธา = 0.65

df['%พื้นที่ถนน'] = df['%พื้นที่สาธา'] * เฉลี่ย_ถนน_ต่อ_สาธา

# Model inputs
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

# Encoding
X = pd.get_dummies(X_raw, columns=['จังหวัด', 'เกรดโครงการ', 'รูปร่างที่ดิน'])
X_train, X_test, y_train, y_test = train_test_split(X, y_ratio.fillna(0), test_size=0.2, random_state=42)

# Train model
model = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42))
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluate accuracy
accuracy_table = pd.DataFrame(columns=['พารามิเตอร์', 'Accuracy (%)'])
for i, col in enumerate(y_ratio.columns):
    mape = mean_absolute_percentage_error(y_test.iloc[:, i], y_pred[:, i])
    acc = 100 - (mape * 100)
    accuracy_table.loc[len(accuracy_table)] = [col, round(acc, 2)]

# Show table
st.dataframe(accuracy_table, use_container_width=True)
st.caption("*ความแม่นยำเทียบกับข้อมูลโครงการจริงที่ถือเป็น Best Practice 100%")
