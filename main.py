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
    y_true = y_test.iloc[:, i]
    y_pred_col = y_pred[:, i]

    mask = y_true != 0
    if mask.sum() == 0:
        acc = None
    else:
        mape = mean_absolute_percentage_error(y_true[mask], y_pred_col[mask])
        acc = 100 - (mape * 100)

    accuracy_table.loc[len(accuracy_table)] = [col, round(acc, 2) if acc is not None else "N/A"]

st.dataframe(accuracy_table, use_container_width=True)
st.caption("*ความแม่นยำเทียบกับข้อมูลโครงการจริงที่ถือเป็น Best Practice 100%")

# -------------------- USER INPUT FOR PREDICTION --------------------
st.markdown("---")
st.markdown("## ✏️ พยากรณ์จากข้อมูลของคุณ")

col1, col2 = st.columns(2)
จังหวัด_input = col1.selectbox("จังหวัด", sorted(df['จังหวัด'].unique()))
เกรด_input = col1.selectbox("เกรดโครงการ", sorted(df['เกรดโครงการ'].unique()))
รูปร่าง_input = col2.selectbox("รูปร่างที่ดิน", sorted(df['รูปร่างที่ดิน'].unique()))
พื้นที่_input = col2.number_input("พื้นที่โครงการ (ตรม)", min_value=1, value=10000)

if st.button("🔍 พยากรณ์ผลลัพธ์"):
    input_df = pd.DataFrame.from_dict({
        'จังหวัด': [จังหวัด_input],
        'เกรดโครงการ': [เกรด_input],
        'พื้นที่โครงการ(ตรม)': [พื้นที่_input],
        'รูปร่างที่ดิน': [รูปร่าง_input]
    })
    input_encoded = pd.get_dummies(input_df)

    for col in X.columns:
        if col not in input_encoded.columns:
            input_encoded[col] = 0
    input_encoded = input_encoded[X.columns]

    y_out = model.predict(input_encoded)[0]
    results = pd.Series(y_out, index=y_ratio.columns)

    # คำนวณค่าจากสัดส่วนจริง
    พท_สาธา = round(results['สัดส่วนพื้นที่สาธา'] * พื้นที่_input / 4, 0)
    พท_ขาย = round(results['สัดส่วนพื้นที่จัดจำหน่าย'] * พื้นที่_input / 4, 0)
    พท_สวน = round(results['สัดส่วนพื้นที่สวน'] * พื้นที่_input / 4, 0)
    พท_ถนน = round(results['สัดส่วนพื้นที่ถนน'] * พื้นที่_input / 4, 0)
    จำนวน_หลัง = round(results['จำนวนหลังต่อไร่'] * (พื้นที่_input / 1600), 0)

    st.markdown("### ✅ ผลลัพธ์ที่คาดการณ์ (พื้นที่และจำนวนแปลง)")
    output_summary = pd.DataFrame({
        "พารามิเตอร์": ["พื้นที่สาธารณะ", "พื้นที่จัดจำหน่าย", "พื้นที่สวน", "พื้นที่ถนน", "จำนวนแปลง"],
        "ค่าที่คาดการณ์ได้": [f"{พท_สาธา:,} ตรว.", f"{พท_ขาย:,} ตรว.", f"{พท_สวน:,} ตรว.", f"{พท_ถนน:,} ตรว.", f"{int(จำนวน_หลัง):,} หลัง"]
    })
    st.dataframe(output_summary, use_container_width=True)

    # คำนวณจำนวนหลังแยกตามประเภท
    th_units = round(results['สัดส่วนทาวโฮม'] * จำนวน_หลัง)
    twin_units = round(results['สัดส่วนบ้านแฝด'] * จำนวน_หลัง)
    sd2_units = round(results['สัดส่วนบ้านเดี่ยว2ชั้น'] * จำนวน_หลัง)
    sd3_units = round(results['สัดส่วนบ้านเดี่ยว3ชั้น'] * จำนวน_หลัง)

    st.markdown("### 🏡 แยกตามประเภทบ้าน")
    st.markdown(f"- ทาวน์โฮม: **{th_units:,} หลัง**")
    st.markdown(f"- บ้านแฝด: **{twin_units:,} หลัง**")
    st.markdown(f"- บ้านเดี่ยว 2 ชั้น: **{sd2_units:,} หลัง**")
    st.markdown(f"- บ้านเดี่ยว 3 ชั้น: **{sd3_units:,} หลัง**")

    st.markdown("### 📐 วิเคราะห์ % ความแม่นยำจากโมเดล")
    st.dataframe(accuracy_table, use_container_width=True)
    st.caption("*เทียบกับค่าที่คาดว่าเป็น Best Practice")
