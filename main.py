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

df = pd.read_excel("layoutdata.xlsx")
df.columns = df.columns.str.strip()
df['หลังต่อซอย'] = df['จำนวนหลัง'] / df['จำนวนซอย'].replace(0, 1)
df['%บ้านเดี่ยว'] = df['บ้านเดี่ยว'] / df['จำนวนหลัง'].replace(0, 1)
df['%บ้านแฝด'] = df['บ้านแฝด'] / df['จำนวนหลัง'].replace(0, 1)
df['%ทาวโฮม'] = df['ทาวโฮม'] / df['จำนวนหลัง'].replace(0, 1)
df['%พื้นที่ขาย'] = df['พื้นที่จัดจำหน่าย(ตรม)'] / df['พื้นที่โครงการ(ตรม)']
df['%พื้นที่สาธา'] = df['พื้นที่สาธา(ตรม)'] / df['พื้นที่โครงการ(ตรม)']
# คำนวณสัดส่วนถนนในอดีตจากข้อมูลจริง
df['พื้นที่ถนน(ตรม)'] = df['พื้นที่สาธา(ตรม)'] - df['พื้นที่สวน(5%ของพื้นที่จัดจำหน่าย)']
df['%ถนนในสาธารณะ'] = df['พื้นที่ถนน(ตรม)'] / df['พื้นที่สาธา(ตรม)']
ถนน_ต่อ_สาธารณะ_เฉลี่ย = df['%ถนนในสาธารณะ'].mean()


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
X_train, X_test, y_train, y_test = train_test_split(X, y_ratio, test_size=0.2, random_state=42)
model = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42)).fit(X_train, y_train)
avg_ซอยต่อหลัง = df.groupby('เกรดโครงการ')['หลังต่อซอย'].mean().to_dict()

# กลุ่มพื้นที่
df_group = df[['เกรดโครงการ', 'พื้นที่โครงการ(ตรม)', 'ทาวโฮม', 'บ้านแฝด', 'บ้านเดี่ยว', 'อาคารพาณิชย์', 'จำนวนหลัง']].copy()
df_group['%ทาวโฮม'] = df_group['ทาวโฮม'] / df_group['จำนวนหลัง'].replace(0, 1)
df_group['%บ้านแฝด'] = df_group['บ้านแฝด'] / df_group['จำนวนหลัง'].replace(0, 1)
df_group['%บ้านเดี่ยว'] = df_group['บ้านเดี่ยว'] / df_group['จำนวนหลัง'].replace(0, 1)
df_group['%อาคารพาณิชย์'] = df_group['อาคารพาณิชย์'] / df_group['จำนวนหลัง'].replace(0, 1)
bins = [0, 20000, 40000, 60000, 80000, 100000, float("inf")]
labels = ["≤20k", "20k-40k", "40k-60k", "60k-80k", "80k-100k", "100k+"]
df_group['กลุ่มพื้นที่'] = pd.cut(df_group['พื้นที่โครงการ(ตรม)'], bins=bins, labels=labels)
grouped_ratio = df_group.groupby(['เกรดโครงการ', 'กลุ่มพื้นที่'], observed=True)[["%ทาวโฮม", "%บ้านแฝด", "%บ้านเดี่ยว", "%อาคารพาณิชย์"]].mean().round(3)
grouped_ratio_dict = grouped_ratio.to_dict(orient="index")

def adjust_by_grade_policy(grade, ratios):
    # Existing policy for PRIMO, BELLA, WATTANALAI
    if grade in ['PRIMO', 'BELLA', 'WATTANALAI']:
        ratios[2] = min(ratios[2], 0.2) # Max 20% detached (บ้านเดี่ยว)
        remain = 1 - ratios[2] - ratios[3] # Recalculate remaining for townhome/semi-detached
        ratios[0] = remain * 0.65 # ทาวโฮม (Index 0)
        ratios[1] = remain * 0.35 # บ้านแฝด (Index 1)

    # NEW POLICY: For grades that should only have detached houses (townhome, semi-detached, commercial are 0)
    # Based on data analysis, grades like MONTARA, PRIMAVILLA, and some PARKVILLE instances
    # tend to be exclusively detached. This enforces that policy.
    # 'PARKVILLE' and 'PARK VILLE' are treated as the same grade for this policy.
    if grade in ['MONTARA', 'PRIMAVILLA', 'PARKVILLE', 'PARK VILLE']:
        ratios[0] = 0.0  # Townhome
        ratios[1] = 0.0  # Semi-detached
        ratios[3] = 0.0  # Commercial
        ratios[2] = 1.0  # Detached - set to 100% of remaining plots if other types are 0

    return ratios

# ====== FORM ======
st.markdown("## 📋 กรอกข้อมูลโครงการ")
with st.form("input_form"):
    col1, col2 = st.columns(2)
    with col1:
        จังหวัด = st.selectbox("📍 จังหวัด", sorted(df['จังหวัด'].dropna().unique()))
        รูปร่าง = st.selectbox("🧱️ รูปร่างที่ดิน", sorted(df['รูปร่างที่ดิน'].dropna().unique()))
    with col2:
        เกรด = st.selectbox("🏧 เกรดโครงการ", sorted(df['เกรดโครงการ'].dropna().unique()))
        พื้นที่_วา = st.number_input("📀 พื้นที่โครงการ (ตารางวา)", min_value=250, value=7500, step=100)
    submitted = st.form_submit_button("🚀 เริ่มพยากรณ์")

# ====== PREDICT ======
if submitted:
    area = พื้นที่_วา * 4
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
    พท_ถนน = พท_สาธา * ถนน_ต่อ_สาธารณะ_เฉลี่ย
    หลังรวม = pred[3] * rai

    ratio_hist = get_ratio_from_lookup(เกรด, area)
    # แก้ไข: ตรวจสอบอย่างชัดเจนว่า ratio_hist ไม่ใช่ None ก่อนนำไปใช้งาน
    if ratio_hist is not None:
        ทาวโฮม, บ้านแฝด, บ้านเดี่ยว, อาคารพาณิชย์ = [หลังรวม * r for r in ratio_hist]
    else:
        # หากไม่พบข้อมูลใน lookup table หรือมีปัญหาในการดึงข้อมูล ให้ใช้ค่าจากโมเดลโดยตรง
        # และปรับตาม policy ที่กำหนดไว้
        total = sum(pred[4:8]) or 1
        raw_ratios = [r / total for r in pred[4:8]]
        raw_ratios = adjust_by_grade_policy(เกรด, raw_ratios)
        ทาวโฮม, บ้านแฝด, บ้านเดี่ยว, อาคารพาณิชย์ = [หลังรวม * r for r in raw_ratios]

    ซอย = หลังรวม / avg_ซอยต่อหลัง.get(เกรด, 12)

    st.markdown("---")
    st.markdown("## 🌟 ผลลัพธ์พยากรณ์")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("พื้นที่สาธารณะ", f"{พท_สาธา / 4:,.0f} ตร.วา")
        st.metric("พื้นที่จัดจำหน่าย", f"{พท_ขาย / 4:,.0f} ตร.วา")
        st.metric("พื้นที่สวน", f"{พท_สวน / 4:,.0f} ตร.วา")
        st.metric("พื้นที่ถนน", f"{พท_ถนน / 4:,.0f} ตร.วา")
    with col2:
        st.metric("จำนวนแปลงรวม", f"{หลังรวม:,.0f} หลัง")
        st.metric("จำนวนซอย", f"{ซอย:,.0f} ซอย")

    st.markdown("### 🏡 แยกตามประเภทบ้าน")
    st.markdown(f"""
        - ทาวน์โฮม: **{ทาวโฮม:,.0f}** หลัง  
        - บ้านแฝด: **{บ้านแฝด:,.0f}** หลัง  
        - บ้านเดี่ยว: **{บ้านเดี่ยว:,.0f}** หลัง  
        - อาคารพาณิชย์: **{อาคารพาณิชย์:,.0f}** หลัง  
    """)

    y_pred_test = model.predict(X_test)
    mae_test = mean_absolute_error(y_test, y_pred_test)
    r2_test = r2_score(y_test, y_pred_test)

    st.markdown("### 📈 ความแม่นยำของโมเดล (Test Set)")
    st.write(f"**MAE (Mean Absolute Error):** {mae_test:.4f}")
    st.write(f"**R² Score:** {r2_test:.4f}")

st.markdown("---")
st.caption("Developed by mmethaa | Smart Layout AI 🚀")
