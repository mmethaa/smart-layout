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
# -------------------- เตรียมข้อมูล X, y สำหรับ ML --------------------
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

# -------------------- grouped historical ratios --------------------
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

# -------------------- ฟังก์ชันช่วย --------------------
def adjust_by_grade_policy(grade, ratios):
    if grade in ['PRIMO', 'BELLA', 'WATTANALAI']:
        ratios[2] = min(ratios[2], 0.2)  # บ้านเดี่ยวไม่เกิน 20%
        remain = 1 - ratios[2] - ratios[3]
        ratios[0] = remain * 0.65
        ratios[1] = remain * 0.35
    return ratios

def get_ratio_from_lookup(grade, area):
    group = labels[-1]
    for i, b in enumerate(bins[:-1]):
        if b < area <= bins[i+1]:
            group = labels[i]
            break
    ratio = grouped_ratio_dict.get((grade, group))
    if ratio and any(pd.notna(list(ratio.values()))):
        total = sum([v for v in ratio.values() if pd.notna(v)]) or 1
        ratios = [ratio.get('%ทาวโฮม', 0)/total,
                  ratio.get('%บ้านแฝด', 0)/total,
                  ratio.get('%บ้านเดี่ยว', 0)/total,
                  ratio.get('%อาคารพาณิชย์', 0)/total]
        return adjust_by_grade_policy(grade, ratios)
    return None
# -------------------- FORM --------------------
st.markdown("## 📋 กรอกข้อมูลโครงการ")
with st.form("input_form"):
    col1, col2 = st.columns(2)
    with col1:
        จังหวัด = st.selectbox("📍 จังหวัด", sorted(df['จังหวัด'].dropna().unique()))
        รูปร่าง = st.selectbox("🧱 รูปร่างที่ดิน", sorted(df['รูปร่างที่ดิน'].dropna().unique()))
    with col2:
        เกรด = st.selectbox("🏧 เกรดโครงการ", sorted(df['เกรดโครงการ'].dropna().unique()))
        พื้นที่ = st.number_input("📀 พื้นที่โครงการ (ตร.ม.)", min_value=1000, value=30000, step=500)
    submitted = st.form_submit_button("🚀 เริ่มพยากรณ์")

if submitted:
    input_df = pd.DataFrame([{ 'จังหวัด': จังหวัด, 'เกรดโครงการ': เกรด, 'พื้นที่โครงการ(ตรม)': พื้นที่, 'รูปร่างที่ดิน': รูปร่าง }])
    encoded = pd.get_dummies(input_df)
    for col in X.columns:
        if col not in encoded.columns:
            encoded[col] = 0
    encoded = encoded[X.columns]
    pred = model.predict(encoded)[0]

    พท_ขาย = pred[1] * (พื้นที่ / 4)
    พท_สาธา = pred[0] * (พื้นที่ / 4)
    พท_สวน = pred[2] * (พื้นที่ / 4)
    พท_ถนน = พท_สาธา - พท_สวน
    หลังรวม = pred[3] * (พื้นที่ / 1600)
    ซอย = หลังรวม / avg_ซอยต่อหลัง.get(เกรด, 12)

    # --- ใช้สัดส่วนจาก grouped_ratio_dict ---
    ratio = get_ratio_from_lookup(เกรด, พื้นที่)
    if not ratio:
        ratio = pred[4:8]
    ทาวโฮม, บ้านแฝด, บ้านเดี่ยว, อาคารพาณิชย์ = [หลังรวม * r for r in ratio]

    # ปรับบ้านเดี่ยว 3 ชั้นไม่ให้เกิดในเกรดที่ไม่ควรมี
    if เกรด not in ["ELEGANCE", "GRANDESSENCE", "ESSENCE"]:
        บ้านเดี่ยว = min(บ้านเดี่ยว, หลังรวม - ทาวโฮม - บ้านแฝด - อาคารพาณิชย์)
 # -------------------- เปรียบเทียบกับค่าจริงใน dataset --------------------
    match_row = df[(df['จังหวัด'] == จังหวัด) & (df['เกรดโครงการ'] == เกรด) & (df['รูปร่างที่ดิน'] == รูปร่าง)]
    if not match_row.empty:
        row = match_row.iloc[0]
        พท_ขาย_จริง = row['พื้นที่จัดจำหน่าย(ตรม)'] / 4
        พท_สาธา_จริง = row['พื้นที่สาธา(ตรม)'] / 4
        พท_สวน_จริง = row['พื้นที่สวน(5%ของพื้นที่จัดจำหน่าย)'] / 4
        พท_ถนน_จริง = พท_สาธา_จริง - พท_สวน_จริง
        หลัง_จริง = row['จำนวนหลัง']
        ซอย_จริง = row['จำนวนซอย']
        err = lambda p, t: abs(p - t) / t * 100 if t else 0
        err_ขาย = err(พท_ขาย, พท_ขาย_จริง)
        err_สาธา = err(พท_สาธา, พท_สาธา_จริง)
        err_สวน = err(พท_สวน, พท_สวน_จริง)
        err_ถนน = err(พท_ถนน, พท_ถนน_จริง)
        err_แปลง = err(หลังรวม, หลัง_จริง)
        err_ซอย = err(ซอย, ซอย_จริง)
    else:
        err_ขาย = err_สาธา = err_สวน = err_ถนน = err_แปลง = err_ซอย = 0

    # -------------------- DISPLAY --------------------
    st.markdown("---")
    st.markdown("## 🌟 ผลลัพธ์พยากรณ์")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("พื้นที่จัดจำหน่าย (ตรว)", f"{พท_ขาย:,.0f}", help=f"%Error: {err_ขาย:.2f}%")
        st.metric("พื้นที่สวน (ตรว)", f"{พท_สวน:,.0f}", help=f"%Error: {err_สวน:.2f}%")
    with col2:
        st.metric("พื้นที่สาธารณะ (ตรว)", f"{พท_สาธา:,.0f}", help=f"%Error: {err_สาธา:.2f}%")
        st.metric("พื้นที่ถนน (ตรว)", f"{พท_ถนน:,.0f}", help=f"%Error: {err_ถนน:.2f}%")
    with col3:
        st.metric("จำนวนแปลงรวม", f"{หลังรวม:,.0f}", help=f"%Error: {err_แปลง:.2f}%")
        st.metric("จำนวนซอย", f"{ซอย:,.0f}", help=f"%Error: {err_ซอย:.2f}%")
    # -------------------- DISPLAY บ้านแต่ละประเภท --------------------
    st.markdown("### 🏡 แยกจำนวนแปลงตามประเภทบ้าน")
    st.markdown(f"""
    - ทาวน์โฮม: **{ทาวโฮม:,.0f}** หลัง  
    - บ้านแฝด: **{บ้านแฝด:,.0f}** หลัง  
    - บ้านเดี่ยว: **{บ้านเดี่ยว:,.0f}** หลัง  
    - อาคารพาณิชย์: **{อาคารพาณิชย์:,.0f}** หลัง
    """)

# -------------------- FOOTER --------------------
st.caption("Developed by mmethaa | Smart Layout AI 🚀")
