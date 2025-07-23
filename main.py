import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import numpy as np # เพิ่มการนำเข้า numpy

st.set_page_config(page_title="Smart Layout AI", page_icon="🏗️", layout="centered")

st.markdown("""
    <style>
    html, body, [class*="css"] {
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

# โหลดข้อมูลจากไฟล์ Excel
sheet_name = "Sheet1 (2)" # ตรวจสอบให้แน่ใจว่าชื่อชีทถูกต้อง
try:
    df = pd.read_excel("layoutdata.xlsx", sheet_name=sheet_name)
except FileNotFoundError:
    st.error("ไม่พบไฟล์ 'layoutdata.xlsx' กรุณาตรวจสอบว่าไฟล์อยู่ในไดเรกทอรีเดียวกันกับสคริปต์")
    st.stop() # หยุดการทำงานของ Streamlit หากไม่พบไฟล์

df.columns = df.columns.str.strip() # ลบช่องว่างหน้า/หลังชื่อคอลัมน์

# จัดการค่าว่าง (NaN) สำหรับคอลัมน์ความกว้างและความยาว
for col in ['ความกว้าง(ทาวโฮม)', 'ความยาว(ทาวโฮม)', 'ความกว้าง(บ้านแฝด)', 'ความยาว(บ้านแฝด)', 'ความกว้าง(บ้านเดี่ยว)', 'ความยาว(บ้านเดี่ยว)']:
    df[col] = df[col].fillna(0) # เติมค่าว่างด้วย 0

# สร้างฟีเจอร์ใหม่: พื้นที่เฉลี่ยของบ้านแต่ละประเภท
for t in ['ทาวโฮม', 'บ้านแฝด', 'บ้านเดี่ยว']:
    df[f'พื้นที่เฉลี่ย({t})'] = df[f'ความกว้าง({t})'] * df[f'ความยาว({t})']

# สร้าง Target ratios (สัดส่วนเป้าหมายที่โมเดลจะทำนาย)
# ใช้ np.nan แทน 1 เพื่อให้ dropna() สามารถลบแถวที่มีปัญหาออกได้
df['หลังต่อซอย'] = df['จำนวนหลัง'] / df['จำนวนซอย'].replace(0, np.nan)
df['%บ้านเดี่ยว'] = df['บ้านเดี่ยว'] / df['จำนวนหลัง'].replace(0, np.nan)
df['%บ้านแฝด'] = df['บ้านแฝด'] / df['จำนวนหลัง'].replace(0, np.nan)
df['%ทาวโฮม'] = df['ทาวโฮม'] / df['จำนวนหลัง'].replace(0, np.nan)
df['%พื้นที่ขาย'] = df['พื้นที่จัดจำหน่าย(ตรม)'] / df['พื้นที่โครงการ(ตรม)'].replace(0, np.nan)
df['%พื้นที่สาธา'] = df['พื้นที่สาธา(ตรม)'] / df['พื้นที่โครงการ(ตรม)'].replace(0, np.nan)

# ตรวจสอบว่าคอลัมน์ 'พื้นที่สวน(5%ของพื้นที่จัดจำหน่าย)' มีอยู่จริงหรือไม่
if 'พื้นที่สวน(5%ของพื้นที่จัดจำหน่าย)' in df.columns:
    df['%พื้นที่สวน'] = df['พื้นที่สวน(5%ของพื้นที่จัดจำหน่าย)'] / df['พื้นที่โครงการ(ตรม)'].replace(0, np.nan)
else:
    st.warning("ไม่พบคอลัมน์ 'พื้นที่สวน(5%ของพื้นที่จัดจำหน่าย)' ในข้อมูล อาจส่งผลต่อความแม่นยำของ %พื้นที่สวน")
    df['%พื้นที่สวน'] = 0.05 # กำหนดค่าเริ่มต้นหากไม่พบ
    
df['%ถนนในสาธารณะ'] = (df['พื้นที่ถนนรวม'] / df['พื้นที่สาธา(ตรม)'].replace(0, np.nan)).fillna(0)

# คำนวณค่าเฉลี่ยของสัดส่วนถนนต่อพื้นที่สาธารณะ
ถนน_ต่อ_สาธารณะ_เฉลี่ย = df['%ถนนในสาธารณะ'].mean()

# กำหนด Features (X_raw) และ Targets (y_ratio) สำหรับการฝึกโมเดล
X_raw = df[[
    'จังหวัด', 'เกรดโครงการ', 'พื้นที่โครงการ(ตรม)', 'รูปร่างที่ดิน',
    'ความยาวถนน', 'ความกว้างถนนปกติ',
    'พื้นที่เฉลี่ย(ทาวโฮม)', 'พื้นที่เฉลี่ย(บ้านแฝด)', 'พื้นที่เฉลี่ย(บ้านเดี่ยว)']]

y_ratio = pd.DataFrame({
    'สัดส่วนพื้นที่สาธา': df['%พื้นที่สาธา'],
    'สัดส่วนพื้นที่จัดจำหน่าย': df['%พื้นที่ขาย'],
    'สัดส่วนพื้นที่สวน': df['%พื้นที่สวน'],
    'จำนวนหลังต่อไร่': df['จำนวนหลัง'] / (df['พื้นที่โครงการ(ตรม)'] / 1600).replace(0, np.nan), # ป้องกันการหารด้วย 0
    'สัดส่วนทาวโฮม': df['%ทาวโฮม'],
    'สัดส่วนบ้านแฝด': df['%บ้านแฝด'],
    'สัดส่วนบ้านเดี่ยว': df['%บ้านเดี่ยว'],
    'สัดส่วนอาคารพาณิชย์': df['อาคารพาณิชย์'] / df['จำนวนหลัง'].replace(0, np.nan) # ป้องกันการหารด้วย 0
})

# รวม X_raw และ y_ratio เข้าด้วยกันเพื่อทำความสะอาดข้อมูลพร้อมกัน
combined_df = pd.concat([X_raw, y_ratio], axis=1)

# แทนที่ค่า inf/-inf ด้วย NaN ก่อนที่จะลบแถว
combined_df.replace([np.inf, -np.inf], np.nan, inplace=True)

# ลบแถวที่มีค่า NaN ออกทั้งหมดจาก DataFrame ที่รวมกัน
original_rows = len(combined_df)
combined_df.dropna(inplace=True)
rows_after_cleaning = len(combined_df)

if original_rows > rows_after_cleaning:
    st.warning(f"ลบ {original_rows - rows_after_cleaning} แถวที่มีค่าว่างหรือค่าผิดปกติออกจากการฝึกโมเดล เพื่อเพิ่มความแม่นยำ")
st.info(f"ใช้ข้อมูล {rows_after_cleaning} แถวในการฝึกโมเดล")

# แยก X และ y ออกจาก DataFrame ที่ทำความสะอาดแล้ว
X_cleaned = combined_df[X_raw.columns]
y_cleaned = combined_df[y_ratio.columns]

# แปลงข้อมูลหมวดหมู่ใน X_cleaned ให้เป็นตัวเลขด้วย One-Hot Encoding
X_encoded = pd.get_dummies(X_cleaned, columns=['จังหวัด', 'เกรดโครงการ', 'รูปร่างที่ดิน'])

# แบ่งข้อมูลเป็นชุดฝึก (Train) และชุดทดสอบ (Test) เพื่อประเมินประสิทธิภาพที่แท้จริงของโมเดล
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_cleaned, test_size=0.2, random_state=42)

# สร้างและฝึกโมเดล MultiOutputRegressor ด้วย RandomForestRegressor
# เพิ่ม n_estimators เป็น 200 และ min_samples_leaf เป็น 5 เพื่อเพิ่มความแม่นยำและลด Overfitting
model = MultiOutputRegressor(RandomForestRegressor(n_estimators=200, min_samples_leaf=5, random_state=42)).fit(X_train, y_train)

st.markdown("## 📈 ความแม่นยำของโมเดล")
# ประเมินความแม่นยำบนชุดข้อมูลทดสอบ (Test Set) เพื่อให้ได้ค่าที่สะท้อนประสิทธิภาพจริง
y_pred_test = model.predict(X_test)
mae_test = mean_absolute_error(y_test, y_pred_test)
r2_test = r2_score(y_test, y_pred_test)
st.write(f"**MAE (Mean Absolute Error) บน Test Set:** {mae_test:.4f}")
st.write(f"**R² Score บน Test Set:** {r2_test:.4f}")
st.info("ค่า MAE และ R² ที่แสดงนี้มาจากข้อมูลที่โมเดลไม่เคยเห็นตอนฝึก (Test Set) ซึ่งสะท้อนความแม่นยำที่แท้จริงของโมเดลได้ดีกว่า")

st.markdown("## 📋 กรอกข้อมูลโครงการ")
with st.form("predict_form"):
    col1, col2 = st.columns(2)
    with col1:
        # ตรวจสอบว่ามีค่า unique ในคอลัมน์หรือไม่ก่อนสร้าง selectbox
        จังหวัด_options = sorted(df['จังหวัด'].dropna().unique().tolist())
        if not จังหวัด_options:
            st.error("ไม่พบข้อมูลจังหวัดที่ใช้ได้ในไฟล์ Excel")
            st.stop()
        จังหวัด = st.selectbox("📍 จังหวัด", จังหวัด_options)

        รูปร่าง_options = sorted(df['รูปร่างที่ดิน'].dropna().unique().tolist())
        if not รูปร่าง_options:
            st.error("ไม่พบข้อมูลรูปร่างที่ดินที่ใช้ได้ในไฟล์ Excel")
            st.stop()
        รูปร่าง = st.selectbox("🧱️ รูปร่างที่ดิน", รูปร่าง_options)
    with col2:
        เกรด_options = sorted(df['เกรดโครงการ'].dropna().unique().tolist())
        if not เกรด_options:
            st.error("ไม่พบข้อมูลเกรดโครงการที่ใช้ได้ในไฟล์ Excel")
            st.stop()
        เกรด = st.selectbox("🏧 เกรดโครงการ", เกรด_options)
        พื้นที่_วา = st.number_input("📀 พื้นที่โครงการ (ตารางวา)", min_value=250, value=7500, step=100)
    submitted = st.form_submit_button("🚀 เริ่มพยากรณ์")

# คำนวณค่าเฉลี่ยสำหรับ lookup functions (ใช้จาก df เดิมที่อาจมี NaN ในบางคอลัมน์ แต่จะถูกจัดการในฟังก์ชัน)
avg_ซอยต่อหลัง = df.groupby('เกรดโครงการ')['หลังต่อซอย'].mean().to_dict()

# กลุ่มพื้นที่สำหรับ get_ratio_from_lookup (ใช้จาก df เดิม)
df_group = df[['เกรดโครงการ', 'พื้นที่โครงการ(ตรม)', 'ทาวโฮม', 'บ้านแฝด', 'บ้านเดี่ยว', 'อาคารพาณิชย์', 'จำนวนหลัง']].copy()
df_group['%ทาวโฮม'] = df_group['ทาวโฮม'] / df_group['จำนวนหลัง'].replace(0, np.nan)
df_group['%บ้านแฝด'] = df_group['บ้านแฝด'] / df_group['จำนวนหลัง'].replace(0, np.nan)
df_group['%บ้านเดี่ยว'] = df_group['บ้านเดี่ยว'] / df_group['จำนวนหลัง'].replace(0, np.nan)
df_group['%อาคารพาณิชย์'] = df_group['อาคารพาณิชย์'] / df_group['จำนวนหลัง'].replace(0, np.nan)
bins = [0, 20000, 40000, 60000, 80000, 100000, float("inf")]
labels = ["≤20k", "20k-40k", "40k-60k", "60k-80k", "80k-100k", "100k+"]
df_group['กลุ่มพื้นที่'] = pd.cut(df_group['พื้นที่โครงการ(ตรม)'], bins=bins, labels=labels, right=True, include_lowest=True) # เพิ่ม include_lowest=True
grouped_ratio = df_group.groupby(['เกรดโครงการ', 'กลุ่มพื้นที่'], observed=True)[["%ทาวโฮม", "%บ้านแฝด", "%บ้านเดี่ยว", "%อาคารพาณิชย์"]].mean().round(3)
grouped_ratio_dict = grouped_ratio.to_dict(orient="index")

def adjust_by_grade_policy(grade, ratios):
    # ปรับปรุงฟังก์ชันให้จัดการกับ NaN ใน ratios ได้
    ratios_clean = [r if pd.notna(r) else 0 for r in ratios] # แทนที่ NaN ด้วย 0 ชั่วคราว
    if grade in ['PRIMO', 'BELLA', 'WATTANALAI']:
        # ตรวจสอบว่า ratios_clean[2] มีค่าหรือไม่ก่อนใช้ min
        ratios_clean[2] = min(ratios_clean[2], 0.2) if pd.notna(ratios_clean[2]) else 0.2
        remain = 1 - ratios_clean[2] - (ratios_clean[3] if pd.notna(ratios_clean[3]) else 0) # ใช้ค่า 0 หากเป็น NaN
        if remain < 0: remain = 0 # ป้องกันค่าติดลบ
        ratios_clean[0] = remain * 0.65
        ratios_clean[1] = remain * 0.35
    
    # ทำให้ผลรวมของ ratios เป็น 1 อีกครั้งหลังการปรับ
    current_sum = sum(ratios_clean)
    if current_sum > 0:
        return [r / current_sum for r in ratios_clean]
    return [0.25, 0.25, 0.25, 0.25] # Default if sum is 0

def get_ratio_from_lookup(grade, area):
    group = labels[-1]
    # ปรับการหา group ให้ครอบคลุมขอบเขต
    for i, b in enumerate(bins[:-1]):
        if i == 0 and area <= bins[i+1]: # สำหรับกลุ่มแรก (<=20k)
            group = labels[i]
            break
        elif b < area <= bins[i+1]:
            group = labels[i]
            break
    
    ratio = grouped_ratio_dict.get((grade, group))
    if ratio and any(pd.notna(list(ratio.values()))):
        # กรองค่า NaN ออกก่อนรวมและคำนวณ total
        valid_values = [v for v in ratio.values() if pd.notna(v)]
        total = sum(valid_values) or 1
        
        ratios = [ratio.get('%ทาวโฮม', 0)/total,
                  ratio.get('%บ้านแฝด', 0)/total,
                  ratio.get('%บ้านเดี่ยว', 0)/total,
                  ratio.get('%อาคารพาณิชย์', 0)/total]
        # ตรวจสอบและแทนที่ NaN ด้วย 0 หากมี
        ratios = [r if pd.notna(r) else 0 for r in ratios]
        return adjust_by_grade_policy(grade, ratios)
    return None

# ====== PREDICT ======
if submitted:
    พื้นที่_ตรม = พื้นที่_วา * 4 # แปลงตารางวาเป็นตารางเมตร
    rai = พื้นที่_ตรม / 1600 # แปลงตารางเมตรเป็นไร่
    
    # สร้าง DataFrame สำหรับข้อมูลนำเข้าเพื่อทำนาย
    # ข้อควรระวัง: การใช้ .mean() สำหรับ 'ความยาวถนน', 'ความกว้างถนนปกติ', 'พื้นที่เฉลี่ย(ทาวโฮม/บ้านแฝด/บ้านเดี่ยว)'
    # อาจลดความแม่นยำหากโครงการใหม่มีค่าเหล่านี้แตกต่างจากค่าเฉลี่ยของข้อมูลเดิมมาก
    # หากต้องการความแม่นยำสูงสุด ควรให้ผู้ใช้ป้อนค่าเหล่านี้สำหรับโครงการใหม่โดยเฉพาะ
    input_data = pd.DataFrame([{ 
        'จังหวัด': จังหวัด,
        'เกรดโครงการ': เกรด,
        'พื้นที่โครงการ(ตรม)': พื้นที่_ตรม,
        'รูปร่างที่ดิน': รูปร่าง,
        'ความยาวถนน': df['ความยาวถนน'].mean(), 
        'ความกว้างถนนปกติ': df['ความกว้างถนนปกติ'].mean(),
        'พื้นที่เฉลี่ย(ทาวโฮม)': df['พื้นที่เฉลี่ย(ทาวโฮม)'].mean(),
        'พื้นที่เฉลี่ย(บ้านแฝด)': df['พื้นที่เฉลี่ย(บ้านแฝด)'].mean(),
        'พื้นที่เฉลี่ย(บ้านเดี่ยว)': df['พื้นที่เฉลี่ย(บ้านเดี่ยว)'].mean(),
    }])
    
    # แปลงข้อมูลนำเข้าให้เป็น One-Hot Encoding และจัดเรียงคอลัมน์ให้ตรงกับ X_encoded
    input_enc = pd.get_dummies(input_data, columns=['จังหวัด', 'เกรดโครงการ', 'รูปร่างที่ดิน'])
    
    # ตรวจสอบและเพิ่มคอลัมน์ที่ขาดหายไป (หากมี) เพื่อให้โครงสร้างตรงกับ X_encoded
    for col in X_encoded.columns: # ใช้ X_encoded.columns ที่ผ่านการทำความสะอาดแล้ว
        if col not in input_enc.columns:
            input_enc[col] = 0
    input_enc = input_enc[X_encoded.columns] # จัดเรียงคอลัมน์ให้ตรงกัน

    # ทำการทำนายด้วยโมเดล
    pred = model.predict(input_enc)[0]

    # คำนวณผลลัพธ์ตามสัดส่วนที่ทำนายได้
    พท_สาธา = pred[0] * พื้นที่_ตรม
    พท_ขาย = pred[1] * พื้นที่_ตรม
    พท_สวน = pred[2] * พื้นที่_ตรม
    หลังรวม = pred[3] * rai # pred[3] คือ 'จำนวนหลังต่อไร่'
    พท_ถนน = พท_สาธา * ถนน_ต่อ_สาธารณะ_เฉลี่ย # ใช้ค่าเฉลี่ยถนนต่อสาธารณะ

    # คำนวณจำนวนหลังแยกตามประเภทบ้าน
    # ตรวจสอบว่า pred[4:8] มีค่า NaN หรือไม่ก่อนนำไปใช้
    ทาวโฮม_ratio = pred[4] if pd.notna(pred[4]) else 0
    บ้านแฝด_ratio = pred[5] if pd.notna(pred[5]) else 0
    บ้านเดี่ยว_ratio = pred[6] if pd.notna(pred[6]) else 0
    อาคารพาณิชย์_ratio = pred[7] if pd.notna(pred[7]) else 0

    ratio_hist = get_ratio_from_lookup(เกรด, พื้นที่_ตรม) # ใช้ พื้นที่_ตรม แทน area
    if ratio_hist:
        ทาวโฮม, บ้านแฝด, บ้านเดี่ยว, อาคารพาณิชย์ = [หลังรวม * r for r in ratio_hist]
    else:
        raw_ratios = [ทาวโฮม_ratio, บ้านแฝด_ratio, บ้านเดี่ยว_ratio, อาคารพาณิชย์_ratio]
        total = sum(raw_ratios) or 1
        raw_ratios = [r / total for r in raw_ratios]
        raw_ratios = adjust_by_grade_policy(เกรด, raw_ratios)
        ทาวโฮม, บ้านแฝด, บ้านเดี่ยว, อาคารพาณิชย์ = [หลังรวม * r for r in raw_ratios]

    # คำนวณจำนวนซอย โดยใช้ค่าเฉลี่ย 'หลังต่อซอย' ตามเกรดโครงการ
    ซอย = หลังรวม / avg_ซอยต่อหลัง.get(เกรด, 12) # ใช้ 12 เป็นค่า default หากไม่พบเกรด

    st.markdown("---")
    st.markdown("## 🌟 ผลลัพธ์พยากรณ์")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("พื้นที่สาธารณะ", f"{พท_สาธา/4:,.0f} ตร.วา")
        st.metric("พื้นที่จัดจำหน่าย", f"{พท_ขาย/4:,.0f} ตร.วา")
        st.metric("พื้นที่สวน", f"{พท_สวน/4:,.0f} ตร.วา")
        st.metric("พื้นที่ถนน", f"{พท_ถนน/4:,.0f} ตร.วา")
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

st.markdown("---")
st.caption("Developed by mmethaa | Smart Layout AI 🚀")
