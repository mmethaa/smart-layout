import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
# import folium # ไม่ได้ใช้ในเวอร์ชันนี้
# from streamlit_folium import st_folium # ไม่ได้ใช้ในเวอร์ชันนี้

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
df['หลังต่อซอย'] = df['จำนวนหลัง'] / df['จำนวนซอย'].replace(0, 1) # ป้องกันการหารด้วย 0
df['%บ้านเดี่ยว'] = df['บ้านเดี่ยว'] / df['จำนวนหลัง'].replace(0, 1)
df['%บ้านแฝด'] = df['บ้านแฝด'] / df['จำนวนหลัง'].replace(0, 1)
df['%ทาวโฮม'] = df['ทาวโฮม'] / df['จำนวนหลัง'].replace(0, 1)
df['%พื้นที่ขาย'] = df['พื้นที่จัดจำหน่าย(ตรม)'] / df['พื้นที่โครงการ(ตรม)']
df['%พื้นที่สาธา'] = df['พื้นที่สาธา(ตรม)'] / df['พื้นที่โครงการ(ตรม)']
# ตรวจสอบว่าคอลัมน์ 'พื้นที่สวน(5%ของพื้นที่จัดจำหน่าย)' มีอยู่จริงหรือไม่
if 'พื้นที่สวน(5%ของพื้นที่จัดจำหน่าย)' in df.columns:
    df['%พื้นที่สวน'] = df['พื้นที่สวน(5%ของพื้นที่จัดจำหน่าย)'] / df['พื้นที่โครงการ(ตรม)']
else:
    st.warning("ไม่พบคอลัมน์ 'พื้นที่สวน(5%ของพื้นที่จัดจำหน่าย)' ในข้อมูล อาจส่งผลต่อความแม่นยำของ %พื้นที่สวน")
    df['%พื้นที่สวน'] = 0.05 # กำหนดค่าเริ่มต้นหากไม่พบ
    
df['%ถนนในสาธารณะ'] = (df['พื้นที่ถนนรวม'] / df['พื้นที่สาธา(ตรม)']).fillna(0)

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
    'จำนวนหลังต่อไร่': df['จำนวนหลัง'] / (df['พื้นที่โครงการ(ตรม)'] / 1600),
    'สัดส่วนทาวโฮม': df['%ทาวโฮม'],
    'สัดส่วนบ้านแฝด': df['%บ้านแฝด'],
    'สัดส่วนบ้านเดี่ยว': df['%บ้านเดี่ยว'],
    'สัดส่วนอาคารพาณิชย์': df['อาคารพาณิชย์'] / df['จำนวนหลัง'].replace(0, 1) # ป้องกันการหารด้วย 0
})

# แปลงข้อมูลหมวดหมู่ใน X_raw ให้เป็นตัวเลขด้วย One-Hot Encoding
X = pd.get_dummies(X_raw, columns=['จังหวัด', 'เกรดโครงการ', 'รูปร่างที่ดิน'])

# แบ่งข้อมูลเป็นชุดฝึก (Train) และชุดทดสอบ (Test) เพื่อประเมินประสิทธิภาพที่แท้จริงของโมเดล
X_train, X_test, y_train, y_test = train_test_split(X, y_ratio, test_size=0.2, random_state=42)

# สร้างและฝึกโมเดล MultiOutputRegressor ด้วย RandomForestRegressor
# เพิ่ม n_estimators เป็น 200 เพื่อเพิ่มความแม่นยำ (อาจใช้เวลานานขึ้นเล็กน้อย)
model = MultiOutputRegressor(RandomForestRegressor(n_estimators=200, random_state=42)).fit(X_train, y_train)

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
        จังหวัด = st.selectbox("📍 จังหวัด", sorted(df['จังหวัด'].dropna().unique()))
        รูปร่าง = st.selectbox("🧱️ รูปร่างที่ดิน", sorted(df['รูปร่างที่ดิน'].dropna().unique()))
    with col2:
        เกรด = st.selectbox("🏧 เกรดโครงการ", sorted(df['เกรดโครงการ'].dropna().unique()))
        พื้นที่_วา = st.number_input("📀 พื้นที่โครงการ (ตารางวา)", min_value=250, value=7500, step=100)
    submitted = st.form_submit_button("🚀 เริ่มพยากรณ์")

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
    
    # แปลงข้อมูลนำเข้าให้เป็น One-Hot Encoding และจัดเรียงคอลัมน์ให้ตรงกับ X_train
    input_enc = pd.get_dummies(input_data, columns=['จังหวัด', 'เกรดโครงการ', 'รูปร่างที่ดิน'])
    
    # ตรวจสอบและเพิ่มคอลัมน์ที่ขาดหายไป (หากมี) เพื่อให้โครงสร้างตรงกับ X_train
    for col in X.columns:
        if col not in input_enc.columns:
            input_enc[col] = 0
    input_enc = input_enc[X.columns] # จัดเรียงคอลัมน์ให้ตรงกัน

    # ทำการทำนายด้วยโมเดล
    pred = model.predict(input_enc)[0]

    # คำนวณผลลัพธ์ตามสัดส่วนที่ทำนายได้
    พท_สาธา = pred[0] * พื้นที่_ตรม
    พท_ขาย = pred[1] * พื้นที่_ตรม
    พท_สวน = pred[2] * พื้นที่_ตรม
    หลังรวม = pred[3] * rai # pred[3] คือ 'จำนวนหลังต่อไร่'
    พท_ถนน = พท_สาธา * ถนน_ต่อ_สาธารณะ_เฉลี่ย # ใช้ค่าเฉลี่ยถนนต่อสาธารณะ

    # คำนวณจำนวนหลังแยกตามประเภทบ้าน
    ทาวโฮม = หลังรวม * pred[4] # pred[4] คือ 'สัดส่วนทาวโฮม'
    บ้านแฝด = หลังรวม * pred[5] # pred[5] คือ 'สัดส่วนบ้านแฝด'
    บ้านเดี่ยว = หลังรวม * pred[6] # pred[6] คือ 'สัดส่วนบ้านเดี่ยว'
    อาคารพาณิชย์ = หลังรวม * pred[7] # pred[7] คือ 'สัดส่วนอาคารพาณิชย์'
    
    # คำนวณจำนวนซอย โดยใช้ค่าเฉลี่ย 'หลังต่อซอย' ตามเกรดโครงการ
    ซอย = หลังรวม / df.groupby('เกรดโครงการ')['หลังต่อซอย'].mean().get(เกรด, 12) # ใช้ 12 เป็นค่า default หากไม่พบเกรด

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
