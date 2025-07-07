import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split

# โหลดข้อมูล
df = pd.read_excel("project_data.xlsx")
df.columns = df.columns.str.strip()

# คำนวณคอลัมน์เพิ่ม
df['หลังต่อซอย'] = df['จำนวนหลัง'] / df['จำนวนซอย'].replace(0, 1)
df['%บ้านเดี่ยว'] = df['บ้านเดี่ยว'] / df['จำนวนหลัง'].replace(0, 1)
df['%บ้านแฝด'] = df['บ้านแฝด'] / df['จำนวนหลัง'].replace(0, 1)
df['%ทาวโฮม'] = df['ทาวโฮม'] / df['จำนวนหลัง'].replace(0, 1)
df['%พื้นที่ขาย'] = df['พื้นที่จัดจำหน่าย(ตรม)'] / df['พื้นที่โครงการ(ตรม)']
df['%พื้นที่สาธา'] = df['พื้นที่สาธา(ตรม)'] / df['พื้นที่โครงการ(ตรม)']

# เตรียมข้อมูลสำหรับ train
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

model = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42))
model.fit(X_train, y_train)

# ค่าเฉลี่ยซอยต่อหลัง
avg_ซอยต่อหลัง_ตามเกรด = df.groupby('เกรดโครงการ')['หลังต่อซอย'].mean().to_dict()

# ==== UI ====
st.title("🏗️ Smart Layout Predictor")

col1, col2 = st.columns(2)
จังหวัด = col1.selectbox("จังหวัด", sorted(df['จังหวัด'].dropna().unique()))
เกรด = col2.selectbox("เกรดโครงการ", sorted(df['เกรดโครงการ'].dropna().unique()))
รูปร่าง = st.selectbox("รูปร่างที่ดิน", sorted(df['รูปร่างที่ดิน'].dropna().unique()))
พื้นที่ = st.number_input("พื้นที่โครงการ (ตร.ม.)", value=30000, step=100)

if st.button("📊 พยากรณ์ผลลัพธ์"):
    area = พื้นที่
    rai = area / 1600

    new_input = pd.DataFrame([{
        'จังหวัด': จังหวัด,
        'เกรดโครงการ': เกรด,
        'พื้นที่โครงการ(ตรม)': area,
        'รูปร่างที่ดิน': รูปร่าง
    }])
    new_encoded = pd.get_dummies(new_input)
    for col in X_train.columns:
        if col not in new_encoded.columns:
            new_encoded[col] = 0
    new_encoded = new_encoded[X_train.columns]

    ratio_pred = model.predict(new_encoded)[0]
    พท_สาธา = ratio_pred[0] * area
    พท_ขาย = ratio_pred[1] * area
    พท_สวน = ratio_pred[2] * area
    หลังรวม = ratio_pred[3] * rai

    # แยกสัดส่วนบ้าน
    total_ratio = sum(ratio_pred[4:8]) or 1
    ทาวโฮม, บ้านแฝด, บ้านเดี่ยว, อาคารพาณิชย์ = [
        หลังรวม * (r / total_ratio) for r in ratio_pred[4:8]
    ]

    # ปรับตามเกรด
    คำเตือน = None
    if เกรด in ['ESSENCE', 'PRIME', 'GRANDESSENCE', 'GRAND', 'PRIMA VILLA']:
        ทาวโฮม = บ้านแฝด = อาคารพาณิชย์ = 0
        บ้านเดี่ยว = หลังรวม
        คำเตือน = "🏡 เกรดพรีเมียม: เฉพาะบ้านเดี่ยว"
    elif เกรด in ['BELLA', 'PRIMO']:
        if บ้านเดี่ยว > หลังรวม * 0.2:
            บ้านเดี่ยว = หลังรวม * 0.2
        if อาคารพาณิชย์ > หลังรวม * 0.1:
            อาคารพาณิชย์ = 0
        คำเตือน = "🏘️ เกรดประหยัด: เน้นทาวน์โฮม + บ้านแฝด"

    ซอย = หลังรวม / avg_ซอยต่อหลัง_ตามเกรด.get(เกรด, 12)

    # ==== OUTPUT ====
    st.subheader("📍 สรุปผลพยากรณ์")
    st.markdown(f"**จังหวัด:** {จังหวัด} &nbsp;&nbsp; **เกรด:** {เกรด} &nbsp;&nbsp; **รูปร่างที่ดิน:** {รูปร่าง}")
    st.markdown(f"**พื้นที่โครงการ:** {area:,.0f} ตร.ม. ({rai:.2f} ไร่)")
    if คำเตือน:
        st.warning(คำเตือน)

    st.markdown(f"""
    ### 📐 พื้นที่ใช้สอย
    - พื้นที่สาธารณะ: **{พท_สาธา:,.0f} ตร.ม.**
    - พื้นที่จัดจำหน่าย: **{พท_ขาย:,.0f} ตร.ม.**
    - พื้นที่สวน: **{พท_สวน:,.0f} ตร.ม.**

    ### 🏘️ จำนวนยูนิต (ทั้งหมด {หลังรวม:,.0f} หลัง)
    - ทาวน์โฮม: {ทาวโฮม:,.0f}
    - บ้านแฝด: {บ้านแฝด:,.0f}
    - บ้านเดี่ยว: {บ้านเดี่ยว:,.0f}
    - อาคารพาณิชย์: {อาคารพาณิชย์:,.0f}

    ### 🛣️ จำนวนซอย: {ซอย:,.0f} (อิงเกรด {avg_ซอยต่อหลัง_ตามเกรด.get(เกรด, 12):.1f} หลัง/ซอย)
    """)