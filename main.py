import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split

# CONFIG
st.set_page_config(page_title="Smart Layout AI", page_icon="🏗️", layout="centered")

# STYLING
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

# LOAD DATA
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

# FORM INPUT
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

# PREDICTION
if submitted:
    input_df = pd.DataFrame([{ 'จังหวัด': จังหวัด, 'เกรดโครงการ': เกรด, 'พื้นที่โครงการ(ตรม)': พื้นที่, 'รูปร่างที่ดิน': รูปร่าง }])
    encoded = pd.get_dummies(input_df)
    for col in X.columns:
        if col not in encoded.columns:
            encoded[col] = 0
    encoded = encoded[X.columns]
    pred = model.predict(encoded)[0]

    # คำนวณผลลัพธ์
    พท_ขาย = pred[1] * (พื้นที่ / 4)
    พท_สาธา = pred[0] * (พื้นที่ / 4)
    พท_สวน = pred[2] * (พื้นที่ / 4)
    พท_ถนน = พท_สาธา - พท_สวน
    หลังรวม = pred[3] * (พื้นที่ / 1600)
    ซอย = หลังรวม / avg_ซอยต่อหลัง.get(เกรด, 12)
    ทาวโฮม, บ้านแฝด, บ้านเดี่ยว, อาคารพาณิชย์ = [หลังรวม * r for r in pred[4:8]]

    def calc_err(pred, true):
        return round(abs(pred - true) / true * 100, 2) if true else None

    err_dict = {}
    match_row = df[(df['จังหวัด'] == จังหวัด) & (df['เกรดโครงการ'] == เกรด) & (df['รูปร่างที่ดิน'] == รูปร่าง)]
    if not match_row.empty:
        row = match_row.iloc[0]
        พท_ขาย_จริง = row['พื้นที่จัดจำหน่าย(ตรม)'] / 4
        พท_สาธา_จริง = row['พื้นที่สาธา(ตรม)'] / 4
        พท_สวน_จริง = row['พื้นที่สวน(5%ของพื้นที่จัดจำหน่าย)'] / 4
        พท_ถนน_จริง = พท_สาธา_จริง - พท_สวน_จริง
        หลัง_จริง = row['จำนวนหลัง']
        ซอย_จริง = row['จำนวนซอย']

        err_dict = {
            'err_ขาย': calc_err(พท_ขาย, พท_ขาย_จริง),
            'err_สาธา': calc_err(พท_สาธา, พท_สาธา_จริง),
            'err_สวน': calc_err(พท_สวน, พท_สวน_จริง),
            'err_ถนน': calc_err(พท_ถนน, พท_ถนน_จริง),
            'err_แปลง': calc_err(หลังรวม, หลัง_จริง),
            'err_ซอย': calc_err(ซอย, ซอย_จริง)
        }

    def format_err(val):
        return f"{val:.2f} %" if val is not None else "-"

    st.markdown("---")
    st.markdown("## 🌟 ผลลัพพยากรณ์")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("พื้นที่ขาย (ตรว.)", f"{พท_ขาย:,.0f}", help=f"Error: {format_err(err_dict.get('err_ขาย'))}")
        st.metric("พื้นที่สวน (ตรว.)", f"{พท_สวน:,.0f}", help=f"Error: {format_err(err_dict.get('err_สวน'))}")
    with col2:
        st.metric("พื้นที่สาธารณะ (ตรว.)", f"{พท_สาธา:,.0f}", help=f"Error: {format_err(err_dict.get('err_สาธา'))}")
        st.metric("พื้นที่ถนน (ตรว.)", f"{พท_ถนน:,.0f}", help=f"Error: {format_err(err_dict.get('err_ถนน'))}")
    with col3:
        st.metric("จำนวนแปลงรวม", f"{หลังรวม:,.0f}", help=f"Error: {format_err(err_dict.get('err_แปลง'))}")
        st.metric("จำนวนซอย", f"{ซอย:,.0f}", help=f"Error: {format_err(err_dict.get('err_ซอย'))}")

    st.markdown("### 🏡 แยกจำนวนแปลงตามแบบบ้าน")
    st.markdown(f"""
    - ทาวน์โฮม: **{ทาวโฮม:,.0f}** หลัง  
    - บ้านแฝด: **{บ้านแฝด:,.0f}** หลัง  
    - บ้านเดี่ยว: **{บ้านเดี่ยว:,.0f}** หลัง  
    - อาคารพาณิชย์: **{อาคารพาณิชย์:,.0f}** หลัง  
    """)

    # ค้นหาโครงการคล้าย
    def find_similar_projects(df, จังหวัด, เกรด, รูปร่าง, พื้นที่, tol=5000):
        return df[
            (df['จังหวัด'] == จังหวัด) &
            (df['เกรดโครงการ'] == เกรด) &
            (df['รูปร่างที่ดิน'] == รูปร่าง) &
            (df['พื้นที่โครงการ(ตรม)'].between(พื้นที่ - tol, พื้นที่ + tol))
        ]

    similar_df = find_similar_projects(df, จังหวัด, เกรด, รูปร่าง, พื้นที่)

    if not similar_df.empty:
        def calc_errors(row):
            พท_จริง = row['พื้นที่โครงการ(ตรม)']
            พท_ขาย_จริง = row['พื้นที่จัดจำหน่าย(ตรม)'] / 4
            พท_สาธา_จริง = row['พื้นที่สาธา(ตรม)'] / 4
            พท_สวน_จริง = row['พื้นที่สวน(5%ของพื้นที่จัดจำหน่าย)'] / 4
            พท_ถนน_จริง = พท_สาธา_จริง - พท_สวน_จริง
            หลัง_จริง = row['จำนวนหลัง']
            ซอย_จริง = row['จำนวนซอย']

            input_temp = pd.DataFrame([{
                'จังหวัด': จังหวัด, 'เกรดโครงการ': เกรด,
                'พื้นที่โครงการ(ตรม)': พท_จริง, 'รูปร่างที่ดิน': รูปร่าง
            }])
            encoded_temp = pd.get_dummies(input_temp)
            for col in X.columns:
                if col not in encoded_temp.columns:
                    encoded_temp[col] = 0
            encoded_temp = encoded_temp[X.columns]

            pred_temp = model.predict(encoded_temp)[0]
            พท_ขาย_pred = pred_temp[1] * (พท_จริง / 4)
            พท_สาธา_pred = pred_temp[0] * (พท_จริง / 4)
            พท_สวน_pred = pred_temp[2] * (พท_จริง / 4)
            พท_ถนน_pred = พท_สาธา_pred - พท_สวน_pred
            หลัง_pred = pred_temp[3] * (พท_จริง / 1600)
            ซอย_pred = หลัง_pred / avg_ซอยต่อหลัง.get(เกรด, 12)

            def err(p, t): return abs(p - t) / t * 100 if t else 0
            return pd.Series({
                'err_ขาย': err(พท_ขาย_pred, พท_ขาย_จริง),
                'err_สาธา': err(พท_สาธา_pred, พท_สาธา_จริง),
                'err_สวน': err(พท_สวน_pred, พท_สวน_จริง),
                'err_ถนน': err(พท_ถนน_pred, พท_ถนน_จริง),
                'err_แปลง': err(หลัง_pred, หลัง_จริง),
                'err_ซอย': err(ซอย_pred, ซอย_จริง)
            })

        error_df = similar_df.apply(calc_errors, axis=1)
        avg_error = error_df.mean()

        st.markdown("### 📊 ความแม่นยำของโมเดลกับโครงการลักษณะเดียวกันในอดีต")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("ความแม่นยำพื้นที่ขาย", f"{100 - avg_error['err_ขาย']:.2f} %")
            st.metric("ความแม่นยำพื้นที่สวน", f"{100 - avg_error['err_สวน']:.2f} %")
        with col2:
            st.metric("ความแม่นยำพื้นที่สาธารณะ", f"{100 - avg_error['err_สาธา']:.2f} %")
            st.metric("ความแม่นยำพื้นที่ถนน", f"{100 - avg_error['err_ถนน']:.2f} %")
        with col3:
            st.metric("ความแม่นยำจำนวนแปลง", f"{100 - avg_error['err_แปลง']:.2f} %")
            st.metric("ความแม่นยำจำนวนซอย", f"{100 - avg_error['err_ซอย']:.2f} %")
    else:
        st.info("🔍 ไม่พบโครงการที่ใกล้เคียงในอดีตเพียงพอสำหรับประเมินความแม่นยำ")
from sklearn.metrics import mean_absolute_error, r2_score

# หลัง train model เสร็จ:
y_pred = model.predict(X_train)

mae = mean_absolute_error(y_train, y_pred)
r2 = r2_score(y_train, y_pred)

st.markdown("### 📈 ความแม่นยำของโมเดล (Train Set)")
st.write(f"**MAE (Mean Absolute Error):** {mae:.4f}")
st.write(f"**R² Score:** {r2:.4f}")

st.caption("Developed by mmethaa | Smart Layout AI 🚀")
