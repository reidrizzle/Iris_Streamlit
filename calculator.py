import streamlit as st

st.title('แอปพลิเคชันเครื่องคิดเลข')

num1 = st.number_input('ป้อนตัวเลขที่ 1', value=0.0)
num2 = st.number_input('ป้อนตัวเลขที่ 2', value=0.0)

operation = st.selectbox('เลือกการคำนวณ', ('บวก (+)', 'ลบ (-)', 'คูณ (*)', 'หาร (/)'))

if st.button('คำนวณผลลัพธ์'):
    if operation == 'บวก (+)':
        result = num1 + num2
        st.success(f'ผลลัพธ์: {result}')
    elif operation == 'ลบ (-)':
        result = num1 - num2
        st.success(f'ผลลัพธ์: {result}')
    elif operation == 'คูณ (*)':
        result = num1 * num2
        st.success(f'ผลลัพธ์: {result}')
    elif operation == 'หาร (/)':
        if num2 != 0:
            result = num1 / num2
            st.success(f'ผลลัพธ์: {result}')
        else:
            st.error('ข้อผิดพลาด: ไม่สามารถหารด้วยศูนย์ได้')