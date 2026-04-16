with open("apps/api/streamlit_app.py", "r") as f:
    content = f.read()

content = content.replace('''if ENABLE_AUTO_REFRESH:
    time.sleep(REFRESH_INTERVAL)
    st.rerun()''', '''# if ENABLE_AUTO_REFRESH:
#     time.sleep(REFRESH_INTERVAL)
#     st.rerun()''')

with open("apps/api/streamlit_app.py", "w") as f:
    f.write(content)
