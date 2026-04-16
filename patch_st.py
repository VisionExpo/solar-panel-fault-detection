with open("apps/api/streamlit_app.py", "r") as f:
    content = f.read()

content = content.replace('st.experimental_rerun()', 'st.rerun()')

with open("apps/api/streamlit_app.py", "w") as f:
    f.write(content)
