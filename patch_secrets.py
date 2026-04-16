with open(".streamlit/secrets.toml", "r") as f:
    content = f.read()

content = content.replace("RENDER_API_KEY=rnd_bn4oOO6D2yoPrfA9jXJiw2spFWF4", 'RENDER_API_KEY="rnd_bn4oOO6D2yoPrfA9jXJiw2spFWF4"')

with open(".streamlit/secrets.toml", "w") as f:
    f.write(content)
