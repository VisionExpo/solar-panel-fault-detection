with open("tests/test_api.py", "r") as f:
    content = f.read()

content = content.replace('@patch("apps.api.fastapi_app.predictor")',
'''@patch("apps.api.fastapi_app.MODEL_READY", True)
@patch("apps.api.fastapi_app.predictor")''')

with open("tests/test_api.py", "w") as f:
    f.write(content)
