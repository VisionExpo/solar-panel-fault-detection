with open("tests/test_inference.py", "r") as f:
    content = f.read()

content = content.replace('''    mock_model.predict.return_value = np.array(
        [
            [0.2, 0.5, 0.3],
            [0.6, 0.2, 0.2],
        ]
    )''', '''    mock_model.predict.return_value = np.array(
        [
            [0.2, 0.5, 0.3],
            [0.6, 0.2, 0.2],
        ]
    )
    mock_model.return_value.numpy.return_value = np.array(
        [
            [0.2, 0.5, 0.3],
            [0.6, 0.2, 0.2],
        ]
    )''')

content = content.replace('''    mock_model.predict.return_value = np.array([[0.1, 0.7, 0.2]])''', '''    mock_model.predict.return_value = np.array([[0.1, 0.7, 0.2]])
    mock_model.return_value.numpy.return_value = np.array([[0.1, 0.7, 0.2]])''')

with open("tests/test_inference.py", "w") as f:
    f.write(content)
