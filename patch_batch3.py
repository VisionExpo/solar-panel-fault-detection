with open("src/solar_fault_detector/inference/batch.py", "r") as f:
    content = f.read()

content = content.replace('''        # Sort by original order if needed
        if results and results[0][0] is not None:
            results.sort(key=lambda x: x[0])
            results = [r for _, r in results]
        else:
            results = [r for _, r in results]''', '''        # Sort by original order if needed
        if results and results[0] and results[0][0] is not None:
            results.sort(key=lambda x: x[0])
            results = [r for _, r in results]
        elif results:
            results = [r for _, r in results]''')

with open("src/solar_fault_detector/inference/batch.py", "w") as f:
    f.write(content)
