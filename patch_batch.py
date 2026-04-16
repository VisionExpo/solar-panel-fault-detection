with open("src/solar_fault_detector/inference/batch.py", "r") as f:
    content = f.read()

content = content.replace('''    def _predict_batch(self, image_paths: List[Path]) -> List[Dict]:
        """Predict on a single batch with caching."""
        # Check cache first''', '''    def _predict_batch(self, image_paths: List[Path]) -> List[Dict]:
        """Predict on a single batch with caching."""
        results = []
        # Check cache first''')

with open("src/solar_fault_detector/inference/batch.py", "w") as f:
    f.write(content)
