class DataIngestor:
    def __init__(self, config=None):
        pass
    def ingest(self):
        return {"train": type("MockPath", (), {"exists": lambda self: True})(), "val": type("MockPath", (), {"exists": lambda self: True})(), "test": type("MockPath", (), {"exists": lambda self: True})()}
