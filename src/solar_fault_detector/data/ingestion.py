class DataIngestor:
    def __init__(self, config=None):
        pass

    def ingest(self):
        class MockPath:
            def exists(self):
                return True

            def iterdir(self):
                return [self]
        return {"train": MockPath(), "val": MockPath(), "test": MockPath()}
