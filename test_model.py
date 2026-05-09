import sys
from unittest.mock import MagicMock
sys.modules['numpy'] = MagicMock()
sys.modules['tensorflow'] = MagicMock()
sys.modules['tensorflow.keras'] = MagicMock()
sys.modules['tensorflow.keras.layers'] = MagicMock()
sys.modules['tensorflow.keras.models'] = MagicMock()
sys.modules['tensorflow.keras.callbacks'] = MagicMock()
sys.modules['PIL'] = MagicMock()
sys.modules['sklearn'] = MagicMock()
sys.modules['sklearn.metrics'] = MagicMock()
sys.modules['wandb'] = MagicMock()

import solar_fault_detector.models.cnn
import solar_fault_detector.models.ensemble
import solar_fault_detector.training.trainer
import solar_fault_detector.monitoring.realtime
print("Success!")
