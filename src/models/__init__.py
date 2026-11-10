from .detector import BEVFusionDetector

try:
    from .early_detector import EarlyBEVFusionDetector
    __all__ = ["BEVFusionDetector", "EarlyBEVFusionDetector"]
except ImportError:
    __all__ = ["BEVFusionDetector"]
