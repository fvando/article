from enum import Enum

class ExperimentMode(str, Enum):
    BASELINE = "baseline"
    ML_F1 = "ml_f1"
    LNS_BASE = "lns_baseline"
    LNS_ML = "lns_ml"
