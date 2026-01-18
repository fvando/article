from dataclasses import dataclass
from typing import Optional

@dataclass
class ExperimentContext:
    mode: str

    use_f1: bool = False
    use_f2: bool = False
    use_lns: bool = False

    @staticmethod
    def from_mode(mode: str) -> "ExperimentContext":
        return ExperimentContext(
            mode=mode,
            use_f1=mode in ("ml_f1", "lns_ml"),
            use_f2=mode == "lns_ml",
            use_lns=mode in ("lns_baseline", "lns_ml"),
        )
