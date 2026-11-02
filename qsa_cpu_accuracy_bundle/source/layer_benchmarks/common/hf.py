"""Access to the isolated HF Transformers overlay used for reference parity.

The runtime env pins transformers 5.5.x for QEff; newer model code
(qwen4_exp, deepseek_v4, minimax_m3_vl, qwen3_5_moe) lives in
``layer_benchmarks/.hf_overlay`` (see README). ``import_hf`` puts the overlay
first on sys.path only when a parity check asks for it.
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

OVERLAY = Path(os.environ.get("LB_HF_OVERLAY", Path(__file__).resolve().parents[1] / ".hf_overlay"))


def import_hf(module: str):
    """Import ``transformers.models.<module>`` from the overlay."""
    if OVERLAY.is_dir() and str(OVERLAY) not in sys.path:
        sys.path.insert(0, str(OVERLAY))
        for name in [m for m in sys.modules if m == "transformers" or m.startswith("transformers.")]:
            del sys.modules[name]
    return importlib.import_module(f"transformers.models.{module}")
