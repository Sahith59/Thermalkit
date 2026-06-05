"""MLX inference wrapper.

`generate` pulls in mlx_lm (and transformers), which is slow to import. To keep
lightweight commands like `stats`, `doctor`, and `export` fast, that symbol is
loaded lazily — `from thermalkit.mlx_wrap import generate` still works, but
importing a submodule (e.g. state_builder) no longer drags in mlx_lm.
"""

from thermalkit.mlx_wrap.inference_log import InferenceLog
from thermalkit.mlx_wrap.state_builder import StateBuilder
from thermalkit.mlx_wrap.tok_sec_meter import TokSecMeter

__all__ = ["generate", "InferenceLog", "StateBuilder", "TokSecMeter"]


def __getattr__(name):
    if name == "generate":
        from thermalkit.mlx_wrap._generate import generate

        return generate
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
