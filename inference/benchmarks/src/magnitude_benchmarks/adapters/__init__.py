from .llama_cpp import LlamaCpp
from .mlx_vlm import MlxVlm
from .native import Native
from .omlx import Omlx

ADAPTERS = {
    "magnitude": Native,
    "mlx-vlm": MlxVlm,
    "omlx": Omlx,
    "llama.cpp": LlamaCpp,
}
