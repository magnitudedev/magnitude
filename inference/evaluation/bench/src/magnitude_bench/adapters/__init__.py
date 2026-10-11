from .llama_cpp import LlamaCpp
from .mlx_vlm import MlxVlm
from .native import Native
from .ollama import Ollama
from .omlx import Omlx

ADAPTERS = {
    "magnitude": Native,
    "llama.cpp": LlamaCpp,
    "ollama": Ollama,
    "ollama-mlx": Ollama,
    "ollama-registry": Ollama,
    "mlx-vlm": MlxVlm,
    "omlx": Omlx,
}
