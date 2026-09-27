from .llama_cpp import LlamaCpp
from .magnitude import Magnitude
from .mlx_vlm import MlxVlm
from .omlx import Omlx

ADAPTERS = {"magnitude": Magnitude, "mlx-vlm": MlxVlm, "omlx": Omlx, "llama.cpp": LlamaCpp}
