import { qwen36 } from "../models/qwen3.6-35b-a3b.model"
import { defineMtpComparison } from "./mtp-comparison"

export default defineMtpComparison({
  id: "qwen36-q8-mtp-mlx-vlm-vs-llama-cpp",
  title: "Qwen3.6 35B-A3B Q8: MLX-VLM MTP vs llama.cpp MTP",
  llamaTarget: qwen36.artifacts.llamaQ8,
  llamaDrafter: qwen36.artifacts.llamaMtpBf16,
  mlxTarget: qwen36.artifacts.mlx8,
  mlxDrafter: qwen36.artifacts.mlxMtpBf16,
})
