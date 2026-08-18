import { qwen36 } from "../models/qwen3.6-35b-a3b.model"
import { defineMtpComparison } from "./mtp-comparison"

export default defineMtpComparison({
  id: "qwen36-q4-mtp-mlx-vlm-vs-llama-cpp",
  title: "Qwen3.6 35B-A3B Q4: MLX-VLM MTP vs llama.cpp MTP",
  llamaTarget: qwen36.artifacts.llamaQ4,
  llamaDrafter: qwen36.artifacts.llamaMtpBf16,
  mlxTarget: qwen36.artifacts.mlx4,
  mlxDrafter: qwen36.artifacts.mlxMtpBf16,
})
