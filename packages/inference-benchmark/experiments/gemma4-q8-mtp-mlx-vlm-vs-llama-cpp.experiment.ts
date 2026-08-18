import { gemma4 } from "../models/gemma-4-26b-a4b-it.model"
import { defineMtpComparison } from "./mtp-comparison"

export default defineMtpComparison({
  id: "gemma4-q8-mtp-mlx-vlm-vs-llama-cpp",
  title: "Gemma 4 26B-A4B Instruct Q8: MLX-VLM MTP vs llama.cpp MTP",
  llamaTarget: gemma4.artifacts.llamaQ8,
  llamaDrafter: gemma4.artifacts.llamaMtpF16,
  mlxTarget: gemma4.artifacts.mlxVlm8,
  mlxDrafter: gemma4.artifacts.mlxMtpBf16,
})
