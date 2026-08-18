import { resolve } from "node:path"
import { defineModel } from "../src/experiment"

export const gemma4 = defineModel({
  id: "gemma-4-26b-a4b-it",
  source: {
    repository: "google/gemma-4-26B-A4B-it",
    revision: "4d7ae4984b7db7de8f8457170b3f1a419ee76d52",
  },
  artifacts: {
    llamaMtpF16: {
      kind: "gguf",
      repository: "ironbcc/gemma-4-26B-A4B-it-MTP-GGUF",
      revision: "89be4e9e7311db68c5886a41f4c6b2a34acb730a",
      file: "gemma-4-26B-A4B-it-assistant-F16.gguf",
      sizeBytes: 855_228_512,
      sha256: "aded0238f9ca7bb8fc91cd1fa8f205ac6cb983d41ea01f8631c80d8fbdd51c32",
      quantization: { family: "gguf", scheme: "F16-MTP" },
    },
    mlxMtpBf16: {
      kind: "mlx",
      repository: "mlx-community/gemma-4-26B-A4B-it-assistant-bf16",
      revision: "cda74908f1dbe7d3dbd3030e66576a7d4094144f",
      manifest: resolve(import.meta.dir, "gemma-4-26b-a4b-it-mtp-mlx-bf16.lock.json"),
      quantization: { family: "mlx-unquantized", dtype: "bfloat16" },
    },
    llamaQ8: {
      kind: "gguf",
      repository: "unsloth/gemma-4-26B-A4B-it-GGUF",
      revision: "c099eb48e663fd284577b04978a94ffccb261841",
      file: "gemma-4-26B-A4B-it-Q8_0.gguf",
      sizeBytes: 26_859_861_728,
      sha256: "5f7cbd0f4564e84342fc34321a09acb54b1a3da9215124e5bf444baa6dda152c",
      quantization: { family: "gguf", scheme: "Q8_0" },
    },
    llamaQ4: {
      kind: "gguf",
      repository: "unsloth/gemma-4-26B-A4B-it-GGUF",
      revision: "c099eb48e663fd284577b04978a94ffccb261841",
      file: "gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf",
      sizeBytes: 17_010_980_576,
      sha256: "ef728c8e0c337fd1067b947af006e38a9ef2419e56feced4fd29b4bf0636e30c",
      quantization: { family: "gguf", scheme: "UD-Q4_K_XL" },
    },
    mlx8: {
      kind: "mlx",
      repository: "mlx-community/gemma-4-text-26b-a4b-it-8bit",
      revision: "b78a5672015c1b8e0a5b9372fa76a4c18e8e38ac",
      manifest: resolve(import.meta.dir, "gemma-4-26b-a4b-it-mlx-8bit.lock.json"),
      quantization: { family: "mlx-affine", bits: 8, groupSize: 64 },
    },
    mlxVlm8: {
      kind: "mlx",
      repository: "mlx-community/gemma-4-26b-a4b-it-8bit",
      revision: "33c6d23798a0af159529890f79329206dbfbd73c",
      manifest: resolve(import.meta.dir, "gemma-4-26b-a4b-it-mlx-vlm-8bit.lock.json"),
      quantization: { family: "mlx-affine", bits: 8, groupSize: 64 },
    },
    mlx4: {
      kind: "mlx",
      repository: "mlx-community/gemma-4-26b-a4b-it-4bit",
      revision: "0d77464eeb233a2da68ebf9d7dc4edaac7db956d",
      manifest: resolve(import.meta.dir, "gemma-4-26b-a4b-it-mlx-4bit.lock.json"),
      quantization: { family: "mlx-affine", bits: 4, groupSize: 64 },
    },
  },
})
