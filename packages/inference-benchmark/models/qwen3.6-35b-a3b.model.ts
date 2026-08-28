import { defineModel } from "../src/experiment"
import { resolve } from "node:path"

export const qwen36 = defineModel({
  id: "qwen3.6-35b-a3b",
  contextLimit: 262_144,
  source: {
    repository: "Qwen/Qwen3.6-35B-A3B",
    revision: "995ad96eacd98c81ed38be0c5b274b04031597b0",
  },
  artifacts: {
    llamaMtpBf16: {
      kind: "gguf",
      repository: "a4lg/Qwen3.6-35B-A3B-MTP-ONLY-GGUF",
      revision: "034793caf3d6f3a60cd9f0c37765434be9931cf4",
      file: "Qwen3.6-35B-A3B-MTP-ONLY-BF16.gguf",
      sizeBytes: 3_735_545_632,
      sha256: "dae566e176666fc149c34cf8300af4abf4b5a98ceab8fc1d2b74846f5f34788a",
      quantization: { family: "gguf", scheme: "BF16-MTP" },
    },
    mlxMtpBf16: {
      kind: "mlx",
      repository: "mlx-community/Qwen3.6-35B-A3B-MTP-bf16",
      revision: "e931b93eed744eae16049d4ebeddf636ef5b90f2",
      manifest: resolve(import.meta.dir, "qwen3.6-35b-a3b-mtp-mlx-bf16.lock.json"),
      quantization: { family: "mlx-unquantized", dtype: "bfloat16" },
    },
    llamaQ4: {
      kind: "gguf",
      repository: "unsloth/Qwen3.6-35B-A3B-GGUF",
      revision: "a483e9e6cbd595906af30beda3187c2663a1118c",
      file: "Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf",
      sizeBytes: 22_360_456_160,
      sha256: "707a55a8a4397ecde44de0c499d3e68c1ad1d240d1da65826b4949d1043f4450",
      quantization: { family: "gguf", scheme: "UD-Q4_K_XL" },
    },
    llamaQ8: {
      kind: "gguf",
      repository: "unsloth/Qwen3.6-35B-A3B-GGUF",
      revision: "a483e9e6cbd595906af30beda3187c2663a1118c",
      file: "Qwen3.6-35B-A3B-Q8_0.gguf",
      sizeBytes: 36_903_140_320,
      sha256: "d1a395809f65a43a13ad119eb4e7acdef1ac6d68120f39902c8ab96e72794a59",
      quantization: { family: "gguf", scheme: "Q8_0" },
    },
    mlx8: {
      kind: "mlx",
      repository: "mlx-community/Qwen3.6-35B-A3B-8bit",
      revision: "e06a74e6236a60c8367e1a3214e83d8b61b637b0",
      manifest: resolve(import.meta.dir, "qwen3.6-35b-a3b-mlx-8bit.lock.json"),
      quantization: { family: "mlx-affine", bits: 8, groupSize: 64 },
    },
    mlx4: {
      kind: "mlx",
      repository: "mlx-community/Qwen3.6-35B-A3B-4bit",
      revision: "38740b847e4cb78f352aba30aa41c76e08e6eb46",
      manifest: resolve(import.meta.dir, "qwen3.6-35b-a3b-mlx-4bit.lock.json"),
      quantization: { family: "mlx-affine", bits: 4, groupSize: 64 },
    },
  },
})
