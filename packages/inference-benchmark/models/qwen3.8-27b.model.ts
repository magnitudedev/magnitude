import { defineModel } from "../src/experiment"

export const qwen38 = defineModel({
  id: "qwen3.8-27b",
  source: {
    repository: "Qwen/Qwen3.8-27B",
    revision: "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
  },
  artifacts: {
    llamaQ4: {
      kind: "gguf",
      repository: "unsloth/Qwen3.8-27B-GGUF",
      revision: "fdd03b8bbd279c1694563650e79d85a2373d9934",
      file: "Qwen3.8-27B-UD-Q4_K_XL.gguf",
      sizeBytes: 17_923_394_624,
      sha256: "bee238bbeb3dc0a34bde4d0dedbaee1f98c009e8bb4226f03070054c12fb1372",
      quantization: { family: "gguf", scheme: "UD-Q4_K_XL" },
    },
    dflashQ8: {
      kind: "gguf",
      repository: "incoai/Qwen3.8-27B-DFlash2-GGUF",
      revision: "6cb5872e2cee6b4e780a8414922350be8e42d65c",
      file: "Qwen3.8-27B-DFlash2-Q8_0.gguf",
      sizeBytes: 2_056_414_752,
      sha256: "7f1c9a31a6ed40044c69f6508b50fd63b87abd8e1fb7fe4290303df549153751",
      quantization: { family: "gguf", scheme: "Q8_0-DFlash2" },
    },
  },
})
