import { basename, dirname } from "node:path"

const MMPROJ_RE = /mmproj/i

/**
 * Pair model files with their corresponding mmproj projector files.
 * Returns a Map from model file path → mmproj file path.
 *
 * Strategy: exact match on base name, then any mmproj in the same directory,
 * preferring BF16/F16 over quantized projectors.
 */
export function pairMmproj(
  modelFiles: readonly string[],
): Map<string, string> {
  const result = new Map<string, string>()
  const mmprojs = modelFiles.filter((f) => MMPROJ_RE.test(basename(f)))
  const models = modelFiles.filter((f) => !MMPROJ_RE.test(basename(f)))

  for (const model of models) {
    const modelDir = dirname(model)
    const modelBase = basename(model).replace(/\.gguf$/i, "")

    const candidates = mmprojs
      .filter((mm) => dirname(mm) === modelDir)
      .map((mm) => {
        const mmBase = basename(mm)
        const score = mmBase.toLowerCase().includes(modelBase.toLowerCase()) ? 0
          : mmBase.match(/bf16|f16/i) ? 1
          : 2
        return { mm, score }
      })
      .sort((a, b) => a.score - b.score)

    if (candidates.length > 0) {
      result.set(model, candidates[0].mm)
    }
  }

  return result
}
