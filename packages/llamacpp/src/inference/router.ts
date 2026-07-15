import { Effect } from "effect"
import * as FileSystem from "@effect/platform/FileSystem"
import * as Path from "@effect/platform/Path"
import { presetDir } from "../paths"
import type { LocalModelInfo } from "../models/types"
import type { PresetDefaults } from "./types"

/**
 * Generate a router mode INI preset string.
 *
 * The `[*]` section contains global defaults.
 * Per-model sections are named by model ID and specify the model file path + alias.
 */
export function generatePreset(
  models: readonly LocalModelInfo[],
  defaults: PresetDefaults,
): string {
  const lines: string[] = []

  // Global defaults section
  lines.push("[*]")
  lines.push(`LLAMA_ARG_N_GPU_LAYERS = ${defaults.ngl}`)
  if (defaults.ctx) lines.push(`LLAMA_ARG_CTX_SIZE = ${defaults.ctx}`)
  lines.push(`LLAMA_ARG_FLASH_ATTN = auto`)
  lines.push(`LLAMA_ARG_CONT_BATCHING = true`)
  lines.push(`LLAMA_ARG_JINJA = true`)
  if (defaults.sleepIdleSeconds) {
    lines.push(`LLAMA_ARG_SLEEP_IDLE_SECONDS = ${defaults.sleepIdleSeconds}`)
  }

  // Per-model sections
  for (const model of models) {
    lines.push("")
    lines.push(`[${model.id}]`)
    lines.push(`LLAMA_ARG_MODEL = ${model.filePath}`)
    lines.push(`LLAMA_ARG_ALIAS = ${model.id}`)
    if (model.mmprojPath) {
      lines.push(`LLAMA_ARG_MMPROJ = ${model.mmprojPath}`)
    }
    if (defaults.loadOnStartup === model.id) {
      lines.push(`__PRESET_LOAD_ON_STARTUP = true`)
    }
  }

  return lines.join("\n")
}

/**
 * Write a router preset to a temp file and return its path.
 * Files are written to `~/.magnitude/llamacpp/presets/`.
 */
export function writePreset(
  content: string,
): Effect.Effect<string, never, FileSystem.FileSystem | Path.Path> {
  return Effect.gen(function* () {
    const fs = yield* FileSystem.FileSystem
    const pathSvc = yield* Path.Path

    const dir = presetDir()
    yield* fs.makeDirectory(dir, { recursive: true }).pipe(
      Effect.catchAll(() => Effect.void),
    )

    const timestamp = Date.now()
    const presetPath = pathSvc.join(dir, `router-${timestamp}.ini`)
    yield* fs.writeFileString(presetPath, content).pipe(
      Effect.catchAll(() => Effect.void),
    )
    return presetPath
  })
}
