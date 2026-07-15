import { Context, Effect, Layer, pipe, Scope } from "effect"
import * as Command from "@effect/platform/Command"
import * as CommandExecutor from "@effect/platform/CommandExecutor"
import * as FileSystem from "@effect/platform/FileSystem"
import * as HttpClient from "@effect/platform/HttpClient"
import { BunCommandExecutor, BunFileSystem, BunPath } from "@effect/platform-bun"
import { FetchHttpClient } from "@effect/platform"
import * as Path from "@effect/platform/Path"
import {
  LlamaCppServerStartFailed,
  LlamaCppServerTimeout,
  LlamaCppServerOutOfMemory,
} from "../errors"
import { findFreePort } from "./ports"
import { waitForReady, detectOom, collectStderr } from "./health"
import { generatePreset, writePreset } from "./router"
import type { ResolvedBinary } from "../binary/types"
import type { LocalModelInfo } from "../models/types"
import type { PresetDefaults, ServerMode } from "./types"

// ── Service Tag (internal) ──

/** Handle to a spawned server process. */
export interface ServerHandle {
  readonly pid: number
  readonly port: number
  readonly endpoint: string
  readonly mode: ServerMode
  readonly process: CommandExecutor.Process
}

export interface LlamaCppServerApi {
  readonly start: (
    binary: ResolvedBinary,
    models: readonly LocalModelInfo[],
    loadOnStartup: string,
    defaults: PresetDefaults,
    port?: number,
  ) => Effect.Effect<
    ServerHandle,
    LlamaCppServerStartFailed | LlamaCppServerTimeout | LlamaCppServerOutOfMemory,
    FileSystem.FileSystem | Path.Path | HttpClient.HttpClient | CommandExecutor.CommandExecutor | Scope.Scope
  >

  readonly stop: (handle: ServerHandle) => Effect.Effect<void, never, never>
}

export class LlamaCppServer extends Context.Tag("LlamaCppServer")<
  LlamaCppServer,
  LlamaCppServerApi
>() {}

// ── Platform layer (baked in) ──

const PlatformLayer = Layer.mergeAll(
  BunCommandExecutor.layer,
  BunPath.layer,
  FetchHttpClient.layer,
).pipe(Layer.provideMerge(BunFileSystem.layer))

// ── Factory ──

export function makeLlamaCppServer(): LlamaCppServerApi {
  const start: LlamaCppServerApi["start"] = (binary, models, loadOnStartup, defaults, preferredPort) =>
    startServer(binary, models, loadOnStartup, defaults, preferredPort).pipe(
      Effect.provide(PlatformLayer),
    )

  const stop: LlamaCppServerApi["stop"] = (handle) =>
    pipe(
      handle.process.kill("SIGTERM"),
      Effect.catchAll(() => Effect.void),
    )

  return { start, stop }
}

// ── Start implementation ──

function startServer(
  binary: ResolvedBinary,
  models: readonly LocalModelInfo[],
  loadOnStartup: string,
  defaults: PresetDefaults,
  preferredPort?: number,
): Effect.Effect<
  ServerHandle,
  LlamaCppServerStartFailed | LlamaCppServerTimeout | LlamaCppServerOutOfMemory,
  FileSystem.FileSystem | Path.Path | HttpClient.HttpClient | CommandExecutor.CommandExecutor | Scope.Scope
> {
  return Effect.gen(function* () {
    // 1. Generate and write preset
    const presetContent = generatePreset(models, { ...defaults, loadOnStartup })
    const presetPath = yield* writePreset(presetContent)

    // 2. Select port
    const port = preferredPort && preferredPort > 0
      ? yield* findFreePort(preferredPort)
      : yield* findFreePort(8080)

    // 3. Spawn process
    const cmd = Command.make(
      binary.path,
      "--models-preset", presetPath,
      "--host", "127.0.0.1",
      "--port", String(port),
      "--flash-attn", "auto",
      "--jinja",
    ).pipe(
      Command.workingDirectory(binary.directory),
      Command.env({
        ...process.env,
        HF_HUB_CACHE: process.env.HF_HUB_CACHE ?? `${process.env.HOME}/.cache/huggingface/hub`,
      }),
    )

    const proc = yield* pipe(
      Command.start(cmd),
      Effect.mapError((err) =>
        new LlamaCppServerStartFailed({ reason: `Failed to spawn process: ${String(err)}` }),
      ),
    )

    const endpoint = `http://127.0.0.1:${port}`

    // 4. Wait for readiness (polls /health)
    // OOM detection happens if waitForReady times out — we then collect stderr
    yield* waitForReady(endpoint).pipe(
      Effect.catchAll((err) =>
        Effect.gen(function* () {
          // On timeout, check stderr for OOM
          const stderrText = yield* collectStderr(proc.stderr)
          if (detectOom(stderrText)) {
            yield* pipe(proc.kill("SIGTERM"), Effect.catchAll(() => Effect.void))
            return yield* new LlamaCppServerOutOfMemory({
              attempted: { ngl: defaults.ngl, ctx: defaults.ctx },
              stderr: stderrText,
            })
          }
          return yield* err
        }),
      ),
    )

    // 5. Clean up preset file (best-effort)
    const fs = yield* FileSystem.FileSystem
    yield* pipe(fs.remove(presetPath), Effect.catchAll(() => Effect.void))

    return {
      pid: proc.pid,
      port,
      endpoint,
      mode: "router",
      process: proc,
    } satisfies ServerHandle
  })
}
