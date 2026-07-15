import { Context, Effect, Layer, pipe } from "effect"
import * as Command from "@effect/platform/Command"
import * as CommandExecutor from "@effect/platform/CommandExecutor"
import * as HttpClient from "@effect/platform/HttpClient"
import { BunCommandExecutor, BunFileSystem } from "@effect/platform-bun"
import { FetchHttpClient } from "@effect/platform"
import { fingerprintServer } from "./fingerprint"
import type { DetectedServer } from "./types"

// ── Service Tag (internal) ──

export interface LlamaCppDetectorApi {
  readonly detect: () => Effect.Effect<
    readonly DetectedServer[],
    never,
    HttpClient.HttpClient | CommandExecutor.CommandExecutor
  >
}

export class LlamaCppDetector extends Context.Tag("LlamaCppDetector")<
  LlamaCppDetector,
  LlamaCppDetectorApi
>() {}

// ── Platform layer (baked in) ──

const PlatformLayer = Layer.mergeAll(
  BunCommandExecutor.layer,
  FetchHttpClient.layer,
).pipe(Layer.provideMerge(BunFileSystem.layer))

// ── Factory ──

export interface LlamaCppDetectorDeps {
  readonly configuredEndpoint?: string
  readonly managedPids?: readonly number[]
}

export function makeLlamaCppDetector(
  deps: LlamaCppDetectorDeps = {},
): LlamaCppDetectorApi {
  const detect: LlamaCppDetectorApi["detect"] = () =>
    detectInstances(deps).pipe(Effect.provide(PlatformLayer))

  return { detect }
}

// ── Detection implementation ──

const COMMON_PORTS = [
  8080, 8081, 8082, 8000, 8001, 11434, 1234, 5000, 5001,
]

function detectInstances(
  deps: LlamaCppDetectorDeps,
): Effect.Effect<
  readonly DetectedServer[],
  never,
  HttpClient.HttpClient | CommandExecutor.CommandExecutor
> {
  return Effect.gen(function* () {
    const processPorts = yield* detectProcessPorts()

    const candidates: string[] = []
    const seen = new Set<string>()

    const addCandidate = (endpoint: string) => {
      if (!seen.has(endpoint)) {
        seen.add(endpoint)
        candidates.push(endpoint)
      }
    }

    if (deps.configuredEndpoint) addCandidate(deps.configuredEndpoint)
    const envEndpoint = process.env.LLAMA_SERVER_ENDPOINT?.trim()
    if (envEndpoint) addCandidate(envEndpoint)
    for (const port of processPorts) {
      addCandidate(`http://127.0.0.1:${port}`)
    }
    for (const port of COMMON_PORTS) {
      addCandidate(`http://127.0.0.1:${port}`)
    }

    const probeOne = (endpoint: string) =>
      pipe(
        fingerprintServer(endpoint),
        Effect.timeout("2 seconds"),
        Effect.catchAll(() => Effect.succeed(null)),
      )

    const results = yield* Effect.all(candidates.map(probeOne), {
      concurrency: 8,
    })

    return results.filter(
      (r): r is DetectedServer => r !== null,
    )
  })
}

function detectProcessPorts(): Effect.Effect<
  readonly number[],
  never,
  CommandExecutor.CommandExecutor
> {
  return Effect.gen(function* () {
    const pgrepResult = yield* pipe(
      Command.string(Command.make("pgrep", "-a", "llama-server")),
      Effect.catchAll(() => Effect.succeed("")),
    )
    const pids = pgrepResult
      .split("\n")
      .map((line) => line.trim().split(/\s+/)[0])
      .filter((pid) => pid && /^\d+$/.test(pid))
      .map(Number)

    if (pids.length === 0) return []

    const portFinder = process.platform === "darwin" ? "lsof" : "ss"
    const ports: number[] = []

    if (portFinder === "lsof") {
      for (const pid of pids) {
        const result = yield* pipe(
          Command.string(Command.make("lsof", "-iTCP", "-sTCP:LISTEN", "-P", "-n", "-p", String(pid))),
          Effect.catchAll(() => Effect.succeed("")),
        )
        for (const line of result.split("\n")) {
          const m = line.match(/:(\d+)\s+LISTEN/)
          if (m) ports.push(Number(m[1]))
        }
      }
    } else {
      const result = yield* pipe(
        Command.string(Command.make("ss", "-ltnp")),
        Effect.catchAll(() => Effect.succeed("")),
      )
      for (const line of result.split("\n")) {
        if (pids.some((pid) => line.includes(`pid=${pid}`))) {
          const m = line.match(/:(\d+)\s/)
          if (m) ports.push(Number(m[1]))
        }
      }
    }

    return [...new Set(ports)]
  })
}
