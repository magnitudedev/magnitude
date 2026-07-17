import * as BunContext from "@effect/platform-bun/BunContext"
import * as FetchHttpClient from "@effect/platform/FetchHttpClient"
import * as FileSystem from "@effect/platform/FileSystem"
import * as Path from "@effect/platform/Path"
import { Effect, Option } from "effect"
import { describe, expect, it } from "vitest"
import {
  DEFAULT_LLAMA_DISTRIBUTION_MANIFEST,
  LlamaBuildNumber,
  makeLlamaCppInstallationRegistry,
} from "."

const makeExecutable = (
  directory: string,
  build: number,
): Effect.Effect<string, unknown, FileSystem.FileSystem | Path.Path> => Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const path = yield* Path.Path
  yield* fs.makeDirectory(directory, { recursive: true })
  const server = path.join(directory, "llama-server")
  const fitParams = path.join(directory, "llama-fit-params")
  const contents = `#!/bin/sh\nprintf 'llama.cpp build ${build}\\n'\n`
  yield* fs.writeFileString(server, contents)
  yield* fs.writeFileString(fitParams, contents)
  yield* fs.chmod(server, 0o755)
  yield* fs.chmod(fitParams, 0o755)
  return server
})

describe("llama.cpp installation registry", () => {
  it("discovers every source and selects the first supported source by precedence", async () => {
    const result = await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const fs = yield* FileSystem.FileSystem
      const path = yield* Path.Path
      const root = yield* fs.makeTempDirectoryScoped()
      const configured = yield* makeExecutable(path.join(root, "configured"), 8680)
      const firstPath = yield* makeExecutable(path.join(root, "path-first"), 9000)
      yield* makeExecutable(path.join(root, "path-second"), 10011)
      const registry = yield* makeLlamaCppInstallationRegistry({
        configuredServerExecutable: Option.some(configured),
        managedRoot: path.join(root, "managed"),
        searchPath: [path.dirname(firstPath), path.join(root, "path-second")],
        managedVariant: Option.none(),
        manifest: DEFAULT_LLAMA_DISTRIBUTION_MANIFEST,
        platform: process.platform,
        nativeArchitecture: process.arch,
      })
      const snapshot = yield* registry.snapshot
      const selected = yield* registry.selected
      return { snapshot, selected }
    }).pipe(Effect.provide([BunContext.layer, FetchHttpClient.layer]))))

    expect(result.snapshot.installations.map((installation) => installation.build)).toEqual([
      LlamaBuildNumber.make(8680),
      LlamaBuildNumber.make(9000),
      LlamaBuildNumber.make(10011),
    ])
    expect(result.selected.build).toBe(9000)
    expect(result.selected.discoveries).toEqual([
      expect.objectContaining({ _tag: "Path", priority: 0 }),
    ])
    expect(Option.getOrThrow(result.snapshot.selectedInstallationId)).toBe(result.selected.id)
  })

  it("reports outdated when installations exist but none meets the minimum build", async () => {
    const reason = await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const fs = yield* FileSystem.FileSystem
      const path = yield* Path.Path
      const root = yield* fs.makeTempDirectoryScoped()
      const configured = yield* makeExecutable(path.join(root, "configured"), 8680)
      const registry = yield* makeLlamaCppInstallationRegistry({
        configuredServerExecutable: Option.some(configured),
        managedRoot: path.join(root, "managed"),
        searchPath: [],
        managedVariant: Option.none(),
        manifest: DEFAULT_LLAMA_DISTRIBUTION_MANIFEST,
        platform: process.platform,
        nativeArchitecture: process.arch,
      })
      return yield* registry.selected.pipe(Effect.flip, Effect.map((failure) => failure.reason))
    }).pipe(Effect.provide([BunContext.layer, FetchHttpClient.layer]))))

    expect(reason).toBe("outdated")
  })
})
