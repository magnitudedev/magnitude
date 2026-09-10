import { BunContext } from "@effect/platform-bun"
import * as FileSystem from "@effect/platform/FileSystem"
import { Effect, Runtime } from "effect"
import { basename, resolve } from "node:path"
import { sha256File } from "../../src/macos-app"
import { appleSigning, AppleDistributionFailed, signAppleCode } from "./signing"

/**
 * Bun's plugin callback is the platform boundary; dependency copies never mutate node_modules.
 * The compiled executable is returned unsigned: the CLI is signed standalone and the ACN is
 * signed as the main executable of Magnitude.app.
 */
export const compileAppleBun = (entry: string, output: string, target: string, kind: "cli" | "acn") => Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const signing = yield* appleSigning
  const inputs = resolve(output, "..", "apple-inputs", kind)
  yield* fs.remove(inputs, { recursive: true, force: true })
  yield* fs.makeDirectory(inputs, { recursive: true })
  const runtime = yield* Effect.runtime<FileSystem.FileSystem | import("@effect/platform/CommandExecutor").CommandExecutor>()
  const plugin: Bun.BunPlugin = {
    name: "magnitude-signed-native-inputs",
    setup(build) {
      build.onLoad({ filter: /(?:\/rg|\.dylib|\.node)$/ }, (args) => Runtime.runPromise(runtime)(Effect.gen(function* () {
        const digest = yield* sha256File(args.path)
        const name = basename(args.path)
        const directory = resolve(inputs, digest)
        yield* fs.makeDirectory(directory, { recursive: true })
        const staged = resolve(directory, name)
        yield* fs.copyFile(args.path, staged)
        yield* signAppleCode(staged, `dev.magnitude.embedded.${name.replace(/[^a-zA-Z0-9.-]/g, "-")}`, name === "rg" ? "native" : "library")
        return { contents: yield* fs.readFile(staged), loader: "file" as const }
      })))
    },
  }
  const result = yield* Effect.tryPromise({
    try: () => Bun.build({
      entrypoints: [entry],
      target: "bun",
      compile: { target: target as Bun.Build.CompileTarget, outfile: output },
      external: ["electron", "chromium-bidi"],
      define: {
        "process.platform": '"darwin"',
        "process.arch": target.includes("arm64") ? '"arm64"' : '"x64"',
        MAGNITUDE_APPLE_TEAM_ID: `"${signing.team}"`,
      },
      plugins: [plugin],
    }),
    catch: (error) => new AppleDistributionFailed({ message: `Bun compilation failed: ${String(error)}` }),
  })
  if (!result.success) return yield* new AppleDistributionFailed({ message: result.logs.map(String).join("\n") })
  return output
})

// Existing build entrypoints are Promise-based; this is their one adapter into Apple build Effects.
export const runAppleBuild = <A, E>(effect: Effect.Effect<A, E, FileSystem.FileSystem | import("@effect/platform/CommandExecutor").CommandExecutor>) =>
  Effect.runPromise(effect.pipe(Effect.provide(BunContext.layer)))
