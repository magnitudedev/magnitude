import * as Command from "@effect/platform/Command"
import * as CommandExecutor from "@effect/platform/CommandExecutor"
import * as FileSystem from "@effect/platform/FileSystem"
import * as Path from "@effect/platform/Path"
import { BunContext } from "@effect/platform-bun"
import { IcnBinaryIdentity } from "@magnitudedev/icn-protocol"
import { ICN_EXECUTABLE_NAME } from "@magnitudedev/release/executables"
import { releaseBuildEnvironment, type ReleaseHost } from "@magnitudedev/release/targets"
import { Effect, Option, Schema, Stream } from "effect"
import { dirname, resolve } from "node:path"
import { fileURLToPath } from "node:url"
import { stageNvrtc } from "../../packages/release/scripts/build/nvrtc"
import { collectWindowsRuntime } from "../../packages/release/scripts/build/windows-runtime"

export const INFERENCE_ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..")
const NVRTC_CACHE = resolve(INFERENCE_ROOT, "target/nvrtc")

/**
 * `release` is the clean, locked, baseline-CPU build of the host's release target;
 * `development` is an incremental debug build for this machine sharing the workspace target directory.
 */
export type InferenceBuildProfile = "release" | "development"

export interface BuildInferenceInput {
  readonly host: ReleaseHost
  readonly profile: InferenceBuildProfile
  /** Print successful compiler diagnostics, or retain them only for a failed build. */
  readonly diagnostics: "all" | "errors"
}

/** The service binary and the files of its installation's `runtime/` directory. */
export interface InferenceBuild {
  readonly binary: string
  readonly identity: IcnBinaryIdentity
  /** NVRTC on CUDA hosts; the MSVC CRT closure on Windows. */
  readonly runtimeLibraries: readonly string[]
  /** License notices of redistributed runtime libraries (NVRTC). */
  readonly runtimeNotices: readonly string[]
}

export class InferenceBuildFailed extends Schema.TaggedError<InferenceBuildFailed>()("InferenceBuildFailed", {
  message: Schema.String,
}) {}

const CargoMessage = Schema.Struct({
  reason: Schema.String,
  executable: Schema.optionalWith(Schema.NullOr(Schema.String), { as: "Option", exact: true }),
  target: Schema.optionalWith(Schema.Struct({ name: Schema.String }), { as: "Option", exact: true }),
  message: Schema.optionalWith(
    Schema.Struct({ rendered: Schema.optionalWith(Schema.NullOr(Schema.String), { as: "Option", exact: true }) }),
    { as: "Option", exact: true },
  ),
})
type CargoMessage = typeof CargoMessage.Type
const decodeCargoMessage = Schema.decodeUnknown(Schema.parseJson(CargoMessage))

const renderedDiagnostic = (message: CargoMessage): Option.Option<string> =>
  message.reason === "compiler-message"
    ? message.message.pipe(Option.flatMap((value) => value.rendered), Option.flatMap(Option.fromNullable))
    : Option.none()

/** The executables Cargo reports for `name`. */
export const cargoExecutables = (messages: readonly CargoMessage[], name: string): readonly string[] =>
  messages.flatMap((message) =>
    message.reason === "compiler-artifact" && Option.exists(message.target, (target) => target.name === name)
      ? message.executable.pipe(Option.flatMap(Option.fromNullable), Option.toArray)
      : [])

/** Decodes Cargo's JSON message stream, forwarding rendered diagnostics as they arrive. */
export const readCargoMessages = (
  lines: Stream.Stream<string, InferenceBuildFailed>,
  writeDiagnostic: (rendered: string) => Effect.Effect<void>,
): Effect.Effect<{ readonly messages: readonly CargoMessage[]; readonly diagnostics: readonly string[] }, InferenceBuildFailed> =>
  lines.pipe(
    Stream.filter((line) => line.trim().length > 0),
    Stream.mapEffect((line) => decodeCargoMessage(line).pipe(
      Effect.mapError(() => new InferenceBuildFailed({ message: `Cargo emitted a malformed message: ${line.slice(0, 200)}` })),
    )),
    Stream.tap((message) => Option.match(renderedDiagnostic(message), { onNone: () => Effect.void, onSome: writeDiagnostic })),
    Stream.runCollect,
    Effect.map((chunk) => {
      const messages = [...chunk]
      return { messages, diagnostics: messages.flatMap((message) => Option.toArray(renderedDiagnostic(message))) }
    }),
  )

const writeStderr = (text: string) => Effect.sync(() => { process.stderr.write(text) })

const runCargoBuild = (
  arguments_: readonly string[],
  /** Added to the inherited environment. */
  environment: Readonly<Record<string, string>>,
  diagnostics: BuildInferenceInput["diagnostics"],
): Effect.Effect<readonly CargoMessage[], InferenceBuildFailed, CommandExecutor.CommandExecutor> =>
  Effect.scoped(Effect.gen(function* () {
    const child = yield* Command.make("cargo", ...arguments_).pipe(
      Command.workingDirectory(INFERENCE_ROOT),
      Command.env(environment),
      Command.start,
    )
    const [cargo, stderr, exitCode] = yield* Effect.all([
      readCargoMessages(
        child.stdout.pipe(Stream.decodeText(), Stream.splitLines, Stream.mapError(() => new InferenceBuildFailed({ message: "Cargo output failed" }))),
        (rendered) => diagnostics === "all" ? writeStderr(rendered) : Effect.void,
      ),
      child.stderr.pipe(
        Stream.decodeText(),
        Stream.tap((text) => diagnostics === "all" ? writeStderr(text) : Effect.void),
        Stream.runFold("", (all, text) => all + text),
      ),
      child.exitCode,
    ], { concurrency: "unbounded" })
    if (exitCode !== 0) {
      return yield* new InferenceBuildFailed({
        message: `cargo build failed with exit ${exitCode}:\n${[stderr, ...cargo.diagnostics].join("\n").trim().slice(-8_000)}`,
      })
    }
    return cargo.messages
  })).pipe(Effect.mapError((cause) =>
    cause instanceof InferenceBuildFailed ? cause : new InferenceBuildFailed({ message: `cargo build failed: ${String(cause)}` })))

const readIdentity = (binary: string) =>
  Command.make(binary, "version", "--json").pipe(
    Command.env({ LD_LIBRARY_PATH: "", DYLD_LIBRARY_PATH: "" }),
    Command.string,
    Effect.flatMap(Schema.decodeUnknown(Schema.parseJson(IcnBinaryIdentity))),
    Effect.mapError((cause) => new InferenceBuildFailed({ message: `the identity probe of ${binary} failed: ${String(cause)}` })),
  )

const windowsRuntime = (files: readonly string[]) => Effect.gen(function* () {
  const redistributable = process.env.VCToolsRedistDir
  if (!redistributable) {
    return yield* new InferenceBuildFailed({ message: "Windows builds require the Visual Studio compiler environment (VCToolsRedistDir)" })
  }
  // Seismic loads nvcuda.dll and vulkan-1.dll at runtime; neither may be an import.
  return yield* collectWindowsRuntime({
    files,
    redistributable: resolve(redistributable, "x64", "Microsoft.VC143.CRT"),
  }).pipe(Effect.mapError((cause) => new InferenceBuildFailed({ message: `Windows runtime closure failed: ${cause.message}` })))
})

export const buildInference = ({
  host,
  profile,
  diagnostics,
}: BuildInferenceInput): Effect.Effect<InferenceBuild, InferenceBuildFailed, CommandExecutor.CommandExecutor | FileSystem.FileSystem | Path.Path> =>
  Effect.gen(function* () {
    const fs = yield* FileSystem.FileSystem
    const release = profile === "release"
    const targetDirectory = resolve(INFERENCE_ROOT, "target", `release-${host.id}`)
    if (release) {
      yield* fs.remove(targetDirectory, { recursive: true, force: true }).pipe(
        Effect.mapError((cause) => new InferenceBuildFailed({ message: `unable to clean ${targetDirectory}: ${String(cause)}` })),
      )
    }
    const messages = yield* runCargoBuild([
      "build",
      ...(release ? ["--release", "--locked", "--target", host.rustTarget] : []),
      "-p",
      "magnitude-service-server",
      "--bin",
      ICN_EXECUTABLE_NAME,
      "--message-format",
      "json-render-diagnostics",
    ], release
      ? {
        ...releaseBuildEnvironment(host),
        CARGO_TARGET_DIR: targetDirectory,
        // Empty so ambient configuration cannot raise the CPU baseline (no `target-cpu`). The
        // server's build script owns the Linux `../runtime` rpath.
        CARGO_ENCODED_RUSTFLAGS: "",
      }
      : {}, diagnostics)
    const executables = cargoExecutables(messages, ICN_EXECUTABLE_NAME)
    if (executables.length !== 1) {
      return yield* new InferenceBuildFailed({ message: `Cargo reported ${executables.length} ${ICN_EXECUTABLE_NAME} executables` })
    }
    const binary = executables[0]!
    const identity = yield* readIdentity(binary)
    const nvrtc = yield* Option.match(host.nvrtc, {
      onNone: () => Effect.succeed({ libraries: [], notices: [] }),
      onSome: (redistributable) => stageNvrtc(redistributable, NVRTC_CACHE).pipe(
        Effect.map((staged) => ({ libraries: staged.libraries, notices: [staged.license] })),
        Effect.mapError((cause) => new InferenceBuildFailed({ message: cause.message })),
      ),
    })
    const crt = host.id.startsWith("windows-")
      ? yield* windowsRuntime([binary, ...nvrtc.libraries])
      : []
    return { binary, identity, runtimeLibraries: [...nvrtc.libraries, ...crt], runtimeNotices: nvrtc.notices }
  })

/** Promise entry point for the release build scripts. */
export const buildInferenceBinary = (input: BuildInferenceInput): Promise<InferenceBuild> =>
  Effect.runPromise(buildInference(input).pipe(Effect.provide(BunContext.layer)))
