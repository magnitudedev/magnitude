import { downloadFileToCacheDir, snapshotDownload } from "@huggingface/hub"
import * as Command from "@effect/platform/Command"
import * as CommandExecutor from "@effect/platform/CommandExecutor"
import * as FileSystem from "@effect/platform/FileSystem"
import { Data, Effect, Schema, Stream } from "effect"
import { arch, cpus, hostname, platform, release, totalmem } from "node:os"
import { dirname, join, resolve } from "node:path"
import { prepareCorpus } from "./corpus"
import type { ModelIdentity } from "./domain"
import {
  type ExperimentDefinition,
  loadExperiment,
  resolveExperimentPaths,
  type VariantId,
} from "./experiment"
import { digestObject } from "./hash"
import { hashFileSha256, resolveModelIdentity } from "./model"
import { resolveIcnExecutable, resolveLlamaCppExecutable } from "./target"

const NonEmptyString = Schema.String.pipe(Schema.minLength(1))
const PositiveInt = Schema.Int.pipe(Schema.greaterThan(0))
const Sha256 = Schema.String.pipe(Schema.pattern(/^[a-f0-9]{64}$/))
const MlxArtifactLockSchema = Schema.Struct({
  repository: NonEmptyString,
  revision: NonEmptyString,
  quantization: Schema.Union(
    Schema.Struct({ family: Schema.Literal("mlx-affine"), bits: PositiveInt, groupSize: PositiveInt }),
    Schema.Struct({ family: Schema.Literal("mlx-unquantized"), dtype: Schema.Literal("bfloat16", "float16") }),
  ),
  files: Schema.NonEmptyArray(Schema.Struct({
    path: NonEmptyString,
    sizeBytes: PositiveInt,
    sha256: Sha256,
  })),
})
export interface PreparedArtifact {
  readonly variantId: VariantId
  readonly role: "target" | "drafter"
  readonly kind: "gguf" | "mlx"
  readonly path: string
  readonly repository: string
  readonly revision: string
  readonly quantization: string
  readonly digest: string
  readonly manifestPath?: string
  readonly manifestDigest?: string
  readonly tokenizerQualification?: string
}

export interface PreparedExperiment {
  readonly version: 4
  readonly experimentPath: string
  readonly experiment: ExperimentDefinition
  readonly preparedAt: string
  readonly corpusRoot: string
  readonly corpusDigest: string
  readonly planModel: ModelIdentity
  readonly artifacts: readonly PreparedArtifact[]
  readonly engines: readonly PreparedEngineEnvironment[]
  readonly host: {
    readonly hostname: string
    readonly platform: string
    readonly release: string
    readonly arch: string
    readonly cpu: string
    readonly logicalCpus: number
    readonly totalMemoryBytes: number
  }
  readonly digest: string
}

export type PreparedEngineEnvironment =
  | { readonly variantId: VariantId; readonly kind: "mlx-lm"; readonly pythonProject: string; readonly lockDigest: string; readonly adapterDigest: string; readonly version: string }
  | { readonly variantId: VariantId; readonly kind: "mlx-vlm"; readonly pythonProject: string; readonly lockDigest: string; readonly version: string }
  | { readonly variantId: VariantId; readonly kind: "llama.cpp"; readonly executable: string; readonly version: string }
  | { readonly variantId: VariantId; readonly kind: "icn"; readonly executable: string; readonly version: string }
  | { readonly variantId: VariantId; readonly kind: "existing-endpoint"; readonly endpoint: string }

export class PreparationError extends Data.TaggedError("PreparationError")<{
  readonly operation: string
  readonly message: string
}> {}

const accessToken = () => process.env.HF_TOKEN?.trim() || process.env.HUGGING_FACE_HUB_TOKEN?.trim() || undefined

const writeJson = (path: string, value: unknown) => Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  yield* fs.makeDirectory(dirname(path), { recursive: true })
  const encoded = yield* Schema.encode(Schema.parseJson(Schema.Unknown, { space: 2 }))(value)
  yield* fs.writeFileString(path, `${encoded}\n`)
})

const verifyFile = (path: string, sizeBytes: number, sha256: string) => Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const stat = yield* fs.stat(path)
  if (stat.type !== "File" || Number(stat.size) !== sizeBytes) {
    return yield* new PreparationError({ operation: "verify-artifact", message: `${path} does not match declared size ${sizeBytes}` })
  }
  const actual = yield* hashFileSha256(path).pipe(Effect.mapError((error) => new PreparationError({ operation: "verify-artifact", message: error.message })))
  if (actual !== sha256) {
    return yield* new PreparationError({ operation: "verify-artifact", message: `${path} digest mismatch: expected ${sha256}, received ${actual}` })
  }
})

const readMlxLock = (path: string) => Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const text = yield* fs.readFileString(path)
  return yield* Schema.decodeUnknown(Schema.parseJson(MlxArtifactLockSchema))(text).pipe(
    Effect.mapError((error) => new PreparationError({ operation: "read-mlx-lock", message: String(error) })),
  )
})

const listRelativeFiles = (root: string, directory = root): Effect.Effect<readonly string[], import("@effect/platform/Error").PlatformError, FileSystem.FileSystem> =>
  Effect.gen(function* () {
    const fs = yield* FileSystem.FileSystem
    const entries = yield* fs.readDirectory(directory)
    const nested = yield* Effect.forEach(entries.sort(), (entry) => Effect.gen(function* () {
      const absolute = join(directory, entry)
      const stat = yield* fs.stat(absolute)
      return stat.type === "Directory" ? yield* listRelativeFiles(root, absolute) : [absolute.slice(root.length + 1)]
    }), { concurrency: 1 })
    return nested.flat()
  })

interface CreateMlxArtifactLockBaseOptions {
  readonly repository: string
  readonly revision: string
  readonly output: string
}
export type CreateMlxArtifactLockOptions = CreateMlxArtifactLockBaseOptions & (
  | { readonly bits: number; readonly groupSize: number }
  | { readonly dtype: "bfloat16" | "float16" }
)

export const createMlxArtifactLock = (options: CreateMlxArtifactLockOptions) => Effect.gen(function* () {
  if ("bits" in options && (!Number.isSafeInteger(options.bits) || options.bits <= 0 || !Number.isSafeInteger(options.groupSize) || options.groupSize <= 0)) {
    return yield* new PreparationError({ operation: "lock-mlx", message: "bits and group size must be positive integers" })
  }
  const fs = yield* FileSystem.FileSystem
  const root = yield* Effect.tryPromise({
    try: () => snapshotDownload({
      repo: { type: "model", name: options.repository },
      revision: options.revision,
      ...(accessToken() ? { accessToken: accessToken() } : {}),
    }),
    catch: (error) => new PreparationError({
      operation: "download-mlx",
      message: error instanceof Error ? error.message : String(error),
    }),
  })
  const paths = yield* listRelativeFiles(root)
  const files = yield* Effect.forEach(paths, (path) => Effect.gen(function* () {
    const absolute = join(root, path)
    const stat = yield* fs.stat(absolute)
    if (stat.type !== "File") return yield* new PreparationError({ operation: "lock-mlx", message: `${absolute} is not a file` })
    const sha256 = yield* hashFileSha256(absolute).pipe(
      Effect.mapError((error) => new PreparationError({ operation: "lock-mlx", message: error.message })),
    )
    return { path, sizeBytes: Number(stat.size), sha256 }
  }), { concurrency: 4 })
  const lock = {
    repository: options.repository,
    revision: options.revision,
    quantization: "bits" in options
      ? { family: "mlx-affine" as const, bits: options.bits, groupSize: options.groupSize }
      : { family: "mlx-unquantized" as const, dtype: options.dtype },
    files,
  }
  yield* writeJson(resolve(options.output), lock)
  return { output: resolve(options.output), root, files: files.length }
}).pipe(Effect.mapError((error) => error instanceof PreparationError ? error : new PreparationError({
  operation: "lock-mlx", message: error instanceof Error ? error.message : String(error),
})))

const runString = (executable: string, args: readonly string[], cwd?: string) => Effect.scoped(Effect.gen(function* () {
  const command = Command.make(executable, ...args).pipe(cwd ? Command.workingDirectory(cwd) : (value) => value)
  const child = yield* Command.start(command)
  const [stdout, stderr, code] = yield* Effect.all([
    child.stdout.pipe(Stream.decodeText(), Stream.runFold("", (output, chunk) => output + chunk)),
    child.stderr.pipe(Stream.decodeText(), Stream.runFold("", (output, chunk) => output + chunk)),
    child.exitCode,
  ], { concurrency: "unbounded" })
  if (code !== 0) {
    return yield* new PreparationError({
      operation: `run-${executable}`,
      message: `${executable} exited with ${code}: ${stderr.trim() || stdout.trim()}`,
    })
  }
  return [stdout.trim(), stderr.trim()].filter(Boolean).join("\n")
})).pipe(Effect.mapError((error) => error instanceof PreparationError ? error : new PreparationError({
  operation: `run-${executable}`, message: error instanceof Error ? error.message : String(error),
})))

const prepareUv = (pythonProject: string) => Effect.gen(function* () {
  yield* runString("uv", ["sync", "--frozen", "--project", pythonProject])
  const version = yield* runString("uv", [
    "run", "--frozen", "--no-sync", "--project", pythonProject,
    "python", "-c", "import importlib.metadata; print(importlib.metadata.version('mlx-lm'))",
  ])
  const lockDigest = yield* hashFileSha256(join(pythonProject, "uv.lock")).pipe(
    Effect.mapError((error) => new PreparationError({ operation: "hash-uv-lock", message: error.message })),
  )
  const sourceRoot = join(pythonProject, "src")
  const sourcePaths = yield* listRelativeFiles(sourceRoot).pipe(
    Effect.mapError((error) => new PreparationError({ operation: "hash-adapter", message: String(error) })),
  )
  const sourceFiles = yield* Effect.forEach(sourcePaths, (path) => hashFileSha256(join(sourceRoot, path)).pipe(
    Effect.map((sha256) => ({ path, sha256 })),
    Effect.mapError((error) => new PreparationError({ operation: "hash-adapter", message: error.message })),
  ), { concurrency: 4 })
  const pyprojectDigest = yield* hashFileSha256(join(pythonProject, "pyproject.toml")).pipe(
    Effect.mapError((error) => new PreparationError({ operation: "hash-adapter", message: error.message })),
  )
  const adapterDigest = digestObject({ pyprojectDigest, sourceFiles })
  return { version, lockDigest, adapterDigest }
})

const prepareMlxVlmUv = (pythonProject: string) => Effect.gen(function* () {
  yield* runString("uv", ["sync", "--frozen", "--project", pythonProject])
  const version = yield* runString("uv", [
    "run", "--frozen", "--no-sync", "--project", pythonProject,
    "python", "-c", "import importlib.metadata; print(importlib.metadata.version('mlx-vlm'))",
  ])
  const lockDigest = yield* hashFileSha256(join(pythonProject, "uv.lock")).pipe(
    Effect.mapError((error) => new PreparationError({ operation: "hash-uv-lock", message: error.message })),
  )
  return { version, lockDigest }
})

const prepareGguf = (variantId: VariantId, role: PreparedArtifact["role"], artifact: Extract<ExperimentDefinition["variants"][number]["artifact"], { kind: "gguf" }>) =>
  Effect.gen(function* () {
    const path = yield* Effect.tryPromise({
      try: () => downloadFileToCacheDir({
        repo: { type: "model", name: artifact.repository }, revision: artifact.revision,
        path: artifact.file, ...(accessToken() ? { accessToken: accessToken() } : {}),
      }),
      catch: (error) => new PreparationError({ operation: "download-gguf", message: error instanceof Error ? error.message : String(error) }),
    })
    yield* verifyFile(path, artifact.sizeBytes, artifact.sha256)
    return {
      variantId, role, kind: "gguf" as const, path, repository: artifact.repository,
      revision: artifact.revision, quantization: artifact.quantization.scheme, digest: artifact.sha256,
    }
  })

const prepareMlx = (variantId: VariantId, role: PreparedArtifact["role"], artifact: Extract<ExperimentDefinition["variants"][number]["artifact"], { kind: "mlx" }>) =>
  Effect.gen(function* () {
    const lock = yield* readMlxLock(artifact.manifest)
    if (
      lock.repository !== artifact.repository
      || lock.revision !== artifact.revision
      || lock.quantization.family !== artifact.quantization.family
      || (lock.quantization.family === "mlx-affine" && artifact.quantization.family === "mlx-affine" && (
        lock.quantization.bits !== artifact.quantization.bits || lock.quantization.groupSize !== artifact.quantization.groupSize
      ))
      || (lock.quantization.family === "mlx-unquantized" && artifact.quantization.family === "mlx-unquantized" && lock.quantization.dtype !== artifact.quantization.dtype)
    ) {
      return yield* new PreparationError({ operation: "verify-mlx-lock", message: `${artifact.manifest} identity does not match the experiment` })
    }
    const path = yield* Effect.tryPromise({
      try: () => snapshotDownload({
        repo: { type: "model", name: artifact.repository }, revision: artifact.revision,
        ...(accessToken() ? { accessToken: accessToken() } : {}),
      }),
      catch: (error) => new PreparationError({ operation: "download-mlx", message: error instanceof Error ? error.message : String(error) }),
    })
    yield* Effect.forEach(lock.files, (file) => verifyFile(join(path, file.path), file.sizeBytes, file.sha256), { concurrency: 4 })
    const manifestDigest = yield* hashFileSha256(artifact.manifest).pipe(
      Effect.mapError((error) => new PreparationError({ operation: "hash-mlx-lock", message: error.message })),
    )
    return {
      variantId, role, kind: "mlx" as const, path, repository: artifact.repository,
      revision: artifact.revision,
      quantization: artifact.quantization.family === "mlx-affine"
        ? `${artifact.quantization.bits}-bit/group-${artifact.quantization.groupSize}`
        : artifact.quantization.dtype,
      digest: digestObject(lock), manifestPath: artifact.manifest, manifestDigest,
    }
  })

export const preparedExperimentPath = (experimentId: string) => resolve("benchmark-results", "prepared", `${experimentId}.json`)

const prepareArtifact = (
  variant: ExperimentDefinition["variants"][number],
  role: PreparedArtifact["role"],
  artifact = variant.artifact,
): Effect.Effect<PreparedArtifact, PreparationError | import("@effect/platform/Error").PlatformError, FileSystem.FileSystem> =>
  artifact.kind === "gguf" ? prepareGguf(variant.id, role, artifact) : prepareMlx(variant.id, role, artifact)

const prepareEngine = (variant: ExperimentDefinition["variants"][number]): Effect.Effect<PreparedEngineEnvironment, PreparationError, FileSystem.FileSystem | CommandExecutor.CommandExecutor> => {
  const engine = variant.engine
  if (engine.kind === "mlx-lm") {
    return prepareUv(engine.pythonProject).pipe(Effect.map(({ version, lockDigest, adapterDigest }) => ({
      variantId: variant.id,
      kind: engine.kind,
      pythonProject: engine.pythonProject,
      version,
      lockDigest,
      adapterDigest,
    })))
  }
  if (engine.kind === "mlx-vlm") {
    return prepareMlxVlmUv(engine.pythonProject).pipe(Effect.map(({ version, lockDigest }) => ({
      variantId: variant.id,
      kind: engine.kind,
      pythonProject: engine.pythonProject,
      version,
      lockDigest,
    })))
  }
  if (engine.kind === "existing-endpoint") {
    return Effect.succeed({ variantId: variant.id, kind: engine.kind, endpoint: engine.endpoint })
  }
  if (engine.kind === "llama.cpp") {
    return resolveLlamaCppExecutable(engine.executable === "managed" ? undefined : engine.executable).pipe(
      Effect.mapError((error) => new PreparationError({ operation: error.operation, message: error.message })),
      Effect.flatMap((executable) => runString(executable, ["--version"]).pipe(Effect.map((version) => ({
        variantId: variant.id,
        kind: engine.kind,
        executable,
        version,
      })))),
    )
  }
  return resolveIcnExecutable(engine.executable === "managed" ? undefined : engine.executable).pipe(
    Effect.mapError((error) => new PreparationError({ operation: error.operation, message: error.message })),
    Effect.flatMap((executable) => runString(executable, ["version", "--json"]).pipe(Effect.map((version) => ({
      variantId: variant.id,
      kind: engine.kind,
      executable,
      version,
    })))),
  )
}

export const prepareExperiment = (experimentPath: string): Effect.Effect<PreparedExperiment, PreparationError, FileSystem.FileSystem | CommandExecutor.CommandExecutor> =>
  Effect.gen(function* () {
    const loaded = yield* loadExperiment(experimentPath).pipe(Effect.mapError((error) => new PreparationError({ operation: error.operation, message: error.message })))
    const experiment = resolveExperimentPaths(loaded, experimentPath)
    const engines = yield* Effect.forEach(experiment.variants, prepareEngine, { concurrency: 1 })
    const downloadedArtifacts = yield* Effect.forEach(experiment.variants, (variant) => prepareArtifact(variant, "target"), { concurrency: 1 })
    const artifacts = yield* Effect.forEach(downloadedArtifacts, (artifact) => {
      if (artifact.kind !== "mlx") return Effect.succeed(artifact)
      const variant = experiment.variants.find(({ id }) => id === artifact.variantId)
      if (!variant) {
        return Effect.fail(new PreparationError({ operation: "qualify-tokenizer", message: `missing engine for ${artifact.variantId}` }))
      }
      if (variant.engine.kind === "mlx-vlm") {
        return Effect.succeed({
          ...artifact,
          tokenizerQualification: "MLX-VLM tokenizer, tool parser, and MTP drafter are qualified by managed readiness before measurement",
        })
      }
      if (variant.engine.kind !== "mlx-lm") {
        return Effect.fail(new PreparationError({ operation: "qualify-tokenizer", message: `missing MLX engine for ${artifact.variantId}` }))
      }
      return runString("uv", [
        "run", "--frozen", "--no-sync", "--project", variant.engine.pythonProject,
        "magnitude-mlx-benchmark-qualify", "--model", artifact.path,
      ]).pipe(Effect.map((tokenizerQualification) => ({ ...artifact, tokenizerQualification })))
    }, { concurrency: 1 })
    const draftArtifacts = yield* Effect.forEach(experiment.variants, (variant) => {
      const engine = variant.engine
      return (engine.kind === "llama.cpp" || engine.kind === "mlx-vlm") && engine.speculativeDecoding.kind === "mtp"
        ? prepareArtifact(variant, "drafter", engine.speculativeDecoding.draftArtifact)
        : Effect.succeed(null)
    }, { concurrency: 1 }).pipe(Effect.map((values) => values.filter((value): value is PreparedArtifact => value !== null)))
    const allArtifacts = [...artifacts, ...draftArtifacts]
    const gguf = allArtifacts.find((artifact) => artifact.role === "target" && artifact.kind === "gguf")
    if (!gguf) return yield* new PreparationError({ operation: "plan-model", message: "experiment requires a GGUF artifact for plan metadata" })
    const planModel = yield* resolveModelIdentity({
      id: experiment.variants[0]!.artifact.modelId,
      artifactPath: gguf.path,
      verifiedArtifactSha256: gguf.digest,
      maxContextTokens: experiment.requestPolicy.contextTokensPerSequence,
    }).pipe(Effect.mapError((error) => new PreparationError({ operation: error.operation, message: error.message })))
    const corpus = yield* prepareCorpus().pipe(Effect.mapError((error) => new PreparationError({ operation: "prepare-corpus", message: String(error) })))
    const identity = {
      version: 4 as const,
      experimentPath: resolve(experimentPath), experiment, preparedAt: new Date().toISOString(),
      corpusRoot: corpus.root, corpusDigest: corpus.digest, planModel, artifacts: allArtifacts,
      engines,
      host: {
        hostname: hostname(), platform: platform(), release: release(), arch: arch(),
        cpu: cpus()[0]?.model ?? "unknown", logicalCpus: cpus().length, totalMemoryBytes: totalmem(),
      },
    }
    const prepared: PreparedExperiment = { ...identity, digest: digestObject(identity) }
    yield* writeJson(preparedExperimentPath(experiment.id), prepared)
    return prepared
  }).pipe(Effect.mapError((error) => error instanceof PreparationError ? error : new PreparationError({
    operation: "prepare", message: error instanceof Error ? error.message : String(error),
  })))

export const loadPreparedExperiment = (experimentId: string): Effect.Effect<PreparedExperiment, PreparationError, FileSystem.FileSystem> =>
  Effect.gen(function* () {
    const path = preparedExperimentPath(experimentId)
    const fs = yield* FileSystem.FileSystem
    const text = yield* fs.readFileString(path).pipe(Effect.mapError(() => new PreparationError({
      operation: "load-prepared", message: `experiment is not prepared; run bun benchmark prepare <experiment.ts>`,
    })))
    const prepared = yield* Effect.try({
      try: () => JSON.parse(text) as PreparedExperiment,
      catch: (error) => new PreparationError({ operation: "load-prepared", message: error instanceof Error ? error.message : String(error) }),
    })
    if (prepared.version !== 4) {
      return yield* new PreparationError({ operation: "load-prepared", message: "prepared experiment format changed; run prepare again" })
    }
    const { digest, ...identity } = prepared
    if (typeof digest !== "string" || digestObject(identity) !== digest) {
      return yield* new PreparationError({ operation: "load-prepared", message: "prepared experiment digest is invalid; run prepare again" })
    }
    return prepared
  })
