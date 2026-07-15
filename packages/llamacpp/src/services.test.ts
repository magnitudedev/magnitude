import { chmod, mkdtemp, mkdir, readFile, readdir, rm, stat, writeFile } from "node:fs/promises"
import { createHash } from "node:crypto"
import { execFile } from "node:child_process"
import { tmpdir } from "node:os"
import { join } from "node:path"
import { promisify } from "node:util"
import { afterEach, describe, expect, it } from "vitest"
import { Effect, Fiber, Layer, Option, Stream } from "effect"
import * as HttpClient from "@effect/platform/HttpClient"
import * as HttpClientResponse from "@effect/platform/HttpClientResponse"
import { BunCommandExecutor, BunFileSystem, BunPath } from "@effect/platform-bun"
import { FetchHttpClient } from "@effect/platform"
import {
  LlamaCppDistribution,
  LlamaCppDistributionLive,
  LlamaCppModelStore,
  LlamaCppModelStoreLive,
  LlamaCppRuntime,
  LlamaCppRuntimeLive,
  type LlamaCppDistributionApi,
  type LlamaCppReleaseAsset,
  type LlamaCppModelStoreApi,
} from "./index"
import { parseRuntimeDevices } from "./host"

const roots: string[] = []
const execFileAsync = promisify(execFile)
afterEach(async () => {
  await Promise.all(roots.splice(0).map((root) => rm(root, { recursive: true, force: true })))
})

const responseClient = (respond: (url: URL, method: string) => Response): HttpClient.HttpClient =>
  HttpClient.make((request) => Effect.succeed(HttpClientResponse.fromWeb(
    request,
    respond(new URL(request.url), request.method),
  )))

const requestClient = (respond: (request: globalThis.Request) => Response): HttpClient.HttpClient =>
  HttpClient.make((request) => Effect.succeed(HttpClientResponse.fromWeb(
    request,
    respond(new Request(request.url, { method: request.method, headers: request.headers })),
  )))

const httpLayer = (client = responseClient(() => new Response("", { status: 404 }))) =>
  Layer.succeed(HttpClient.HttpClient, client)

const platformLayer = (client?: HttpClient.HttpClient) => Layer.mergeAll(
  BunPath.layer,
  BunCommandExecutor.layer,
  httpLayer(client),
).pipe(Layer.provideMerge(BunFileSystem.layer))

const sha256 = (value: Uint8Array): string => createHash("sha256").update(value).digest("hex")

describe("llama.cpp services", () => {
  it("reads the real llama-server version stream from stderr", async () => {
    const root = await mkdtemp(join(tmpdir(), "magnitude-llamacpp-version-"))
    roots.push(root)
    const executable = join(root, "llama-server")
    const managedRoot = join(root, "managed")
    await writeFile(executable, "#!/bin/sh\necho 'version: 10011 (test)' >&2\necho 'built for test' >&2\n")
    await chmod(executable, 0o755)
    const layer = LlamaCppDistributionLive({ managedRoot, configuredExecutable: executable }).pipe(
      Layer.provide(platformLayer()),
    )

    const state = await Effect.runPromise(
      Effect.flatMap(LlamaCppDistribution, (service) => service.inspect).pipe(Effect.provide(layer)),
    )

    expect(state._tag).toBe("Ready")
    if (state._tag === "Ready") expect(state.distribution.build).toBe(10011)
  })

  it("inspects a configured distribution without creating managed state", async () => {
    const root = await mkdtemp(join(tmpdir(), "magnitude-llamacpp-distribution-"))
    roots.push(root)
    const executable = join(root, "llama-server")
    const managedRoot = join(root, "managed")
    await writeFile(executable, "#!/bin/sh\necho 'version: 10011 (test)'\n")
    await chmod(executable, 0o755)
    const layer = LlamaCppDistributionLive({ managedRoot, configuredExecutable: executable }).pipe(
      Layer.provide(platformLayer()),
    )
    const state = await Effect.runPromise(
      Effect.flatMap(LlamaCppDistribution, (service) => service.inspect).pipe(Effect.provide(layer)),
    )
    expect(state).toEqual({
      _tag: "Ready",
      distribution: {
        executablePath: executable,
        directory: root,
        build: 10011,
        source: "configured",
      },
    })
    await expect(stat(managedRoot)).rejects.toThrow()
  })

  it("installs, verifies, atomically publishes, and re-inspects a managed distribution", async () => {
    const platform = process.platform === "darwin"
      ? "darwin"
      : process.platform === "linux"
        ? "linux"
        : null
    const architecture = process.arch === "arm64"
      ? "arm64"
      : process.arch === "x64"
        ? "x64"
        : null
    if (!platform || !architecture) return

    const root = await mkdtemp(join(tmpdir(), "magnitude-llamacpp-install-"))
    roots.push(root)
    const bundle = join(root, "bundle")
    await mkdir(bundle)
    const executable = join(bundle, "llama-server")
    await writeFile(executable, "#!/bin/sh\necho 'version: 10011 (test)'\n")
    await chmod(executable, 0o755)
    const archivePath = join(root, "llama.tar.gz")
    await execFileAsync("tar", ["-czf", archivePath, "-C", root, "bundle"])
    const archive = new Uint8Array(await readFile(archivePath))
    const asset: LlamaCppReleaseAsset = {
      platform,
      architecture,
      accelerator: platform === "darwin" && architecture === "arm64" ? "metal" : "cpu",
      fileName: "llama.tar.gz",
      url: "https://releases.invalid/llama.tar.gz",
      sizeBytes: archive.byteLength,
      sha256: sha256(archive),
    }
    const managedRoot = join(root, "managed")
    const layer = LlamaCppDistributionLive({
      managedRoot,
      release: { build: 10011, tag: "b10011", assets: [asset] },
    }).pipe(Layer.provide(platformLayer(responseClient(() => new Response(archive)))))

    const events = await Effect.runPromise(Effect.gen(function* () {
      const service = yield* LlamaCppDistribution
      const installed = yield* Stream.runCollect(service.install)
      const inspected = yield* service.inspect
      return { installed: Array.from(installed), inspected }
    }).pipe(Effect.provide(layer)))

    expect(events.installed.map((event) => event._tag)).toEqual([
      "Resolving",
      "Downloading",
      "Verifying",
      "Extracting",
      "Verifying",
      "Publishing",
      "Ready",
    ])
    expect(events.inspected).toEqual({
      _tag: "Ready",
      distribution: {
        executablePath: join(managedRoot, "llama-b10011", "llama-server"),
        directory: join(managedRoot, "llama-b10011"),
        build: 10011,
        source: "managed",
      },
    })
  })

  it("reports typed integrity failure and removes temporary install state", async () => {
    const platform = process.platform === "darwin"
      ? "darwin"
      : process.platform === "linux"
        ? "linux"
        : null
    const architecture = process.arch === "arm64"
      ? "arm64"
      : process.arch === "x64"
        ? "x64"
        : null
    if (!platform || !architecture) return

    const root = await mkdtemp(join(tmpdir(), "magnitude-llamacpp-integrity-"))
    roots.push(root)
    const archive = new TextEncoder().encode("not-the-pinned-release")
    const managedRoot = join(root, "managed")
    const asset: LlamaCppReleaseAsset = {
      platform,
      architecture,
      accelerator: platform === "darwin" && architecture === "arm64" ? "metal" : "cpu",
      fileName: "llama.tar.gz",
      url: "https://releases.invalid/llama.tar.gz",
      sizeBytes: archive.byteLength,
      sha256: "0".repeat(64),
    }
    const layer = LlamaCppDistributionLive({
      managedRoot,
      release: { build: 10011, tag: "b10011", assets: [asset] },
    }).pipe(Layer.provide(platformLayer(responseClient(() => new Response(archive)))))

    const result = await Effect.runPromise(Effect.gen(function* () {
      const distribution = yield* LlamaCppDistribution
      return yield* Stream.runDrain(distribution.install).pipe(Effect.either)
    }).pipe(Effect.provide(layer)))

    expect(result._tag).toBe("Left")
    if (result._tag === "Left") expect(result.left.code).toBe("integrity_failed")
    expect(await readdir(managedRoot)).toEqual([])
  })

  it("removes temporary distribution state when installation is interrupted", async () => {
    const platform = process.platform === "darwin"
      ? "darwin"
      : process.platform === "linux"
        ? "linux"
        : null
    const architecture = process.arch === "arm64"
      ? "arm64"
      : process.arch === "x64"
        ? "x64"
        : null
    if (!platform || !architecture) return

    const root = await mkdtemp(join(tmpdir(), "magnitude-llamacpp-interrupted-install-"))
    roots.push(root)
    const managedRoot = join(root, "managed")
    const chunk = new TextEncoder().encode("partial archive")
    const response = new Response(new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(chunk)
      },
    }))
    const asset: LlamaCppReleaseAsset = {
      platform,
      architecture,
      accelerator: platform === "darwin" && architecture === "arm64" ? "metal" : "cpu",
      fileName: "llama.tar.gz",
      url: "https://releases.invalid/llama.tar.gz",
      sizeBytes: chunk.byteLength * 2,
      sha256: "0".repeat(64),
    }
    const layer = LlamaCppDistributionLive({
      managedRoot,
      release: { build: 10011, tag: "b10011", assets: [asset] },
    }).pipe(Layer.provide(platformLayer(responseClient(() => response))))

    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const distribution = yield* LlamaCppDistribution
      const fiber = yield* Stream.runDrain(distribution.install).pipe(Effect.forkScoped)
      yield* Effect.sleep("50 millis")
      yield* Fiber.interrupt(fiber)
    }).pipe(Effect.provide(layer))))

    expect(await readdir(managedRoot)).toEqual([])
  })

  it("parses stable runtime device capacity separately from current free memory", () => {
    expect(parseRuntimeDevices([
      "MTL0: Apple M4 Max (65536 MiB, 12000 MiB free)",
      "MTL0: Apple M4 Max (65536 MiB, 12000 MiB free)",
    ].join("\n"))).toEqual([{
      backend: "MTL0",
      name: "Apple M4 Max",
      totalBytes: 65_536 * 1024 * 1024,
      freeBytes: 12_000 * 1024 * 1024,
    }])
  })

  it("rejects unsafe download plans before network access", async () => {
    const root = await mkdtemp(join(tmpdir(), "magnitude-llamacpp-models-"))
    roots.push(root)
    const layer = LlamaCppModelStoreLive({ ownedRoot: root, huggingFaceCacheRoot: join(root, "hf") }).pipe(
      Layer.provide(platformLayer()),
    )
    const result = await Effect.runPromise(Effect.gen(function* () {
      const store = yield* LlamaCppModelStore
      return yield* Stream.runDrain(store.download({
        artifactId: "unsafe",
        repo: "org/model",
        revision: "main",
        files: [{ path: "../escape.gguf", sizeBytes: 1, sha256: "0".repeat(64) }],
        safetyReserveBytes: 0,
      })).pipe(Effect.either)
    }).pipe(Effect.provide(layer)))
    expect(result._tag).toBe("Left")
    if (result._tag === "Left") expect(result.left.code).toBe("invalid_plan")
  })

  it("checks space on the owned destination before starting network work", async () => {
    const root = await mkdtemp(join(tmpdir(), "magnitude-llamacpp-space-"))
    roots.push(root)
    let requests = 0
    const layer = LlamaCppModelStoreLive({ ownedRoot: root, huggingFaceCacheRoot: join(root, "hf") }).pipe(
      Layer.provide(platformLayer(responseClient(() => {
        requests += 1
        return new Response("x")
      }))),
    )
    const result = await Effect.runPromise(Effect.gen(function* () {
      const store = yield* LlamaCppModelStore
      return yield* Stream.runDrain(store.download({
        artifactId: "too-large",
        repo: "org/model",
        revision: "c".repeat(40),
        files: [{ path: "model.gguf", sizeBytes: 1, sha256: sha256(new TextEncoder().encode("x")) }],
        safetyReserveBytes: Number.MAX_SAFE_INTEGER - 1,
      })).pipe(Effect.either)
    }).pipe(Effect.provide(layer)))

    expect(result._tag).toBe("Left")
    if (result._tag === "Left") expect(result.left.code).toBe("insufficient_space")
    expect(requests).toBe(0)
  })

  it("rejects a same-size download with the wrong SHA-256 before publication", async () => {
    const root = await mkdtemp(join(tmpdir(), "magnitude-llamacpp-model-integrity-"))
    roots.push(root)
    const body = new TextEncoder().encode("model-bytes")
    const ownedRoot = join(root, "owned")
    const layer = LlamaCppModelStoreLive({ ownedRoot, huggingFaceCacheRoot: join(root, "hf") }).pipe(
      Layer.provide(platformLayer(responseClient(() => new Response(body)))),
    )
    const result = await Effect.runPromise(Effect.gen(function* () {
      const store = yield* LlamaCppModelStore
      return yield* Stream.runDrain(store.download({
        artifactId: "bad-hash",
        repo: "org/model",
        revision: "d".repeat(40),
        files: [{ path: "model.gguf", sizeBytes: body.byteLength, sha256: "0".repeat(64) }],
        safetyReserveBytes: 0,
      })).pipe(Effect.either)
    }).pipe(Effect.provide(layer)))

    expect(result._tag).toBe("Left")
    if (result._tag === "Left") expect(result.left.code).toBe("integrity_failed")
    await expect(stat(join(ownedRoot, "bad-hash"))).rejects.toThrow()
  })

  it("discards partial bytes whose persisted plan fingerprint differs", async () => {
    const root = await mkdtemp(join(tmpdir(), "magnitude-llamacpp-plan-fingerprint-"))
    roots.push(root)
    const ownedRoot = join(root, "owned")
    const partialRoot = join(ownedRoot, ".partial-revision-change")
    await mkdir(partialRoot, { recursive: true })
    await writeFile(join(partialRoot, "plan.json"), JSON.stringify({
      artifactId: "revision-change",
      repo: "org/model",
      revision: "e".repeat(40),
      files: [{ path: "model.gguf", sizeBytes: 6, sha256: "1".repeat(64) }],
    }))
    await writeFile(join(partialRoot, "model.gguf.incomplete"), "old")

    const replacement = new TextEncoder().encode("new-model")
    let requestedRange: string | null = "not-requested"
    const layer = LlamaCppModelStoreLive({ ownedRoot, huggingFaceCacheRoot: join(root, "hf") }).pipe(
      Layer.provide(platformLayer(requestClient((request) => {
        requestedRange = request.headers.get("range")
        return new Response(replacement)
      }))),
    )
    await Effect.runPromise(Effect.gen(function* () {
      const store = yield* LlamaCppModelStore
      yield* Stream.runDrain(store.download({
        artifactId: "revision-change",
        repo: "org/model",
        revision: "f".repeat(40),
        files: [{ path: "model.gguf", sizeBytes: replacement.byteLength, sha256: sha256(replacement) }],
        safetyReserveBytes: 0,
      }))
    }).pipe(Effect.provide(layer)))

    expect(requestedRange).toBeNull()
  })

  it("never deletes models discovered in user directories", async () => {
    const root = await mkdtemp(join(tmpdir(), "magnitude-llamacpp-user-model-"))
    roots.push(root)
    const userDirectory = join(root, "user")
    await mkdir(userDirectory)
    const modelFile = join(userDirectory, "model.gguf")
    await writeFile(modelFile, "not-a-real-gguf")
    const layer = LlamaCppModelStoreLive({
      ownedRoot: join(root, "owned"),
      huggingFaceCacheRoot: join(root, "hf"),
      userDirectories: [{ directoryId: "user-models", path: userDirectory }],
    }).pipe(Layer.provide(platformLayer()))
    const result = await Effect.runPromise(Effect.gen(function* () {
      const store = yield* LlamaCppModelStore
      const snapshot = yield* store.inspect
      expect(snapshot.artifacts).toHaveLength(1)
      const artifact = snapshot.artifacts[0]
      if (!artifact) return yield* Effect.dieMessage("Expected one discovered user artifact")
      return yield* store.deleteOwned(artifact.modelId).pipe(Effect.either)
    }).pipe(Effect.provide(layer)))
    expect(result._tag).toBe("Left")
    if (result._tag === "Left") expect(result.left.code).toBe("artifact_not_owned")
    expect(await readFile(modelFile, "utf8")).toBe("not-a-real-gguf")
  })

  it("reports invalid owned artifacts instead of silently dropping them", async () => {
    const root = await mkdtemp(join(tmpdir(), "magnitude-llamacpp-invalid-owned-"))
    roots.push(root)
    const ownedRoot = join(root, "owned")
    await mkdir(join(ownedRoot, "broken"), { recursive: true })
    await writeFile(join(ownedRoot, "broken", "manifest.json"), "not-json")
    const layer = LlamaCppModelStoreLive({
      ownedRoot,
      huggingFaceCacheRoot: join(root, "hf"),
    }).pipe(Layer.provide(platformLayer()))

    const snapshot = await Effect.runPromise(
      Effect.flatMap(LlamaCppModelStore, (store) => store.inspect).pipe(Effect.provide(layer)),
    )
    expect(snapshot.artifacts).toEqual([])
    expect(snapshot.warnings).toEqual([expect.objectContaining({ code: "invalid_owned_artifact" })])
  })

  it("resumes a multi-file download without fetching verified finalized files again", async () => {
    const root = await mkdtemp(join(tmpdir(), "magnitude-llamacpp-resume-"))
    roots.push(root)
    const model = new TextEncoder().encode("not-a-real-gguf")
    const projector = new TextEncoder().encode("not-a-real-projector")
    const plan = {
      artifactId: "resume-artifact",
      repo: "org/model",
      revision: "a".repeat(40),
      files: [
        { path: "model.gguf", sizeBytes: model.byteLength, sha256: sha256(model) },
        { path: "mmproj.gguf", sizeBytes: projector.byteLength, sha256: sha256(projector) },
      ],
      safetyReserveBytes: 0,
    } as const
    const firstRequests: string[] = []
    const firstClient = responseClient((url) => {
      firstRequests.push(url.pathname)
      return url.pathname.endsWith("/model.gguf")
        ? new Response(model)
        : new Response("failure", { status: 500 })
    })
    const firstLayer = LlamaCppModelStoreLive({
      ownedRoot: join(root, "owned"),
      huggingFaceCacheRoot: join(root, "hf"),
    }).pipe(Layer.provide(platformLayer(firstClient)))
    const first = await Effect.runPromise(Effect.gen(function* () {
      const store = yield* LlamaCppModelStore
      return yield* Stream.runDrain(store.download(plan)).pipe(Effect.either)
    }).pipe(Effect.provide(firstLayer)))
    expect(first._tag).toBe("Left")
    expect(firstRequests).toHaveLength(2)

    const resumedRequests: string[] = []
    const resumedClient = responseClient((url) => {
      resumedRequests.push(url.pathname)
      return url.pathname.endsWith("/mmproj.gguf")
        ? new Response(projector)
        : new Response(model)
    })
    const resumedLayer = LlamaCppModelStoreLive({
      ownedRoot: join(root, "owned"),
      huggingFaceCacheRoot: join(root, "hf"),
    }).pipe(Layer.provide(platformLayer(resumedClient)))
    await Effect.runPromise(Effect.gen(function* () {
      const store = yield* LlamaCppModelStore
      yield* Stream.runDrain(store.download(plan))
      const artifact = yield* store.resolve(plan.artifactId)
      expect(artifact.source._tag).toBe("MagnitudeOwned")
      expect(artifact.hasVisionProjector).toBe(true)
      expect(artifact.sizeBytes).toBe(model.byteLength)
    }).pipe(Effect.provide(resumedLayer)))
    expect(resumedRequests).toHaveLength(1)
    expect(resumedRequests[0]).toMatch(/mmproj\.gguf$/)
  })

  it("interrupts an active download and resumes its verified partial bytes", async () => {
    const root = await mkdtemp(join(tmpdir(), "magnitude-llamacpp-interrupted-download-"))
    roots.push(root)
    const complete = new TextEncoder().encode("interruption-safe-model")
    const firstChunk = complete.slice(0, 8)
    const remainder = complete.slice(firstChunk.byteLength)
    const artifactId = "interrupted-artifact"
    const plan = {
      artifactId,
      repo: "org/model",
      revision: "b".repeat(40),
      files: [{ path: "model.gguf", sizeBytes: complete.byteLength, sha256: sha256(complete) }],
      safetyReserveBytes: 0,
    } as const
    const ownedRoot = join(root, "owned")
    const hangingResponse = new Response(new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(firstChunk)
      },
    }))
    const interruptedLayer = LlamaCppModelStoreLive({
      ownedRoot,
      huggingFaceCacheRoot: join(root, "hf"),
    }).pipe(Layer.provide(platformLayer(responseClient(() => hangingResponse))))

    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const store = yield* LlamaCppModelStore
      const fiber = yield* Stream.runDrain(store.download(plan)).pipe(Effect.forkScoped)
      yield* Effect.sleep("50 millis")
      yield* Fiber.interrupt(fiber)
    }).pipe(Effect.provide(interruptedLayer))))

    const partialFile = join(ownedRoot, `.partial-${artifactId}`, "model.gguf.incomplete")
    expect(new Uint8Array(await readFile(partialFile))).toEqual(firstChunk)

    let requestedRange: string | null = null
    const resumedLayer = LlamaCppModelStoreLive({
      ownedRoot,
      huggingFaceCacheRoot: join(root, "hf"),
    }).pipe(Layer.provide(platformLayer(requestClient((request) => {
      requestedRange = request.headers.get("range")
      return new Response(remainder, { status: 206 })
    }))))
    await Effect.runPromise(Effect.gen(function* () {
      const store = yield* LlamaCppModelStore
      yield* Stream.runDrain(store.download(plan))
      const artifact = yield* store.resolve(artifactId)
      expect(artifact.sizeBytes).toBe(complete.byteLength)
    }).pipe(Effect.provide(resumedLayer)))
    expect(requestedRange).toBe(`bytes=${firstChunk.byteLength}-`)
  })

  it("verifies an explicitly selected external target without mutating it", async () => {
    const client = responseClient((url) => {
      switch (url.pathname) {
        case "/health": return Response.json({ status: "ok" })
        case "/v1/models": return Response.json({ data: [{
          id: "local-model",
          object: "model",
          path: "/models/local-model.gguf",
          meta: {
            n_ctx: 8192,
            size: 10,
            ftype: "Q6_K",
            "general.name": "Local Model",
          },
        }] })
        case "/props": return Response.json({
          build_info: "b10011",
          default_generation_settings: { n_ctx: url.searchParams.get("model") === "local-model" ? 8192 : 0 },
        })
        default: return new Response("", { status: 404 })
      }
    })
    const distribution: LlamaCppDistributionApi = {
      inspect: Effect.succeed({ _tag: "Missing" }),
      install: Stream.empty,
    }
    const modelStore: LlamaCppModelStoreApi = {
      inspect: Effect.succeed({ artifacts: [], warnings: [] }),
      resolve: () => Effect.die("managed resolution must not run"),
      download: () => Stream.empty,
      deleteOwned: () => Effect.void,
    }
    const root = await mkdtemp(join(tmpdir(), "magnitude-llamacpp-runtime-"))
    roots.push(root)
    const dependencies = Layer.mergeAll(
      Layer.succeed(LlamaCppDistribution, distribution),
      Layer.succeed(LlamaCppModelStore, modelStore),
      platformLayer(client),
    )
    const runtimeLayer = LlamaCppRuntimeLive({
      runtimeRoot: root,
      externalConnections: () => Effect.succeed([{
        connectionId: "configured-endpoint",
        connection: {
          baseUrl: "http://127.0.0.1:9090",
          apiKey: Option.none(),
        },
      }]),
    }).pipe(Layer.provide(dependencies))
    const { snapshot, target } = await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const runtime = yield* LlamaCppRuntime
      const snapshot = yield* runtime.inspect
      const target = yield* runtime.ensureServing({
        _tag: "External",
        connectionId: "configured-endpoint",
        providerModelId: "local-model",
        contextTokens: 8192,
      })
      return { snapshot, target }
    }).pipe(Effect.provide(runtimeLayer))))
    expect(target.ownership).toBe("external")
    expect(target.serverId).toBe("configured-endpoint")
    expect(target.configuredContextTokens).toBe(8192)
    expect(snapshot.external[0]?.models).toEqual([{
      providerModelId: "local-model",
      modelPath: "/models/local-model.gguf",
      displayName: "Local Model",
      contextTokens: 8192,
      quantization: "Q6_K",
      sizeBytes: 10,
    }])
  })

  it("serializes concurrent managed ensures and stops the one process at scope close", async () => {
    const root = await mkdtemp(join(tmpdir(), "magnitude-llamacpp-managed-runtime-"))
    roots.push(root)
    const executable = join(root, "fake-llama-server.ts")
    const lifecycleLog = join(root, "lifecycle.log")
    const modelPath = join(root, "model.gguf")
    await writeFile(modelPath, "fake-model")
    await writeFile(executable, [
      "#!/usr/bin/env bun",
      "import { appendFileSync } from 'node:fs'",
      "const args = process.argv.slice(2)",
      "const valueAfter = (name) => args[args.indexOf(name) + 1]",
      "const port = Number(valueAfter('--port'))",
      "const preset = await Bun.file(valueAfter('--models-preset')).text()",
      "const alias = preset.match(/LLAMA_ARG_ALIAS = (.+)/)?.[1]?.trim() ?? 'model'",
      "const modelPath = preset.match(/LLAMA_ARG_MODEL = (.+)/)?.[1]?.trim()",
      "const context = Number(preset.match(/LLAMA_ARG_CTX_SIZE = (\\d+)/)?.[1] ?? '0')",
      "if (!preset.includes('LLAMA_ARG_N_PARALLEL = 1')) throw new Error('missing supported parallel preset key')",
      "if (!process.env.LLAMA_API_KEY) throw new Error('missing supported API key environment variable')",
      `const log = ${JSON.stringify(lifecycleLog)}`,
      "appendFileSync(log, 'start\\n')",
      "const server = Bun.serve({ port, hostname: '127.0.0.1', fetch(request) {",
      "  const url = new URL(request.url)",
      "  const path = url.pathname",
      "  if (path === '/health') return alias === 'slow-model' ? new Response('', { status: 503 }) : Response.json({ status: 'ok' })",
      "  if (path === '/v1/models') return Response.json({ data: [{ id: alias, object: 'model', path: alias === 'path-undisclosed' ? 'none' : modelPath, meta: { size: 10 } }] })",
      "  if (path === '/props') return Response.json({ build_info: 'b10011', default_generation_settings: { n_ctx: url.searchParams.get('model') === alias ? (alias === 'wrong-context' ? context + 1 : context) : 0 } })",
      "  return new Response('', { status: 404 })",
      "} })",
      "process.on('SIGTERM', () => { appendFileSync(log, 'stop\\n'); server.stop(true); process.exit(0) })",
      "await new Promise(() => {})",
    ].join("\n"))
    await chmod(executable, 0o755)

    const distribution: LlamaCppDistributionApi = {
      inspect: Effect.succeed({
        _tag: "Ready",
        distribution: { executablePath: executable, directory: root, build: 10011, source: "configured" },
      }),
      install: Stream.empty,
    }
    const modelStore: LlamaCppModelStoreApi = {
      inspect: Effect.succeed({ artifacts: [], warnings: [] }),
      resolve: (modelId) => Effect.succeed({
        modelId,
        source: { _tag: "MagnitudeOwned", manifestId: modelId },
        sizeBytes: modelId === "wrong-size" ? 11 : 10,
        metadata: {
          displayName: "Model",
          architecture: null,
          quantization: null,
          contextLength: 8192,
          parameterCount: null,
          layerCount: null,
          tokenizerModel: null,
          tokenizerPre: null,
          baseModelNames: [],
        },
        hasVisionProjector: false,
        primaryPath: modelPath,
        shardPaths: [modelPath],
        projectorPath: null,
      }),
      download: () => Stream.empty,
      deleteOwned: () => Effect.void,
    }
    const realPlatform = Layer.mergeAll(
      BunPath.layer,
      BunCommandExecutor.layer,
      FetchHttpClient.layer,
    ).pipe(Layer.provideMerge(BunFileSystem.layer))
    const dependencies = Layer.mergeAll(
      Layer.succeed(LlamaCppDistribution, distribution),
      Layer.succeed(LlamaCppModelStore, modelStore),
      realPlatform,
    )
    const runtimeLayer = LlamaCppRuntimeLive({
      runtimeRoot: join(root, "runtime"),
      preferredPort: 18_080,
      externalConnections: () => Effect.succeed([{
        connectionId: "default-external",
        connection: { baseUrl: "http://127.0.0.1:18080/", apiKey: Option.none() },
      }]),
    }).pipe(Layer.provide(dependencies))
    const request = {
      _tag: "Managed",
      modelId: "artifact",
      providerModelId: "local-model",
      contextTokens: 8192,
      fitPlan: {
        requiredBytes: 10,
        stableCapacityBytes: 100,
        parallelSlots: 1,
        gpuLayers: 0,
        splitMode: "none",
        fits: true,
      },
    } as const

    const serverIds = await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const runtime = yield* LlamaCppRuntime
      const targets = yield* Effect.all([
        runtime.ensureServing(request),
        runtime.ensureServing(request),
      ], { concurrency: 2 })
      const snapshot = yield* runtime.inspect
      expect(snapshot.managed?.serverId).toBe(targets[0]?.serverId)
      expect(snapshot.external).toEqual([])
      const collision = yield* runtime.ensureServing({
        _tag: "External",
        connectionId: "default-external",
        providerModelId: "local-model",
        contextTokens: 8192,
      }).pipe(Effect.either)
      expect(collision._tag).toBe("Left")
      if (collision._tag === "Left") expect(collision.left.code).toBe("external_unavailable")
      const mismatch = yield* runtime.ensureServing({ ...request, modelId: "wrong-size" }).pipe(Effect.either)
      expect(mismatch._tag).toBe("Left")
      if (mismatch._tag === "Left") expect(mismatch.left.code).toBe("identity_mismatch")
      const contextMismatch = yield* runtime.ensureServing({
        ...request,
        providerModelId: "wrong-context",
      }).pipe(Effect.either)
      expect(contextMismatch._tag).toBe("Left")
      if (contextMismatch._tag === "Left") expect(contextMismatch.left.code).toBe("context_mismatch")
      const undisclosedPath = yield* runtime.ensureServing({
        ...request,
        providerModelId: "path-undisclosed",
      })
      expect(undisclosedPath.providerModelId).toBe("path-undisclosed")
      const starting = yield* runtime.ensureServing({
        ...request,
        modelId: "slow-artifact",
        providerModelId: "slow-model",
      }).pipe(Effect.fork)
      yield* Effect.sleep("100 millis")
      yield* Fiber.interrupt(starting)
      return targets.map((target) => target.serverId)
    }).pipe(Effect.provide(runtimeLayer))))

    expect(new Set(serverIds).size).toBe(1)
    expect((await readFile(lifecycleLog, "utf8")).trim().split("\n")).toEqual([
      "start", "stop",
      "start", "stop",
      "start", "stop",
      "start", "stop",
      "start", "stop",
    ])
    expect(await readdir(join(root, "runtime"))).toEqual([])
  })
})
