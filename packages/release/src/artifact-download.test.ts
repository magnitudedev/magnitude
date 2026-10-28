import * as FetchHttpClient from "@effect/platform/FetchHttpClient"
import * as FileSystem from "@effect/platform/FileSystem"
import * as HttpClient from "@effect/platform/HttpClient"
import * as Path from "@effect/platform/Path"
import * as NodeFileSystem from "@effect/platform-node/NodeFileSystem"
import * as NodePath from "@effect/platform-node/NodePath"
import { createHash } from "node:crypto"
import { mkdir, mkdtemp, readFile, readdir, rm, writeFile } from "node:fs/promises"
import { tmpdir } from "node:os"
import { join } from "node:path"
import { Effect, Fiber, Layer, Option } from "effect"
import { describe, expect, it } from "vitest"
import {
  defaultArtifactDownloadPolicy,
  downloadArtifact,
  partialDirectoryName,
  type ArtifactDownloadPolicy,
} from "./artifact-download"

const DownloadLayer = Layer.mergeAll(
  NodeFileSystem.layer,
  NodePath.layer,
  FetchHttpClient.layer,
)

const sha256 = (bytes: Uint8Array) =>
  createHash("sha256").update(bytes).digest("hex")

const policy = (
  overrides: Partial<ArtifactDownloadPolicy> = {},
): ArtifactDownloadPolicy => ({
  ...defaultArtifactDownloadPolicy,
  ...overrides,
})

const run = <A, E>(
  effect: Effect.Effect<
    A,
    E,
    FileSystem.FileSystem | Path.Path | HttpClient.HttpClient
  >,
) => Effect.runPromise(effect.pipe(Effect.provide(DownloadLayer)))

const parseRange = (header: string): { start: number; end: number } => {
  const match = /^bytes=(\d+)-(\d+)$/.exec(header)
  if (!match) throw new Error(`invalid test range ${header}`)
  return { start: Number(match[1]), end: Number(match[2]) }
}

describe("artifact downloader", () => {
  it("retries the complete sequential transfer after a transient response", async () => {
    const bytes = new TextEncoder().encode("sequential artifact")
    const root = await mkdtemp(join(tmpdir(), "artifact-download-test-"))
    let requests = 0
    const server = Bun.serve({
      hostname: "127.0.0.1",
      port: 0,
      fetch: () => {
        requests += 1
        return requests === 1
          ? new Response("retry", { status: 503 })
          : new Response(bytes, {
            headers: { "content-length": String(bytes.byteLength) },
          })
      },
    })
    try {
      const destination = join(root, "artifact")
      const result = await run(downloadArtifact({
        url: `http://127.0.0.1:${server.port}/artifact`,
        destination,
        bytes: bytes.byteLength,
        sha256: sha256(bytes),
        strategy: { _tag: "Sequential" },
        policy: policy({ retryDelay: "1 millis" }),
        onProgress: Option.none(),
        onVerificationProgress: Option.none(),
      }))
      expect(result.strategy).toBe("Sequential")
      expect(requests).toBe(2)
      expect(new Uint8Array(await readFile(destination))).toEqual(bytes)
    } finally {
      server.stop(true)
      await rm(root, { recursive: true, force: true })
    }
  })

  it("bounds range concurrency and retries only the failed part", async () => {
    const bytes = new TextEncoder().encode("0123456789abcdefghijklmnopqrstuv")
    const root = await mkdtemp(join(tmpdir(), "artifact-download-test-"))
    const attempts = new Map<string, number>()
    let active = 0
    let maximumActive = 0
    const server = Bun.serve({
      hostname: "127.0.0.1",
      port: 0,
      fetch: async (request) => {
        const header = request.headers.get("range")
        if (!header) return new Response("range required", { status: 400 })
        const { start, end } = parseRange(header)
        const key = `${start}-${end}`
        const attempt = (attempts.get(key) ?? 0) + 1
        attempts.set(key, attempt)
        if ((key === "0-0" || key === "8-15") && attempt === 1) {
          return new Response("retry", { status: 503 })
        }
        active += 1
        maximumActive = Math.max(maximumActive, active)
        await Bun.sleep(20)
        active -= 1
        const part = bytes.slice(start, end + 1)
        return new Response(part, {
          status: 206,
          headers: {
            "content-range": `bytes ${start}-${end}/${bytes.byteLength}`,
            "content-length": String(part.byteLength),
            etag: "\"stable-test-representation\"",
          },
        })
      },
    })
    try {
      const destination = join(root, "artifact")
      const result = await run(downloadArtifact({
        url: `http://127.0.0.1:${server.port}/artifact`,
        destination,
        bytes: bytes.byteLength,
        sha256: sha256(bytes),
        strategy: {
          _tag: "Segmented",
          concurrency: 2,
          chunkBytes: 8,
          fallbackToSequential: false,
        },
        policy: policy({ retryDelay: "1 millis" }),
        onProgress: Option.none(),
        onVerificationProgress: Option.none(),
      }))
      expect(result.strategy).toBe("Segmented")
      expect(maximumActive).toBe(2)
      expect(attempts.get("0-0")).toBe(2)
      expect(attempts.get("8-15")).toBe(2)
      expect(attempts.get("0-7")).toBe(1)
      expect(attempts.get("16-23")).toBe(1)
      expect(attempts.get("24-31")).toBe(1)
      expect(new Uint8Array(await readFile(destination))).toEqual(bytes)
    } finally {
      server.stop(true)
      await rm(root, { recursive: true, force: true })
    }
  })

  it("falls back to a retriable sequential transfer when ranges are unsupported", async () => {
    const bytes = new TextEncoder().encode("fallback artifact")
    const root = await mkdtemp(join(tmpdir(), "artifact-download-test-"))
    let rangeRequests = 0
    let sequentialRequests = 0
    const server = Bun.serve({
      hostname: "127.0.0.1",
      port: 0,
      fetch: (request) => {
        if (request.headers.has("range")) rangeRequests += 1
        else sequentialRequests += 1
        return new Response(bytes, {
          headers: { "content-length": String(bytes.byteLength) },
        })
      },
    })
    try {
      const result = await run(downloadArtifact({
        url: `http://127.0.0.1:${server.port}/artifact`,
        destination: join(root, "artifact"),
        bytes: bytes.byteLength,
        sha256: sha256(bytes),
        strategy: {
          _tag: "Segmented",
          concurrency: 4,
          chunkBytes: 8,
          fallbackToSequential: true,
        },
        policy: policy({ retryDelay: "1 millis" }),
        onProgress: Option.none(),
        onVerificationProgress: Option.none(),
      }))
      expect(result.strategy).toBe("Sequential")
      expect(rangeRequests).toBe(1)
      expect(sequentialRequests).toBe(1)
    } finally {
      server.stop(true)
      await rm(root, { recursive: true, force: true })
    }
  })

  it("does not publish an artifact whose digest is invalid", async () => {
    const bytes = new TextEncoder().encode("corrupt artifact")
    const root = await mkdtemp(join(tmpdir(), "artifact-download-test-"))
    const destination = join(root, "artifact")
    const server = Bun.serve({
      hostname: "127.0.0.1",
      port: 0,
      fetch: () => new Response(bytes),
    })
    try {
      const error = await run(downloadArtifact({
        url: `http://127.0.0.1:${server.port}/artifact`,
        destination,
        bytes: bytes.byteLength,
        sha256: "0".repeat(64),
        strategy: { _tag: "Sequential" },
        policy: policy({ retryDelay: "1 millis" }),
        onProgress: Option.none(),
        onVerificationProgress: Option.none(),
      }).pipe(Effect.flip))
      expect(error.phase).toBe("integrity")
      await expect(readFile(destination)).rejects.toMatchObject({ code: "ENOENT" })
    } finally {
      server.stop(true)
      await rm(root, { recursive: true, force: true })
    }
  })

  it.each([
    { invalidProbe: true, shiftedBounds: false, expectedRequests: 1 },
    { invalidProbe: false, shiftedBounds: false, expectedRequests: 2 },
    { invalidProbe: false, shiftedBounds: true, expectedRequests: 2 },
  ])("rejects inconsistent ranges without publishing ($invalidProbe, $shiftedBounds)", async ({ invalidProbe, shiftedBounds, expectedRequests }) => {
    const bytes = new TextEncoder().encode("inconsistent range")
    const root = await mkdtemp(join(tmpdir(), "artifact-download-test-"))
    const destination = join(root, "artifact")
    let requests = 0
    const server = Bun.serve({
      hostname: "127.0.0.1",
      port: 0,
      fetch: (request) => {
        requests += 1
        const { start, end } = parseRange(request.headers.get("range")!)
        const part = bytes.slice(start, end + 1)
        return new Response(part, {
          status: 206,
          headers: {
            "content-range": `bytes ${shiftedBounds && end > 0 ? start + 1 : start}-${end}/${!invalidProbe && start === 0 && end === 0 ? bytes.byteLength : bytes.byteLength + 1}`,
            "content-length": String(part.byteLength),
          },
        })
      },
    })
    try {
      const error = await run(downloadArtifact({
        url: `http://127.0.0.1:${server.port}/artifact`,
        destination,
        bytes: bytes.byteLength,
        sha256: sha256(bytes),
        strategy: {
          _tag: "Segmented",
          concurrency: 1,
          chunkBytes: 8,
          fallbackToSequential: true,
        },
        policy: policy({ retryDelay: "1 millis", retryWindow: "0 millis" }),
        onProgress: Option.none(),
        onVerificationProgress: Option.none(),
      }).pipe(Effect.flip))
      expect(error.phase).toBe("protocol")
      expect(error.message).toContain(`/${bytes.byteLength + 1}; expected `)
      expect(requests).toBe(expectedRequests)
      await expect(readFile(destination)).rejects.toMatchObject({ code: "ENOENT" })
    } finally {
      server.stop(true)
      await rm(root, { recursive: true, force: true })
    }
  })

  it("rejects range bytes from a changed strong ETag", async () => {
    const bytes = new TextEncoder().encode("changed representation")
    const root = await mkdtemp(join(tmpdir(), "artifact-download-test-"))
    const destination = join(root, "artifact")
    let requests = 0
    const server = Bun.serve({
      hostname: "127.0.0.1",
      port: 0,
      fetch: (request) => {
        requests += 1
        const { start, end } = parseRange(request.headers.get("range")!)
        const part = bytes.slice(start, end + 1)
        return new Response(part, {
          status: 206,
          headers: {
            "content-range": `bytes ${start}-${end}/${bytes.byteLength}`,
            "content-length": String(part.byteLength),
            etag: requests === 1 ? "\"first\"" : "\"second\"",
          },
        })
      },
    })
    try {
      const error = await run(downloadArtifact({
        url: `http://127.0.0.1:${server.port}/artifact`,
        destination,
        bytes: bytes.byteLength,
        sha256: sha256(bytes),
        strategy: {
          _tag: "Segmented",
          concurrency: 2,
          chunkBytes: 8,
          fallbackToSequential: true,
        },
        policy: policy({ retryDelay: "1 millis" }),
        onProgress: Option.none(),
        onVerificationProgress: Option.none(),
      }).pipe(Effect.flip))
      expect(error.phase).toBe("protocol")
      expect(error.message).toContain("different ETag")
      await expect(readFile(destination)).rejects.toMatchObject({ code: "ENOENT" })
    } finally {
      server.stop(true)
      await rm(root, { recursive: true, force: true })
    }
  })

  it("keeps the received parts when the download is interrupted", async () => {
    const bytes = new TextEncoder().encode("0123456789abcdef")
    const root = await mkdtemp(join(tmpdir(), "artifact-download-test-"))
    let started!: () => void
    const secondRangeStarted = new Promise<void>((resolve) => {
      started = resolve
    })
    const server = Bun.serve({
      hostname: "127.0.0.1",
      port: 0,
      fetch: async (request) => {
        const { start, end } = parseRange(request.headers.get("range")!)
        if (start === 8) {
          started()
          await Bun.sleep(10_000)
        }
        const part = bytes.slice(start, end + 1)
        return new Response(part, {
          status: 206,
          headers: {
            "content-range": `bytes ${start}-${end}/${bytes.byteLength}`,
            "content-length": String(part.byteLength),
          },
        })
      },
    })
    try {
      const fiber = Effect.runFork(downloadArtifact({
        url: `http://127.0.0.1:${server.port}/artifact`,
        destination: join(root, "artifact"),
        bytes: bytes.byteLength,
        sha256: sha256(bytes),
        strategy: { _tag: "Segmented", concurrency: 1, chunkBytes: 8, fallbackToSequential: false },
        policy: policy(),
        onProgress: Option.none(),
        onVerificationProgress: Option.none(),
      }).pipe(Effect.provide(DownloadLayer)))
      await secondRangeStarted
      await Effect.runPromise(Fiber.interrupt(fiber))
      const partial = join(root, partialDirectoryName(sha256(bytes)))
      expect(await readdir(root)).toEqual([partialDirectoryName(sha256(bytes))])
      expect(new TextDecoder().decode(await readFile(join(partial, "part-0")))).toBe("01234567")
    } finally {
      server.stop(true)
      await rm(root, { recursive: true, force: true })
    }
  })

  it("resumes kept parts and dropped ranges from their received bytes", async () => {
    const bytes = new TextEncoder().encode("0123456789abcdefghijklmnopqrstuv")
    const root = await mkdtemp(join(tmpdir(), "artifact-download-test-"))
    const partial = join(root, partialDirectoryName(sha256(bytes)))
    await mkdir(partial)
    await writeFile(join(partial, "part-0"), "01234567")
    await writeFile(join(partial, "part-1"), "89ab")
    const requested: string[] = []
    let dropped = false
    const server = Bun.serve({
      hostname: "127.0.0.1",
      port: 0,
      fetch: (request) => {
        const header = request.headers.get("range")!
        requested.push(header)
        const { start, end } = parseRange(header)
        const part = bytes.slice(start, end + 1)
        const headers = {
          "content-range": `bytes ${start}-${end}/${bytes.byteLength}`,
          "content-length": String(part.byteLength),
        }
        if (start === 16 && !dropped) {
          dropped = true
          return new Response(new ReadableStream({
            async start(controller) {
              controller.enqueue(part.slice(0, 3))
              await Bun.sleep(50)
              controller.error(new Error("connection reset"))
            },
          }), { status: 206, headers })
        }
        return new Response(part, { status: 206, headers })
      },
    })
    try {
      const destination = join(root, "artifact")
      await run(downloadArtifact({
        url: `http://127.0.0.1:${server.port}/artifact`,
        destination,
        bytes: bytes.byteLength,
        sha256: sha256(bytes),
        strategy: { _tag: "Segmented", concurrency: 1, chunkBytes: 8, fallbackToSequential: false },
        policy: policy({ retryDelay: "1 millis" }),
        onProgress: Option.none(),
        onVerificationProgress: Option.none(),
      }))
      expect(requested).toEqual([
        "bytes=0-0",
        "bytes=12-15",
        "bytes=16-23",
        "bytes=19-23",
        "bytes=24-31",
      ])
      expect(new Uint8Array(await readFile(destination))).toEqual(bytes)
      expect(await readdir(root)).toEqual(["artifact"])
    } finally {
      server.stop(true)
      await rm(root, { recursive: true, force: true })
    }
  })

  it("keeps retrying an unavailable range for its retry window", async () => {
    const bytes = new TextEncoder().encode("0123456789abcdef")
    const root = await mkdtemp(join(tmpdir(), "artifact-download-test-"))
    const outageEnds = Date.now() + 300
    let unavailable = 0
    const server = Bun.serve({
      hostname: "127.0.0.1",
      port: 0,
      fetch: (request) => {
        const { start, end } = parseRange(request.headers.get("range")!)
        if (start === 8 && Date.now() < outageEnds) {
          unavailable += 1
          return new Response("unavailable", { status: 503 })
        }
        const part = bytes.slice(start, end + 1)
        return new Response(part, {
          status: 206,
          headers: {
            "content-range": `bytes ${start}-${end}/${bytes.byteLength}`,
            "content-length": String(part.byteLength),
          },
        })
      },
    })
    try {
      const destination = join(root, "artifact")
      await run(downloadArtifact({
        url: `http://127.0.0.1:${server.port}/artifact`,
        destination,
        bytes: bytes.byteLength,
        sha256: sha256(bytes),
        strategy: { _tag: "Segmented", concurrency: 2, chunkBytes: 8, fallbackToSequential: false },
        policy: policy({ retryDelay: "5 millis", retryMaxDelay: "40 millis", retryWindow: "5 seconds" }),
        onProgress: Option.none(),
        onVerificationProgress: Option.none(),
      }))
      expect(unavailable).toBeGreaterThan(3)
      expect(new Uint8Array(await readFile(destination))).toEqual(bytes)
    } finally {
      server.stop(true)
      await rm(root, { recursive: true, force: true })
    }
  })

  it("gives up once the retry window passes without progress", async () => {
    const bytes = new TextEncoder().encode("0123456789abcdef")
    const root = await mkdtemp(join(tmpdir(), "artifact-download-test-"))
    const server = Bun.serve({
      hostname: "127.0.0.1",
      port: 0,
      fetch: (request) => {
        const { start, end } = parseRange(request.headers.get("range")!)
        if (start === 8) return new Response("unavailable", { status: 503 })
        const part = bytes.slice(start, end + 1)
        return new Response(part, {
          status: 206,
          headers: {
            "content-range": `bytes ${start}-${end}/${bytes.byteLength}`,
            "content-length": String(part.byteLength),
          },
        })
      },
    })
    try {
      const error = await run(downloadArtifact({
        url: `http://127.0.0.1:${server.port}/artifact`,
        destination: join(root, "artifact"),
        bytes: bytes.byteLength,
        sha256: sha256(bytes),
        strategy: { _tag: "Segmented", concurrency: 2, chunkBytes: 8, fallbackToSequential: false },
        policy: policy({ retryDelay: "5 millis", retryMaxDelay: "10 millis", retryWindow: "100 millis" }),
        onProgress: Option.none(),
        onVerificationProgress: Option.none(),
      }).pipe(Effect.flip))
      expect(error.message).toBe("range 1 returned HTTP 503, expected 206")
      expect(error.transient).toBe(true)
      expect(await readdir(join(root, partialDirectoryName(sha256(bytes))))).toEqual(["part-0"])
    } finally {
      server.stop(true)
      await rm(root, { recursive: true, force: true })
    }
  })

  it("bounds the complete acquisition independently of attempt retries", async () => {
    const bytes = new TextEncoder().encode("deadline artifact")
    const root = await mkdtemp(join(tmpdir(), "artifact-download-test-"))
    const destination = join(root, "artifact")
    const server = Bun.serve({
      hostname: "127.0.0.1",
      port: 0,
      fetch: async () => {
        await Bun.sleep(10_000)
        return new Response(bytes)
      },
    })
    try {
      const error = await run(downloadArtifact({
        url: `http://127.0.0.1:${server.port}/artifact`,
        destination,
        bytes: bytes.byteLength,
        sha256: sha256(bytes),
        strategy: { _tag: "Sequential" },
        policy: policy({
          attemptTimeout: "1 minute",
          totalTimeout: "10 millis",
        }),
        onProgress: Option.none(),
        onVerificationProgress: Option.none(),
      }).pipe(Effect.flip))
      expect(error.phase).toBe("stream")
      expect(error.message).toContain("total deadline")
      expect(await readdir(root)).toEqual([partialDirectoryName(sha256(bytes))])
    } finally {
      server.stop(true)
      await rm(root, { recursive: true, force: true })
    }
  })
})
