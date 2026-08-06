/**
 * Dev server — the single server for `bun web`.
 *
 * One process, one port. This server:
 * 1. Exposes ACN process management
 * 2. Serves the web app via Vite's middleware
 * 3. Exposes current, launch, and exact termination operations
 * 4. Proxies RPC, health, and logs only to the selected exact ACN
 *
 * The browser talks only to this same-origin server.
 */
import http, { createServer, type ServerResponse } from "node:http"
import { createServer as createViteServer } from "vite"
import { Effect, Exit, Layer, Runtime, Option, Schema, Scope, Stream } from "effect"
import { FetchHttpClient } from "@effect/platform"
import { BunContext } from "@effect/platform-bun"
import {
  makeLocalAcnProcessManager,
  BunDetachedChildProcessSpawner,
  ChildProcessSpawner,
  AcnLaunchRequestSchema,
  RemoteAcnCurrentResponseSchema,
  RemoteAcnErrorResponseSchema,
  RemoteAcnLaunchMessageSchema,
  RemoteAcnTerminateRequestSchema,
  DaemonError,
  DaemonSpawnFailed,
  type RemoteAcnLaunchMessage,
} from "@magnitudedev/sdk"
import { resolve } from "node:path"

// ─── Daemon host boundaries ─────────────────────────────────────────────────

const rt = Runtime.defaultRuntime
const processManagerScope = await Runtime.runPromise(rt)(Scope.make())

async function createProcessManager() {
  return Runtime.runPromise(rt)(makeLocalAcnProcessManager().pipe(
    Effect.provideService(ChildProcessSpawner, BunDetachedChildProcessSpawner),
    Effect.provideService(Scope.Scope, processManagerScope),
    Effect.provide(Layer.mergeAll(FetchHttpClient.layer, BunContext.layer)),
  ))
}

const processManagerPromise = createProcessManager()

async function currentAcn() {
  const manager = await processManagerPromise
  const result = await Runtime.runPromise(rt)(manager.observeCurrent)
  return Option.getOrElse(result, () => null)
}

// ─── Dev-mode launch command ────────────────────────────────────────────────

const acnSourcePath = resolve(import.meta.dir, "..", "..", "packages", "acn", "src", "binary.ts")
const defaultLaunchCommand = Option.some<ReadonlyArray<string>>([
  "bun",
  acnSourcePath,
  "serve",
  "--register",
])
const decodeLaunchRequest = Schema.decode(
  Schema.parseJson(AcnLaunchRequestSchema),
)
const encodeLaunchMessage = Schema.encode(
  Schema.parseJson(RemoteAcnLaunchMessageSchema),
)
const encodeCurrentResponse = Schema.encode(
  Schema.parseJson(RemoteAcnCurrentResponseSchema),
)
const decodeTerminateRequest = Schema.decode(
  Schema.parseJson(RemoteAcnTerminateRequestSchema),
)
const encodeErrorResponse = Schema.encode(
  Schema.parseJson(RemoteAcnErrorResponseSchema),
)

const respondError = async (
  res: ServerResponse,
  status: number,
  error: unknown,
): Promise<void> => {
  res.writeHead(status, { "Content-Type": "application/json" })
  res.end(JSON.stringify({ error: String(error) }))
}

const asDaemonError = (cause: unknown) => Schema.is(DaemonError)(cause)
  ? cause
  : new DaemonSpawnFailed({ reason: String(cause) })

const respondDaemonError = async (
  res: ServerResponse,
  status: number,
  cause: unknown,
): Promise<void> => {
  res.writeHead(status, { "Content-Type": "application/json" })
  res.end(await Runtime.runPromise(rt)(
    encodeErrorResponse({ error: asDaemonError(cause) }),
  ))
}
// ─── HTTP server with Vite middleware ─────────────────────────────────

const PORT = Number(process.env.PORT) || 5173

const vite = await createViteServer({
  configFile: resolve(import.meta.dir, "..", "vite.config.ts"),
  root: resolve(import.meta.dir, ".."),
  server: { middlewareMode: true },
})

const server = createServer(async (req, res) => {
  const url = new URL(req.url!, `http://localhost:${PORT}`)

  // ── ACN process operations ──────────────────────────────────────
  if (url.pathname === "/acn/current" && req.method === "GET") {
    try {
      const found = await currentAcn()
      res.writeHead(200, { "Content-Type": "application/json" })
      res.end(await Runtime.runPromise(rt)(encodeCurrentResponse({ instance: found })))
    } catch (err) {
      await respondDaemonError(res, 500, err)
    }
    return
  }

  if (url.pathname === "/acn/launch" && req.method === "POST") {
    try {
      const chunks: Buffer[] = []
      for await (const chunk of req) chunks.push(chunk as Buffer)
      const raw = Buffer.concat(chunks).toString()
      const body = await Runtime.runPromise(rt)(decodeLaunchRequest(
        raw.length === 0 ? "{}" : raw,
      ))
      res.writeHead(200, {
        "Content-Type": "application/x-ndjson",
        "Cache-Control": "no-store",
      })
      const write = (message: RemoteAcnLaunchMessage) =>
        encodeLaunchMessage(message).pipe(
          Effect.flatMap((encoded) =>
            Effect.sync(() => {
              if (!res.destroyed) res.write(`${encoded}\n`)
            }),
          ),
        )
      try {
        const manager = await processManagerPromise
        await Runtime.runPromise(rt)(
          manager
            .launch({
              ...body,
              command: Option.orElse(body.command, () => defaultLaunchCommand),
            })
            .pipe(
              Stream.runForEach(write),
            ),
        )
      } catch (err) {
        await Runtime.runPromise(rt)(write({ _tag: "Failed", error: asDaemonError(err) }))
      }
      res.end()
    } catch (err) {
      await respondDaemonError(res, 500, err)
    }
    return
  }

  if (url.pathname === "/acn/terminate" && req.method === "POST") {
    try {
      const chunks: Buffer[] = []
      for await (const chunk of req) chunks.push(chunk as Buffer)
      const body = await Runtime.runPromise(rt)(decodeTerminateRequest(
        Buffer.concat(chunks).toString(),
      ))
      const manager = await processManagerPromise
      await Runtime.runPromise(rt)(manager.terminate(body.instance))
      res.writeHead(204)
      res.end()
    } catch (err) {
      await respondDaemonError(res, 500, err)
    }
    return
  }

  // ── Same-origin ACN proxy (streaming) ───────────────────────────
  const proxyMatch = url.pathname.match(/^\/acn\/([^/]+)(\/rpc|\/health|\/logs)$/)
  if (proxyMatch) {
    const expectedId = decodeURIComponent(proxyMatch[1]!)
    const targetPath = proxyMatch[2]!
    let current
    try {
      current = await currentAcn()
    } catch (error) {
      await respondError(res, 502, error)
      return
    }
    if (!current) {
      await respondError(res, 503, "No daemon available")
      return
    }
    if (current.id !== expectedId) {
      await respondError(res, 409, "Selected ACN instance is no longer current")
      return
    }

    const target = new URL(current.url)
    const proxyReq = http.request({
      hostname: target.hostname,
      port: target.port,
      path: targetPath + url.search,
      method: req.method,
      headers: req.headers,
    }, (proxyRes) => {
      res.writeHead(proxyRes.statusCode ?? 502, proxyRes.headers)
      proxyRes.pipe(res)
    })

    proxyReq.on("error", (error) => {
      if (res.headersSent) {
        res.destroy(error)
        return
      }
      void respondError(res, 502, error)
    })

    req.pipe(proxyReq)
    return
  }

  // ── Everything else → Vite ───────────────────────────────────────
  vite.middlewares(req, res)
})

server.listen(PORT, () => {
  console.log(`[dev] Server running at http://localhost:${PORT}`)
})

server.on("close", () => {
  void Runtime.runPromise(rt)(Scope.close(processManagerScope, Exit.void))
})
