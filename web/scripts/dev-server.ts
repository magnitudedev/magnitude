/**
 * Dev server — the single server for `bun web`.
 *
 * One process, one port. This server:
 * 1. Exposes daemon discovery and launch
 * 2. Serves the web app via Vite's middleware
 * 3. Handles /current + /launch
 * 4. Proxies /rpc, /health, and /logs to the authoritative ACN
 *
 * The browser talks only to this same-origin server.
 */
import http, { createServer, type ServerResponse } from "node:http"
import { createServer as createViteServer } from "vite"
import { Effect, Layer, Runtime, Option, Schema, Stream } from "effect"
import { FetchHttpClient } from "@effect/platform"
import { BunContext } from "@effect/platform-bun"
import {
  makeLocalDaemonDiscovery,
  makeLocalDaemonLauncher,
  BunDetachedChildProcessSpawner,
  ChildProcessSpawner,
  RemoteDaemonCurrentResponseSchema,
  RemoteDaemonErrorResponseSchema,
  RemoteDaemonLaunchRequestSchema,
  RemoteDaemonLaunchMessageSchema,
  type RemoteDaemonLaunchMessage,
} from "@magnitudedev/sdk"
import type { DaemonStatus } from "@magnitudedev/sdk"
import { resolve } from "node:path"

// ─── Daemon host boundaries ─────────────────────────────────────────────────

const rt = Runtime.defaultRuntime

async function createDiscovery() {
  return Runtime.runPromise(rt)(makeLocalDaemonDiscovery().pipe(
    Effect.provide(Layer.mergeAll(FetchHttpClient.layer, BunContext.layer)),
  ))
}

async function createLauncher() {
  return Runtime.runPromise(rt)(makeLocalDaemonLauncher().pipe(
    Effect.provideService(ChildProcessSpawner, BunDetachedChildProcessSpawner),
    Effect.provide(Layer.mergeAll(FetchHttpClient.layer, BunContext.layer)),
  ))
}

let discoveryPromise: ReturnType<typeof createDiscovery> | null = null
let launcherPromise: ReturnType<typeof createLauncher> | null = null
let daemonUrl = ""

async function currentAcn(): Promise<DaemonStatus | null> {
  discoveryPromise ??= createDiscovery()
  const discovery = await discoveryPromise
  const result = await Runtime.runPromise(rt)(discovery.current())
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
  Schema.parseJson(RemoteDaemonLaunchRequestSchema),
)
const encodeLaunchMessage = Schema.encode(
  Schema.parseJson(RemoteDaemonLaunchMessageSchema),
)
const encodeCurrentResponse = Schema.encode(
  Schema.parseJson(RemoteDaemonCurrentResponseSchema),
)
const encodeErrorResponse = Schema.encode(
  Schema.parseJson(RemoteDaemonErrorResponseSchema),
)

const respondError = async (
  res: ServerResponse,
  status: number,
  error: unknown,
): Promise<void> => {
  res.writeHead(status, { "Content-Type": "application/json" })
  res.end(await Runtime.runPromise(rt)(
    encodeErrorResponse({ error: String(error) }),
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
  if (url.pathname === "/current" && req.method === "GET") {
    try {
      const found = await currentAcn()
      if (found) daemonUrl = found.url
      res.writeHead(200, { "Content-Type": "application/json" })
      res.end(await Runtime.runPromise(rt)(encodeCurrentResponse({ daemon: found })))
    } catch (err) {
      await respondError(res, 500, err)
    }
    return
  }

  if (url.pathname === "/launch" && req.method === "POST") {
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
      const write = (message: RemoteDaemonLaunchMessage) =>
        encodeLaunchMessage(message).pipe(
          Effect.flatMap((encoded) =>
            Effect.sync(() => {
              if (message._tag === "Ready") daemonUrl = message.endpoint.url
              if (!res.destroyed) res.write(`${encoded}\n`)
            }),
          ),
        )
      try {
        launcherPromise ??= createLauncher()
        const launcher = await launcherPromise
        await Runtime.runPromise(rt)(
          launcher
            .launch(Option.orElse(body.command, () => defaultLaunchCommand))
            .pipe(
              Stream.runForEach(write),
            ),
        )
      } catch (err) {
        const message = String(err).trim() || "ACN startup failed"
        await Runtime.runPromise(rt)(write({ _tag: "Failed", message }))
      }
      res.end()
    } catch (err) {
      await respondError(res, 500, err)
    }
    return
  }

  // ── Same-origin ACN proxy (streaming) ───────────────────────────
  if (
    url.pathname === "/rpc" ||
    url.pathname === "/health" ||
    url.pathname === "/logs"
  ) {
    if (!daemonUrl) {
      try {
        const current = await currentAcn()
        if (current) daemonUrl = current.url
      } catch (error) {
        await respondError(res, 502, error)
        return
      }
    }
    if (!daemonUrl) {
      await respondError(res, 503, "No daemon available")
      return
    }

    const target = new URL(daemonUrl)
    const proxyReq = http.request({
      hostname: target.hostname,
      port: target.port,
      path: url.pathname + url.search,
      method: req.method,
      headers: req.headers,
    }, (proxyRes) => {
      res.writeHead(proxyRes.statusCode ?? 502, proxyRes.headers)
      proxyRes.pipe(res)
    })

    proxyReq.on("error", (error) => {
      daemonUrl = ""
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
