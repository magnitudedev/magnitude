/**
 * Dev server — the single server for `bun web`.
 *
 * One process, one port. This server:
 * 1. Exposes ACN ensurance
 * 2. Serves the web app via Vite's middleware
 * 3. Exposes one cancellable ensure stream
 * 4. Proxies RPC, health, and logs to the fixed daemon endpoint; RPC retains the SDK instance fence
 *
 * The browser talks only to this same-origin server.
 */
import http, { createServer, type ServerResponse } from "node:http"
import { createServer as createViteServer } from "vite"
import { Effect, Exit, Fiber, Layer, Runtime, Option, Schema, Scope, Stream } from "effect"
import { FetchHttpClient } from "@effect/platform"
import { BunContext } from "@effect/platform-bun"
import { makeLocalAcnInstanceManager, BunDetachedChildProcessSpawner, ChildProcessSpawner, DAEMON_TARGET, makeServiceStarter } from "@magnitudedev/daemon-management"
import { MAGNITUDE_SERVICE_ORIGIN, ServiceStartFailed } from "@magnitudedev/sdk"
import { RemoteServiceStartMessageSchema, type RemoteServiceStartMessage } from "@magnitudedev/client-common"
import { BunSqliteDriverLayer } from "@magnitudedev/daemon-management/bun"
import { resolve } from "node:path"

// ─── Daemon host boundaries ─────────────────────────────────────────────────

const rt = Runtime.defaultRuntime
const ensurerScope = await Runtime.runPromise(rt)(Scope.make())
const acnSourcePath = resolve(import.meta.dir, "..", "..", "packages", "acn", "src", "binary.ts")
const developmentDataDir = process.env.MAGNITUDE_DEV_DATA_DIR


async function createEnsurer() {
  return Runtime.runPromise(rt)(makeLocalAcnInstanceManager({
    dataDir: developmentDataDir,
    launchOverride: {
      target: DAEMON_TARGET,
      command: ["bun", acnSourcePath, "serve"],
    },
  }).pipe(
    Effect.provideService(ChildProcessSpawner, BunDetachedChildProcessSpawner),
    Effect.provideService(Scope.Scope, ensurerScope),
    Effect.provide(Layer.mergeAll(FetchHttpClient.layer, BunContext.layer, BunSqliteDriverLayer)),
  ))
}

const managerPromise = createEnsurer()


// ─── Dev-mode launch command ────────────────────────────────────────────────

const encodeStartMessage = Schema.encode(Schema.parseJson(RemoteServiceStartMessageSchema))

const respondError = async (
  res: ServerResponse,
  status: number,
  error: unknown,
): Promise<void> => {
  res.writeHead(status, { "Content-Type": "application/json" })
  res.end(JSON.stringify({ error: String(error) }))
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
  if (url.pathname === "/service/start" && req.method === "POST") {
    try {
      const manager = await managerPromise
      res.writeHead(200, {
        "Content-Type": "application/x-ndjson",
        "Cache-Control": "no-store",
      })
      const write = (message: RemoteServiceStartMessage) => {

        return encodeStartMessage(message).pipe(
          Effect.flatMap((encoded) =>
            Effect.sync(() => {
              if (!res.destroyed) res.write(`${encoded}\n`)
            }),
          ),
          // Every message originates from the schema-typed instance manager. An
          // encode failure is therefore an internal invariant violation, not an
          // ensurance-domain failure that can be represented on this same stream.
          Effect.orDie,
        )
      }
      const observer = Runtime.runFork(rt)(makeServiceStarter(manager).start.pipe(
        Stream.runForEach(write),
        Effect.catchAll((error) => write({ _tag: "Failed", error })),
      ))
      const interrupt = () => { Runtime.runFork(rt)(Fiber.interrupt(observer)) }
      res.once("close", interrupt)
      await Runtime.runPromise(rt)(Fiber.await(observer))
      res.off("close", interrupt)
      if (!res.destroyed) res.end()
    } catch (err) {
      if (res.headersSent) res.destroy(err instanceof Error ? err : new Error(String(err)))
      else await respondError(res, 500, err)
    }
    return
  }

  // ── Same-origin ACN proxy (streaming) ───────────────────────────
  const proxyMatch = url.pathname.match(/^\/acn(\/rpc|\/health|\/logs)$/)
  if (proxyMatch) {
    const targetPath = proxyMatch[1]!
    const target = new URL(MAGNITUDE_SERVICE_ORIGIN)
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
      if (targetPath === "/health") res.destroy(error)
      else void respondError(res, 502, error)
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
  void Runtime.runPromise(rt)(Scope.close(ensurerScope, Exit.void))
})
