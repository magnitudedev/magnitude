/**
 * Dev server — the single server for `bun web`.
 *
 * One process, one port. This server:
 * 1. Ensures a daemon on startup (via makeLocalDaemonSpawner + Bun.spawn)
 * 2. Serves the web app via Vite's middleware
 * 3. Handles /discover + /spawn (daemon lifecycle)
 * 4. Proxies /rpc, /health, /logs to the daemon
 *
 * The browser talks to one endpoint. No Vite proxy config, no port wiring.
 * This is "the thing that never dies and has spawn capability."
 */
import http, { createServer } from "node:http"
import { createServer as createViteServer } from "vite"
import { Effect, Layer, Runtime, Option } from "effect"
import { FetchHttpClient } from "@effect/platform"
import { BunContext } from "@effect/platform-bun"
import {
  makeLocalDaemonSpawner,
  type SpawnProcess,
} from "@magnitudedev/sdk"
import { resolve } from "node:path"

// ─── Bun spawner ──────────────────────────────────────────────────────

const bunSpawn: SpawnProcess = (command) => {
  const proc = Bun.spawn({
    cmd: command,
    detached: true,
    stdio: ["ignore", "ignore", "ignore"],
    env: process.env,
  })
  proc.unref()
  return { pid: proc.pid, exited: proc.exited }
}

// ─── Spawner for proxy endpoints ──────────────────────────────────────

const rt = Runtime.defaultRuntime

async function getSpawner() {
  return Runtime.runPromise(rt)(makeLocalDaemonSpawner(bunSpawn).pipe(
    Effect.provide(Layer.mergeAll(FetchHttpClient.layer, BunContext.layer)),
  ))
}

// ─── Daemon URL ───────────────────────────────────────────────────────

let daemonUrl: string = ""

async function discoverDaemonUrl(): Promise<string | null> {
  const spawner = await getSpawner()
  const result = await Runtime.runPromise(rt)(spawner.discover())
  return Option.getOrElse(result, () => null)
}

async function spawnDaemon(command: string[] | undefined): Promise<string> {
  const spawner = await getSpawner()
  return Runtime.runPromise(rt)(spawner.spawn(command ?? defaultSpawnCommand))
}

// ─── Dev-mode spawn command ───────────────────────────────────────────

const acnSourcePath = resolve(import.meta.dir, "..", "..", "packages", "acn", "src", "binary.ts")
const defaultSpawnCommand = ["bun", acnSourcePath, "serve", "--register"]

// ─── Ensure daemon on startup ─────────────────────────────────────────

console.log("[dev] Ensuring ACN daemon...")
daemonUrl = (await discoverDaemonUrl()) ?? ""
if (!daemonUrl) {
  try {
    daemonUrl = await spawnDaemon(defaultSpawnCommand)
  } catch (err) {
    console.error("[dev] Failed to spawn daemon:", String(err))
  }
}
if (daemonUrl) {
  console.log(`[dev] Daemon at ${daemonUrl}`)
} else {
  console.warn("[dev] No daemon available — /spawn will start one on demand")
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

  // ── Daemon lifecycle endpoints ───────────────────────────────────
  if (url.pathname === "/discover" && req.method === "GET") {
    try {
      const found = await discoverDaemonUrl()
      if (found) daemonUrl = found
      res.writeHead(200, { "Content-Type": "application/json" })
      res.end(JSON.stringify({ url: found }))
    } catch (err) {
      res.writeHead(500, { "Content-Type": "application/json" })
      res.end(JSON.stringify({ error: String(err) }))
    }
    return
  }

  if (url.pathname === "/spawn" && req.method === "POST") {
    try {
      const chunks: Buffer[] = []
      for await (const chunk of req) chunks.push(chunk as Buffer)
      const raw = Buffer.concat(chunks).toString()
      const body = raw ? JSON.parse(raw) as { command?: string[] | null } : {}
      const spawnedUrl = await spawnDaemon(body.command ?? undefined)
      daemonUrl = spawnedUrl
      res.writeHead(200, { "Content-Type": "application/json" })
      res.end(JSON.stringify({ url: spawnedUrl }))
    } catch (err) {
      res.writeHead(500, { "Content-Type": "application/json" })
      res.end(JSON.stringify({ error: String(err) }))
    }
    return
  }

  // ── Proxy RPC to daemon (streaming) ──────────────────────────────
  if (url.pathname === "/rpc" || url.pathname === "/health" || url.pathname === "/logs") {
    if (!daemonUrl) {
      const found = await discoverDaemonUrl()
      if (found) {
        daemonUrl = found
      } else {
        try {
          daemonUrl = await spawnDaemon(defaultSpawnCommand)
        } catch {
          res.writeHead(503, { "Content-Type": "application/json" })
          res.end(JSON.stringify({ error: "No daemon available" }))
          return
        }
      }
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

    proxyReq.on("error", (err) => {
      daemonUrl = ""
      if (!res.headersSent) {
        res.writeHead(502, { "Content-Type": "application/json" })
        res.end(JSON.stringify({ error: String(err) }))
      }
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
