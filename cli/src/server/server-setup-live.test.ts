import { mkdir, mkdtemp, rm, writeFile } from "node:fs/promises"
import { tmpdir } from "node:os"
import { join } from "node:path"
import { Effect } from "effect"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { LOCAL_CONTROL_HEADER, MAGNITUDE_RPC_VERSION } from "@magnitudedev/sdk"

const TOKEN = "control-token-for-test"

/** A system service that admits loopback control only with the current start's token. */
const startService = () => {
  let enabled = false
  const calls: Array<{ readonly path: string; readonly token: string | null; readonly tag?: string }> = []
  const settings = () => ({ enabled, requireApiKey: true, port: 10100, pending: false, interfaces: [{ name: "eth0", address: "192.168.1.5", kind: "lan" }],
    ...(enabled ? { apiKey: "mag-network-key" } : {}) })
  const server = Bun.serve({ port: 0, hostname: "127.0.0.1", fetch: async request => {
    const path = new URL(request.url).pathname
    const token = request.headers.get(LOCAL_CONTROL_HEADER)
    if (path === "/health") {
      calls.push({ path, token })
      return Response.json(token === TOKEN
        ? { service: "magnitude-acn", version: "0.0.1", revision: 1, rpcVersion: MAGNITUDE_RPC_VERSION, id: "instance", pid: 1, state: { _tag: "Ready" } }
        : { service: "magnitude-acn", version: "0.0.1", state: { _tag: "Ready" } })
    }
    const message = JSON.parse((await request.text()).trim()) as { readonly id: string; readonly tag: string }
    calls.push({ path, token, tag: message.tag })
    if (token !== TOKEN) return new Response(null, { status: 401 })
    if (message.tag.endsWith("SetNetworkAccess")) enabled = true
    return new Response(`${JSON.stringify({ _tag: "Exit", requestId: message.id, exit: { _tag: "Success", value: settings() } })}\n`)
  } })
  return { server, calls }
}

let root: string
let service: ReturnType<typeof startService>
beforeEach(async () => {
  root = await mkdtemp(join(tmpdir(), "magnitude-setup-control-"))
  service = startService()
  vi.resetModules()
  vi.stubEnv("MAGNITUDE_DEV_DATA_DIR", join(root, "data"))
  vi.stubEnv("MAGNITUDE_DESKTOP_STATE_DIR", join(root, "state"))
  vi.stubEnv("MAGNITUDE_DEV_PORT", String(service.server.port))
})
afterEach(async () => {
  service.server.stop(true)
  vi.unstubAllEnvs()
  await rm(root, { recursive: true, force: true })
})

const enableNetworkAccess = async () => {
  const { ServerSetupHost } = await import("./server-setup")
  const { ServerSetupHostLive } = await import("./server-setup-live")
  return Effect.runPromise(Effect.flatMap(ServerSetupHost, host => host.enableNetworkAccess).pipe(Effect.provide(ServerSetupHostLive), Effect.either))
}

describe.skipIf(process.platform === "win32")("server setup against a service that requires its local control token", () => {
  it("turns on network access with the token the service issued", async () => {
    await mkdir(join(root, "state"), { recursive: true })
    await writeFile(join(root, "state", "control-token"), `${TOKEN}\n`, { mode: 0o600 })
    const result = await enableNetworkAccess()
    expect(result._tag === "Right" && result.right).toEqual({ addresses: ["192.168.1.5"], port: 10100, key: "mag-network-key" })
    expect(service.calls.some(call => call.tag?.endsWith("SetNetworkAccess"))).toBe(true)
    expect(service.calls.every(call => call.token === TOKEN)).toBe(true)
  })

  it("stops at once, without turning anything on, when this account cannot present the token", async () => {
    const result = await enableNetworkAccess()
    expect(result._tag === "Left" && result.left.message).toContain("can't control it yet")
    expect(service.calls.map(call => call.path)).toEqual(["/health"])
  })
})
