import { BunContext, BunHttpServer } from "@effect/platform-bun"
import { mkdtemp, rm } from "node:fs/promises"
import { tmpdir } from "node:os"
import { join } from "node:path"
import type { MagnitudeHealthResponse } from "@magnitudedev/acn-protocol"
import { FetchHttpClient, HttpBody, HttpClient, HttpClientRequest, HttpServer, HttpServerRequest, HttpServerResponse } from "@effect/platform"
import * as HttpLayerRouter from "@effect/platform/HttpLayerRouter"
import { Rpc, RpcGroup, RpcSerialization, RpcServer } from "@effect/rpc"
import { Cause, Context, Effect, Layer, Option, Schema, Stream } from "effect"
import { networkInterfaces } from "node:os"
import type { NetworkAccess } from "@magnitudedev/storage"
import { IcnBinaryNotFound } from "@magnitudedev/icn"
import { describe, expect, it, vi } from "vitest"
import { ACN_INSTANCE_ID } from "./identity"
import { makeAcnServiceLifecycle } from "./service-lifecycle"
import { ACN_PUBLIC_PORT, acnStartupFailureDetail, installAcnHealthRoutes, installAcnPublicRoutes, launchAcnServer } from "./server"

describe("ACN startup failure presentation", () => {
  it("finishes a delayed stopping report after application acquisition fails", async () => {
    const root = await mkdtemp(join(tmpdir(), "magnitude-health-failure-"))
    const delivered: MagnitudeHealthResponse[] = []
    vi.stubEnv("MAGNITUDE_ICN_PATH", join(root, "absent-installation.json"))
    try {
      const result = await Effect.runPromise(launchAcnServer({ dataDir: root, port: 0 }, {
        awaitStart: Effect.void, awaitShutdown: Effect.never,
        reportHealth: health => Effect.sleep(health.state._tag === "Stopping" ? "25 millis" : "0 millis").pipe(
          Effect.zipRight(Effect.sync(() => { delivered.push(health) })),
        ),
      }).pipe(Effect.exit, Effect.provide([BunContext.layer, FetchHttpClient.layer])))
      expect(result._tag).toBe("Failure")
      const terminal = delivered.at(-1)?.state
      expect(terminal?._tag).toBe("Stopping")
      if (terminal?._tag === "Stopping") {
        expect(terminal.reason).toBe("startup-failed")
        expect(terminal.safeDetail._tag).toBe("Some")
        if (terminal.safeDetail._tag === "Some") {
          expect(terminal.safeDetail.value).toContain("not found")
          expect(terminal.safeDetail.value).not.toContain("\n")
        }
      }
    } finally {
      vi.unstubAllEnvs()
      await rm(root, { recursive: true, force: true })
    }
  })
  it("preserves the actionable native error without its stack", () => {
    const error = new IcnBinaryNotFound({ path: "/isolated/bin/magnitude-inference" })
    expect(acnStartupFailureDetail(Cause.fail(error))).toBe(error.message)
  })
  it("bounds multiline messages and keeps defects in diagnostics", () => {
    expect(acnStartupFailureDetail(Cause.fail(new Error("Engine unavailable\nprivate diagnostic stack")))).toBe("Engine unavailable")
    expect(acnStartupFailureDetail(Cause.fail(new Error("x".repeat(2000))))).toHaveLength(500)
    expect(acnStartupFailureDetail(Cause.die(new Error("private defect")))).toBe("Magnitude service could not start. See diagnostics for details.")
  })
})

const TestRpcs = RpcGroup.make(
  Rpc.make("Ping", { success: Schema.String }),
  Rpc.make("Watch", { success: Schema.String, stream: true }),
)

const listen = (router: HttpLayerRouter.HttpRouter, port: number, hostname = "127.0.0.1") => Effect.gen(function* () {
  const infrastructure = yield* Layer.build(BunHttpServer.layer({
    hostname, port, idleTimeout: 0,
  }))
  const server = Context.get(infrastructure, HttpServer.HttpServer)
  yield* server.serve(router.asHttpEffect()).pipe(Effect.provide(infrastructure))
  if (server.address._tag !== "TcpAddress") return yield* Effect.dieMessage("Expected TCP")
  return server.address.port
})
const loopbackOrigin = (port: number) => `http://127.0.0.1:${port}`
const externalAddress = () => Object.values(networkInterfaces()).flat()
  .find(entry => entry !== undefined && entry.family === "IPv4" && !entry.internal)?.address

describe("ACN public HTTP listener", () => {
  it("serves fenced RPC and inference with shared lifecycle health, without a shutdown listener", async () => {
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const http = yield* HttpClient.HttpClient
      const lifecycle = yield* makeAcnServiceLifecycle()
      const icn = yield* HttpLayerRouter.make
      yield* icn.add("GET", "/v1/models", Effect.gen(function* () {
        const request = yield* HttpServerRequest.HttpServerRequest
        expect(request.headers.authorization).toBe("Bearer private-icn")
        return HttpServerResponse.text("inference models")
      }))
      const icnOrigin = loopbackOrigin(yield* listen(icn, 0))
      const publicRouter = yield* HttpLayerRouter.make
      yield* installAcnHealthRoutes(publicRouter, lifecycle)
      yield* installAcnPublicRoutes(publicRouter, lifecycle, {
        origin: new URL(icnOrigin),
        clientOptions: { headers: { authorization: "Bearer private-icn" } },
      })
      const origin = loopbackOrigin(yield* listen(publicRouter, 0))
      expect(ACN_PUBLIC_PORT).toBe(10100)
      expect(yield* (yield* http.get(`${origin}/`)).text).toContain("/inference/v1")

      const rpc = (base: string, id: string | undefined, tag = "Ping") => http.execute(
        HttpClientRequest.post(`${base}/rpc`, {
          headers: id === undefined ? {} : { "x-magnitude-acn-id": id },
          body: HttpBody.text(`${JSON.stringify({
            _tag: "Request", id: "1", tag, payload: {}, headers: [],
          })}\n`, "application/ndjson"),
        }),
      )
      expect((yield* http.get(`${origin}/health`)).status).toBe(503)
      expect((yield* rpc(origin, ACN_INSTANCE_ID)).status).toBe(503)

      let dispatched = 0
      const rpcRouter = yield* HttpLayerRouter.make
      const protocol = yield* RpcServer.makeProtocolHttpRouter({ path: "/rpc" }).pipe(
        Effect.provideService(HttpLayerRouter.HttpRouter, rpcRouter),
        Effect.provide(RpcSerialization.layerNdjson),
      )
      yield* RpcServer.make(TestRpcs).pipe(
        Effect.provide(TestRpcs.toLayer({
          Ping: () => Effect.sync(() => { dispatched += 1; return "pong" }),
          Watch: () => Stream.make("first", "second"),
        })),
        Effect.provideService(RpcServer.Protocol, protocol),
        Effect.forkScoped,
      )
      yield* lifecycle.becomeReady(rpcRouter.asHttpEffect().pipe(Effect.orDie))

      expect((yield* http.get(`${origin}/health`)).status).toBe(200)
      expect((yield* rpc(origin, undefined)).status).toBe(409)
      expect((yield* rpc(origin, "previous-instance")).status).toBe(409)
      expect(dispatched).toBe(0)
      const pong = yield* rpc(origin, ACN_INSTANCE_ID)
      expect(pong.status).toBe(200)
      expect(yield* pong.text).toContain('"value":"pong"')
      expect(dispatched).toBe(1)
      const watch = yield* rpc(origin, ACN_INSTANCE_ID, "Watch")
      const events = yield* watch.text
      expect(events).toContain('"_tag":"Chunk"')
      expect(events).toContain("first")
      expect(events).toContain("second")
      const models = yield* http.get(`${origin}/inference/v1/models`)
      expect(models.status).toBe(200)
      expect(yield* models.text).toBe("inference models")
      expect((yield* http.get(`${origin}/inference/api/v1/models`)).status).toBe(404)

      expect((yield* http.post(`${origin}/shutdown`)).status).toBe(404)
      yield* lifecycle.beginStopping({ reason: "administrative" })
      expect((yield* http.get(`${origin}/health`)).status).toBe(503)
      expect((yield* rpc(origin, ACN_INSTANCE_ID)).status).toBe(503)
      expect(dispatched).toBe(1)
    })).pipe(Effect.provide(FetchHttpClient.layer)))
  })
})

describe("ACN network access", () => {
  const network: NetworkAccess = {
    enabled: true, bind: "0.0.0.0", apiKey: Option.some("mag-test-key"), requireApiKey: true, allowedHosts: ["my-mac.local"], warning: Option.none(),
  }
  it("keeps loopback callers unchanged and gates remote callers by key, host, and route", async () => {
    const external = externalAddress()
    if (external === undefined) return
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const http = yield* HttpClient.HttpClient
      const lifecycle = yield* makeAcnServiceLifecycle()
      const icn = yield* HttpLayerRouter.make
      yield* icn.add("GET", "/v1/models", Effect.succeed(HttpServerResponse.text("inference models")))
      const icnOrigin = loopbackOrigin(yield* listen(icn, 0))
      const publicRouter = yield* HttpLayerRouter.make
      yield* installAcnHealthRoutes(publicRouter, lifecycle, network)
      yield* installAcnPublicRoutes(publicRouter, lifecycle, { origin: new URL(icnOrigin), clientOptions: { headers: {} } })
      const port = yield* listen(publicRouter, 0, "0.0.0.0")
      const rpcRouter = yield* HttpLayerRouter.make
      yield* lifecycle.becomeReady(rpcRouter.asHttpEffect().pipe(Effect.orDie))
      const local = loopbackOrigin(port)
      const remote = `http://${external}:${port}`
      const get = (url: string, headers: Record<string, string> = {}) => http.execute(HttpClientRequest.get(url, { headers }))
      const rpc = (base: string) => http.execute(HttpClientRequest.post(`${base}/rpc`, { headers: { "x-magnitude-acn-id": ACN_INSTANCE_ID }, body: HttpBody.text("{}\n", "application/ndjson") }))

      const localHealth = yield* get(`${local}/health`)
      expect(localHealth.status).toBe(200)
      expect(yield* localHealth.text).toContain('"id"')
      expect((yield* get(`${local}/inference/v1/models`)).status).toBe(200)
      expect((yield* rpc(local)).status).not.toBe(403)

      const remoteHealth = yield* get(`${remote}/health`)
      expect(remoteHealth.status).toBe(200)
      expect(yield* remoteHealth.text).not.toContain('"id"')
      expect((yield* rpc(remote)).status).toBe(403)
      const unauthenticated = yield* get(`${remote}/inference/v1/models`)
      expect(unauthenticated.status).toBe(401)
      expect(unauthenticated.headers["www-authenticate"]).toContain("Bearer")
      expect((yield* get(`${remote}/inference/v1/models`, { authorization: "Bearer magnitude-local" })).status).toBe(401)
      const bearer = yield* get(`${remote}/inference/v1/models`, { authorization: "Bearer mag-test-key" })
      expect(bearer.status).toBe(200)
      expect(yield* bearer.text).toBe("inference models")
      expect((yield* get(`${remote}/inference/v1/models`, { "x-api-key": "mag-test-key" })).status).toBe(200)

      const preflight = yield* http.execute(HttpClientRequest.options(`${remote}/inference/v1/models`, { headers: { origin: "http://localhost:3000" } }))
      expect(preflight.status).toBe(204)
      expect(preflight.headers["access-control-allow-origin"]).toBe("http://localhost:3000")
      expect((yield* http.execute(HttpClientRequest.options(`${remote}/inference/v1/models`, { headers: { origin: "http://evil.com" } }))).status).toBe(403)
      expect((yield* get(`${remote}/health`, { host: "evil.com" })).status).toBe(421)
      expect((yield* get(`${remote}/health`, { host: "my-mac.local:1" })).status).toBe(200)
      expect((yield* get(`${remote}/health`, { host: "host.docker.internal:1" })).status).toBe(200)
      expect((yield* get(`${remote}/health`, { host: "mac.tail1234.ts.net" })).status).toBe(200)
    })).pipe(Effect.provide(FetchHttpClient.layer)))
  })
  it("refuses non-local hosts and never gates while loopback only", async () => {
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const http = yield* HttpClient.HttpClient
      const lifecycle = yield* makeAcnServiceLifecycle()
      const publicRouter = yield* HttpLayerRouter.make
      yield* installAcnHealthRoutes(publicRouter, lifecycle)
      const origin = loopbackOrigin(yield* listen(publicRouter, 0))
      const get = (headers: Record<string, string>) => http.execute(HttpClientRequest.get(`${origin}/health`, { headers }))
      expect((yield* get({ host: "192.168.1.2:1" })).status).toBe(421)
      expect((yield* get({ host: "host.docker.internal" })).status).toBe(421)
      expect((yield* get({ host: "localhost:1" })).status).toBe(503)
    })).pipe(Effect.provide(FetchHttpClient.layer)))
  })
})
