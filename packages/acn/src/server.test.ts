import { BunHttpServer } from "@effect/platform-bun"
import { FetchHttpClient, HttpBody, HttpClient, HttpClientRequest, HttpServer, HttpServerRequest, HttpServerResponse } from "@effect/platform"
import * as HttpLayerRouter from "@effect/platform/HttpLayerRouter"
import { Rpc, RpcGroup, RpcSerialization, RpcServer } from "@effect/rpc"
import { Context, Effect, Layer, Schema, Stream } from "effect"
import { describe, expect, it } from "vitest"
import { ACN_INSTANCE_ID } from "./identity"
import { makeAcnServiceLifecycle } from "./service-lifecycle"
import { ACN_PUBLIC_PORT, installAcnControlRoutes, installAcnPublicRoutes } from "./server"

const TestRpcs = RpcGroup.make(
  Rpc.make("Ping", { success: Schema.String }),
  Rpc.make("Watch", { success: Schema.String, stream: true }),
)

const listen = (router: HttpLayerRouter.HttpRouter, port: number) => Effect.gen(function* () {
  const infrastructure = yield* Layer.build(BunHttpServer.layer({
    hostname: "127.0.0.1", port, idleTimeout: 0,
  }))
  const server = Context.get(infrastructure, HttpServer.HttpServer)
  yield* server.serve(router.asHttpEffect()).pipe(Effect.provide(infrastructure))
  if (server.address._tag !== "TcpAddress") return yield* Effect.dieMessage("Expected TCP")
  return `http://127.0.0.1:${server.address.port}`
})

describe("ACN public and control HTTP listeners", () => {
  it("serves fenced RPC only on 10100, alongside inference, with shared lifecycle health", async () => {
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const http = yield* HttpClient.HttpClient
      const lifecycle = yield* makeAcnServiceLifecycle()
      const control = yield* HttpLayerRouter.make
      yield* installAcnControlRoutes(control, lifecycle)
      const controlOrigin = yield* listen(control, 0)

      const icn = yield* HttpLayerRouter.make
      yield* icn.add("GET", "/v1/models", Effect.gen(function* () {
        const request = yield* HttpServerRequest.HttpServerRequest
        expect(request.headers.authorization).toBe("Bearer private-icn")
        return HttpServerResponse.text("inference models")
      }))
      const icnOrigin = yield* listen(icn, 0)
      const publicRouter = yield* HttpLayerRouter.make
      yield* installAcnPublicRoutes(publicRouter, lifecycle, {
        origin: new URL(icnOrigin),
        clientOptions: { headers: { authorization: "Bearer private-icn" } },
      })
      const origin = yield* listen(publicRouter, ACN_PUBLIC_PORT)
      expect(origin).toBe("http://127.0.0.1:10100")
      expect(controlOrigin).not.toBe(origin)

      const rpc = (base: string, id: string | undefined, tag = "Ping") => http.execute(
        HttpClientRequest.post(`${base}/rpc`, {
          headers: id === undefined ? {} : { "x-magnitude-acn-id": id },
          body: HttpBody.text(`${JSON.stringify({
            _tag: "Request", id: "1", tag, payload: {}, headers: [],
          })}\n`, "application/ndjson"),
        }),
      )
      expect((yield* http.get(`${controlOrigin}/health`)).status).toBe(503)
      expect((yield* http.get(`${origin}/health`)).status).toBe(503)
      expect((yield* rpc(origin, ACN_INSTANCE_ID)).status).toBe(503)
      expect((yield* rpc(controlOrigin, ACN_INSTANCE_ID)).status).toBe(404)

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

      expect((yield* http.get(`${controlOrigin}/health`)).status).toBe(200)
      expect((yield* http.get(`${origin}/health`)).status).toBe(200)
      expect((yield* rpc(controlOrigin, ACN_INSTANCE_ID)).status).toBe(404)
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
      expect((yield* http.post(`${controlOrigin}/shutdown`)).status).toBe(202)
      expect((yield* http.get(`${origin}/health`)).status).toBe(503)
      expect((yield* http.get(`${controlOrigin}/health`)).status).toBe(503)
      expect((yield* rpc(origin, ACN_INSTANCE_ID)).status).toBe(503)
      expect(dispatched).toBe(1)
    })).pipe(Effect.provide(FetchHttpClient.layer)))
  })
})
