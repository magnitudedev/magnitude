import { BunContext, BunHttpServer } from "@effect/platform-bun"
import { mkdtemp, rm } from "node:fs/promises"
import { tmpdir } from "node:os"
import { join } from "node:path"
import type { MagnitudeHealthResponse } from "@magnitudedev/acn-protocol"
import { FetchHttpClient, HttpBody, HttpClient, HttpClientRequest, HttpServer, HttpServerRequest, HttpServerResponse } from "@effect/platform"
import * as HttpLayerRouter from "@effect/platform/HttpLayerRouter"
import { Rpc, RpcGroup, RpcSerialization, RpcServer } from "@effect/rpc"
import { Cause, Context, Duration, Effect, Layer, Logger, Schema, Stream } from "effect"
import { IcnBinaryNotFound, IcnExitedBeforeReady, IcnStartupRecordTimedOut, IcnReadinessTimedOut } from "@magnitudedev/icn"
import { describe, expect, it, vi } from "vitest"
import { ACN_INSTANCE_ID } from "./identity"
import { makeAcnServiceLifecycle } from "./service-lifecycle"
import { ACN_PUBLIC_PORT, acnStartupFailureDetail, acnStartupNativeOutput, logAcnStartupFailure, installAcnHealthRoutes, installAcnPublicRoutes, launchAcnServer } from "./server"

describe("ACN startup failure presentation", () => {
  it("retains bounded redacted native startup output in logs without expanding health detail", async () => {
    const error = new IcnExitedBeforeReady({ pid: 42, code: 1,
      output: 'Backend initialization failed\nAuthorization: Bearer private-bearer\nHF_TOKEN=private-hf\n{"api_key":"private-key"}\ncalibration unavailable' })
    const cause = Cause.fail(error)
    const diagnostic = acnStartupNativeOutput(cause)
    expect(diagnostic).toContain("calibration unavailable")
    expect(diagnostic).toContain("ICN process 42")
    for (const secret of ["private-bearer", "private-hf", "private-key"]) expect(diagnostic).not.toContain(secret)
    expect(acnStartupFailureDetail(cause)).toBe(error.message)
    expect(acnStartupFailureDetail(cause)).not.toContain("calibration")
    const logs: string[] = []
    await Effect.runPromise(logAcnStartupFailure(cause).pipe(Effect.provide(Logger.replace(Logger.defaultLogger, Logger.make(options => {
      logs.push(Logger.logfmtLogger.log(options))
    })))))
    expect(logs).toHaveLength(1)
    expect(logs[0]).toContain("nativeStartupOutput=")
    expect(logs[0]).toContain("calibration unavailable")
    for (const secret of ["private-bearer", "private-hf", "private-key"]) expect(logs[0]).not.toContain(secret)
    const large = new IcnExitedBeforeReady({ pid: 43, code: 1, output: `Bearer ${"s".repeat(20_000)}\n${"x".repeat(20_000)}\nlast diagnostic` })
    expect(acnStartupNativeOutput(Cause.fail(large))).toHaveLength(16_384)
    expect(acnStartupNativeOutput(Cause.fail(large))).toMatch(/last diagnostic$/)
  })
  it("admits only typed model-free startup output, including startup timeout tails", () => {
    for (const error of [new IcnStartupRecordTimedOut({ pid: 42, timeout: Duration.seconds(1), output: "record timeout detail" }),
      new IcnReadinessTimedOut({ pid: 43, timeout: Duration.seconds(1), output: "readiness timeout detail" })]) {
      expect(acnStartupNativeOutput(Cause.fail(error))).toContain(error.output)
    }
    expect(acnStartupNativeOutput(Cause.fail({ output: "private arbitrary data" }))).toBe("")
    expect(acnStartupNativeOutput(Cause.die(new Error("private defect")))).toBe("")
  })
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

const listen = (router: HttpLayerRouter.HttpRouter, port: number) => Effect.gen(function* () {
  const infrastructure = yield* Layer.build(BunHttpServer.layer({
    hostname: "127.0.0.1", port, idleTimeout: 0,
  }))
  const server = Context.get(infrastructure, HttpServer.HttpServer)
  yield* server.serve(router.asHttpEffect()).pipe(Effect.provide(infrastructure))
  if (server.address._tag !== "TcpAddress") return yield* Effect.dieMessage("Expected TCP")
  return `http://127.0.0.1:${server.address.port}`
})

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
      const icnOrigin = yield* listen(icn, 0)
      const publicRouter = yield* HttpLayerRouter.make
      yield* installAcnHealthRoutes(publicRouter, lifecycle)
      yield* installAcnPublicRoutes(publicRouter, lifecycle, {
        origin: new URL(icnOrigin),
        clientOptions: { headers: { authorization: "Bearer private-icn" } },
      })
      const origin = yield* listen(publicRouter, 0)
      expect(ACN_PUBLIC_PORT).toBe(10100)

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
