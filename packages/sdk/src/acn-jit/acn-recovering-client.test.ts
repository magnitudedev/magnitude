import { describe, expect, it } from "vitest"
import { Effect, Exit, Layer, Option, Schema, Stream } from "effect"
import { Rpc, RpcClient } from "@effect/rpc"
import * as HttpClient from "@effect/platform/HttpClient"
import * as HttpClientError from "@effect/platform/HttpClientError"
import * as HttpClientRequest from "@effect/platform/HttpClientRequest"
import * as HttpClientResponse from "@effect/platform/HttpClientResponse"
import { AcnOwnerIdSchema, MagnitudeRpcs, SessionNotFound } from "@magnitudedev/acn-protocol"
import { DaemonDiscovery, type DaemonDiscovery as DaemonDiscoveryService } from "./daemon-discovery"
import { DaemonLauncher, type DaemonLauncher as DaemonLauncherService } from "./daemon-launcher"
import { makeAcnJitRuntime } from "./acn-recovering-client"
import { DaemonSpawnFailed } from "./errors"
import type { AcnClient } from "../protocol"
import { SDK_VERSION } from "../version"

const getRpc = (tag: string) => {
  const rpc = MagnitudeRpcs.requests.get(tag)
  if (!rpc) throw new Error(`no rpc ${tag}`)
  return rpc
}

const encodeExitFor = (tag: string, exit: Exit.Exit<unknown, unknown>): unknown =>
  Schema.encodeUnknownSync(Rpc.exitSchema(getRpc(tag)))(exit)

const requestText = (request: HttpClientRequest.HttpClientRequest): string => {
  const body = request.body
  if (body._tag === "Uint8Array") return new TextDecoder().decode(body.body)
  if (body._tag === "Raw" && typeof body.body === "string") return body.body
  throw new Error(`unexpected request body: ${body._tag}`)
}

const extractRequestId = (request: HttpClientRequest.HttpClientRequest): string => {
  const parsed = Schema.decodeUnknownSync(
    Schema.Struct({ id: Schema.String }),
  )(JSON.parse(requestText(request).split("\n")[0]))
  return parsed.id
}

type Attempt =
  | { readonly kind: "refuse" }
  | { readonly kind: "lines"; readonly make: (requestId: string) => ReadonlyArray<unknown> }

const makeFakeHttp = (attempts: ReadonlyArray<Attempt>) => {
  let calls = 0
  const client = HttpClient.make((request) =>
    Effect.suspend(() => {
      const attempt = attempts[Math.min(calls, attempts.length - 1)]
      calls++
      if (attempt.kind === "refuse") {
        return Effect.fail(new HttpClientError.RequestError({
          request,
          reason: "Transport",
          cause: new Error("connection refused"),
        }))
      }
      const requestId = extractRequestId(request)
      const body = `${attempt.make(requestId).map((line) => JSON.stringify(line)).join("\n")}\n`
      return Effect.succeed(HttpClientResponse.fromWeb(request, new Response(body)))
    }),
  )
  return { client, calls: () => calls }
}

const makeFakeProcesses = (options: {
  readonly current: ReadonlyArray<Option.Option<string>>
  readonly startUrl?: string
}) => {
  const endpoint = (url: string) => ({
    id: AcnOwnerIdSchema.make(url),
    version: SDK_VERSION,
    url,
  })
  let currentCalls = 0
  let startCalls = 0
  const discovery: DaemonDiscoveryService = {
    current: () => Effect.sync(() => {
      const result = options.current[Math.min(currentCalls, options.current.length - 1)]
      currentCalls++
      return Option.map(result, (url) => ({
        ...endpoint(url),
        pid: 123,
        state: { _tag: "Ready" as const },
      }))
    }),
  }
  const launcher: DaemonLauncherService = {
    launch: () =>
      Stream.suspend(() => {
        startCalls++
        return options.startUrl === undefined
          ? Stream.fail(
              new DaemonSpawnFailed({ reason: "start disabled in test" }),
            )
          : Stream.succeed({ _tag: "Ready", endpoint: endpoint(options.startUrl) })
      }),
  }
  return { services: { discovery, launcher }, currentCalls: () => currentCalls, startCalls: () => startCalls }
}

const withClient = <A, E>(
  services: { readonly discovery: DaemonDiscoveryService; readonly launcher: DaemonLauncherService },
  http: HttpClient.HttpClient,
  use: (client: AcnClient) => Effect.Effect<A, E>,
): Promise<A> =>
  Effect.runPromise(Effect.scoped(Effect.gen(function* () {
    const runtime = yield* makeAcnJitRuntime().pipe(
      Effect.provideService(DaemonDiscovery, services.discovery),
      Effect.provideService(DaemonLauncher, services.launcher),
    )
    const client = yield* RpcClient.make(MagnitudeRpcs).pipe(
      Effect.provide(runtime.protocolLayer.pipe(
        Layer.provide(Layer.succeed(HttpClient.HttpClient, http)),
      )),
    )
    return yield* use(client)
  })))

const exitMessage = (tag: string, requestId: string, exit: Exit.Exit<unknown, unknown>) => ({
  _tag: "Exit",
  requestId,
  exit: encodeExitFor(tag, exit),
})

const chunkMessage = (requestId: string, values: ReadonlyArray<unknown>) => ({
  _tag: "Chunk",
  requestId,
  values,
})

const payload = (event: string, path: string) => ({
  _tag: "payload" as const,
  payload: { event, path },
})

const endOfStream = (id: string) =>
  exitMessage("WatchFile", id, Exit.fail(new SessionNotFound({ sessionId: "s" })))

const collectPaths = (client: AcnClient) =>
  client.WatchFile({ cwd: "/project", path: "/watched" }).pipe(
    Stream.catchAll(() => Stream.empty),
    Stream.runCollect,
    Effect.map((events) => Array.from(events, (event) => event.path)),
  )

describe("AcnJitRuntime", () => {
  it("performs read-only discovery without starting a daemon", async () => {
    const { services, currentCalls, startCalls } = makeFakeProcesses({
      current: [Option.some("http://daemon")],
    })
    const runtime = await Effect.runPromise(
      makeAcnJitRuntime().pipe(
        Effect.provideService(DaemonDiscovery, services.discovery),
        Effect.provideService(DaemonLauncher, services.launcher),
      ),
    )

    expect(await Effect.runPromise(runtime.startup.prepare)).toEqual({
      _tag: "Ready",
      endpoint: {
        id: "http://daemon",
        url: "http://daemon",
        version: SDK_VERSION,
      },
    })
    expect(await Effect.runPromise(runtime.startup.state.get)).toEqual({
      _tag: "Ready",
      endpoint: {
        id: "http://daemon",
        url: "http://daemon",
        version: SDK_VERSION,
      },
    })
    expect(currentCalls()).toBe(1)
    expect(startCalls()).toBe(0)
  })

  it("performs startup demand once and shares one selected endpoint", async () => {
    const { services, currentCalls, startCalls } = makeFakeProcesses({
      current: [Option.some("http://daemon")],
    })
    const { client: http } = makeFakeHttp([
      { kind: "lines", make: (id) => [exitMessage("CheckFileExists", id, Exit.succeed(true))] },
    ])

    const runtime = await Effect.runPromise(
      makeAcnJitRuntime().pipe(
        Effect.provideService(DaemonDiscovery, services.discovery),
        Effect.provideService(DaemonLauncher, services.launcher),
      ),
    )
    const call = Effect.scoped(RpcClient.make(MagnitudeRpcs).pipe(
      Effect.provide(runtime.protocolLayer.pipe(
        Layer.provide(Layer.succeed(HttpClient.HttpClient, http)),
      )),
      Effect.flatMap((client) => client.CheckFileExists({ cwd: "/project", path: "/x" })),
    ))

    expect(await Effect.runPromise(call)).toBe(true)
    expect(await Effect.runPromise(call)).toBe(true)
    expect(currentCalls()).toBe(1)
    expect(startCalls()).toBe(0)
  })

  it("recovers finite work through the client lifecycle owner", async () => {
    const { services, startCalls } = makeFakeProcesses({
      current: [Option.some("http://dead"), Option.none()],
      startUrl: "http://fresh",
    })
    const { client, calls } = makeFakeHttp([
      { kind: "refuse" },
      { kind: "lines", make: (id) => [exitMessage("CheckFileExists", id, Exit.succeed(true))] },
    ])

    const result = await withClient(services, client, (acn) =>
      acn.CheckFileExists({ cwd: "/project", path: "/x" }),
    )
    expect(result).toBe(true)
    expect(calls()).toBe(2)
    expect(startCalls()).toBe(1)
  })

  it("automatically exposes payloads while consuming subscription controls", async () => {
    const { services } = makeFakeProcesses({ current: [Option.some("http://daemon")] })
    const { client } = makeFakeHttp([{ kind: "lines", make: (id) => [
      chunkMessage(id, [{ _tag: "keepalive" }]),
      chunkMessage(id, [
        { _tag: "suspended", reason: "session-offloaded" },
        payload("changed", "/real"),
      ]),
      endOfStream(id),
    ] }])

    expect(await withClient(services, client, collectPaths)).toEqual(["/real"])
  })

  it("parks on terminated and reconnects to a successor without spawning", async () => {
    const { services, startCalls } = makeFakeProcesses({
      current: [Option.some("http://old"), Option.some("http://new")],
      startUrl: "http://must-not-start",
    })
    const { client } = makeFakeHttp([
      { kind: "lines", make: (id) => [
        chunkMessage(id, [payload("changed", "/before")]),
        chunkMessage(id, [{ _tag: "terminated", reason: "acn-shutdown" }]),
      ] },
      { kind: "lines", make: (id) => [
        chunkMessage(id, [payload("changed", "/after")]),
        endOfStream(id),
      ] },
    ])

    expect(await withClient(services, client, collectPaths)).toEqual(["/before", "/after"])
    expect(startCalls()).toBe(0)
  })
})
