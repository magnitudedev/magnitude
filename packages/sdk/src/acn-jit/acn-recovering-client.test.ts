import { describe, expect, it } from "vitest"
import { Effect, Exit, Layer, Option, Schema, Stream } from "effect"
import { Rpc, RpcClient } from "@effect/rpc"
import * as HttpClient from "@effect/platform/HttpClient"
import * as HttpClientError from "@effect/platform/HttpClientError"
import * as HttpClientRequest from "@effect/platform/HttpClientRequest"
import * as HttpClientResponse from "@effect/platform/HttpClientResponse"
import { MagnitudeRpcs, SessionNotFound } from "@magnitudedev/protocol"
import { DaemonSpawnerTag, type DaemonSpawner } from "./daemon-spawner"
import { makeAcnJitRuntime } from "./acn-recovering-client"
import { DaemonSpawnFailed } from "./errors"
import type { AcnClient } from "../protocol"

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

const makeFakeSpawner = (options: {
  readonly discover: ReadonlyArray<Option.Option<string>>
  readonly spawnUrl?: string
}) => {
  let discoverCalls = 0
  let spawnCalls = 0
  const spawner: DaemonSpawner = {
    discover: () => Effect.sync(() => {
      const result = options.discover[Math.min(discoverCalls, options.discover.length - 1)]
      discoverCalls++
      return result
    }),
    spawn: () => Effect.suspend(() => {
      spawnCalls++
      return options.spawnUrl === undefined
        ? Effect.fail(new DaemonSpawnFailed({ reason: "spawn disabled in test" }))
        : Effect.succeed(options.spawnUrl)
    }),
  }
  return { spawner, discoverCalls: () => discoverCalls, spawnCalls: () => spawnCalls }
}

const withClient = <A, E>(
  spawner: DaemonSpawner,
  http: HttpClient.HttpClient,
  use: (client: AcnClient) => Effect.Effect<A, E>,
): Promise<A> =>
  Effect.runPromise(Effect.scoped(Effect.gen(function* () {
    const runtime = yield* makeAcnJitRuntime().pipe(
      Effect.provideService(DaemonSpawnerTag, spawner),
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
  it("performs startup demand once and shares one coordinator across protocol consumers", async () => {
    const { spawner, discoverCalls, spawnCalls } = makeFakeSpawner({
      discover: [Option.some("http://daemon")],
    })
    const { client: http } = makeFakeHttp([
      { kind: "lines", make: (id) => [exitMessage("CheckFileExists", id, Exit.succeed(true))] },
    ])

    const runtime = await Effect.runPromise(
      makeAcnJitRuntime().pipe(Effect.provideService(DaemonSpawnerTag, spawner)),
    )
    const call = Effect.scoped(RpcClient.make(MagnitudeRpcs).pipe(
      Effect.provide(runtime.protocolLayer.pipe(
        Layer.provide(Layer.succeed(HttpClient.HttpClient, http)),
      )),
      Effect.flatMap((client) => client.CheckFileExists({ cwd: "/project", path: "/x" })),
    ))

    expect(await Effect.runPromise(call)).toBe(true)
    expect(await Effect.runPromise(call)).toBe(true)
    expect(discoverCalls()).toBe(1)
    expect(spawnCalls()).toBe(0)
  })

  it("recovers finite work through the same coordinator", async () => {
    const { spawner, spawnCalls } = makeFakeSpawner({
      discover: [Option.some("http://dead"), Option.none()],
      spawnUrl: "http://fresh",
    })
    const { client, calls } = makeFakeHttp([
      { kind: "refuse" },
      { kind: "lines", make: (id) => [exitMessage("CheckFileExists", id, Exit.succeed(true))] },
    ])

    const result = await withClient(spawner, client, (acn) =>
      acn.CheckFileExists({ cwd: "/project", path: "/x" }),
    )
    expect(result).toBe(true)
    expect(calls()).toBe(2)
    expect(spawnCalls()).toBe(1)
  })

  it("automatically exposes payloads while consuming subscription controls", async () => {
    const { spawner } = makeFakeSpawner({ discover: [Option.some("http://daemon")] })
    const { client } = makeFakeHttp([{ kind: "lines", make: (id) => [
      chunkMessage(id, [{ _tag: "keepalive" }]),
      chunkMessage(id, [
        { _tag: "suspended", reason: "session-offloaded" },
        payload("changed", "/real"),
      ]),
      endOfStream(id),
    ] }])

    expect(await withClient(spawner, client, collectPaths)).toEqual(["/real"])
  })

  it("parks on terminated and reconnects to a successor without spawning", async () => {
    const { spawner, spawnCalls } = makeFakeSpawner({
      discover: [Option.some("http://old"), Option.some("http://new")],
      spawnUrl: "http://must-not-spawn",
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

    expect(await withClient(spawner, client, collectPaths)).toEqual(["/before", "/after"])
    expect(spawnCalls()).toBe(0)
  })
})
