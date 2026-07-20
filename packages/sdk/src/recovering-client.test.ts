import { describe, expect, it } from "vitest"
import { Effect, Exit, FiberId, Layer, Option, Schema, Stream } from "effect"
import { Rpc, RpcClient, RpcClientError } from "@effect/rpc"
import * as HttpClient from "@effect/platform/HttpClient"
import * as HttpClientError from "@effect/platform/HttpClientError"
import * as HttpClientRequest from "@effect/platform/HttpClientRequest"
import * as HttpClientResponse from "@effect/platform/HttpClientResponse"
import { MagnitudeRpcs, SessionNotFound } from "@magnitudedev/protocol"
import { DaemonSpawnerTag, type DaemonSpawner } from "./daemon-spawner"
import { makeRecoveringProtocolLayer } from "./recovering-client"
import { DaemonSpawnFailed } from "./errors"
import type { AcnClient } from "./protocol"

const { RpcClientError: TransportError } = RpcClientError

// ─── Wire helpers ────────────────────────────────────────────────────────────

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
  const parsed: unknown = JSON.parse(requestText(request).split("\n")[0])
  if (typeof parsed === "object" && parsed !== null && "id" in parsed && typeof parsed.id === "string") {
    return parsed.id
  }
  throw new Error("request had no id")
}

// ─── Fake daemon (HTTP layer) ────────────────────────────────────────────────

/** Script for one POST attempt: refuse the connection, or serve ndjson lines then EOF. */
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
      const text = attempt.make(requestId).map((line) => JSON.stringify(line)).join("\n") + "\n"
      return Effect.succeed(HttpClientResponse.fromWeb(request, new Response(text, { status: 200 })))
    })
  )
  return { client, calls: () => calls }
}

// ─── Fake spawner ────────────────────────────────────────────────────────────

const makeFakeSpawner = (options: {
  readonly discover: ReadonlyArray<Option.Option<string>>
  readonly spawnUrl?: string
}) => {
  let discoverCalls = 0
  let spawnCalls = 0
  const spawner: DaemonSpawner = {
    discover: () =>
      Effect.sync(() => {
        const result = options.discover[Math.min(discoverCalls, options.discover.length - 1)]
        discoverCalls++
        return result
      }),
    spawn: () =>
      Effect.suspend(() => {
        spawnCalls++
        return options.spawnUrl !== undefined
          ? Effect.succeed(options.spawnUrl)
          : Effect.fail(new DaemonSpawnFailed({ reason: "spawn disabled in test" }))
      }),
  }
  return { spawner, discoverCalls: () => discoverCalls, spawnCalls: () => spawnCalls }
}

// ─── Client under test ───────────────────────────────────────────────────────

const withClient = <A, E>(
  spawner: DaemonSpawner,
  http: HttpClient.HttpClient,
  use: (client: AcnClient) => Effect.Effect<A, E>
): Promise<A> =>
  Effect.runPromise(
    Effect.scoped(
      Effect.gen(function* () {
        const protocolLayer = yield* makeRecoveringProtocolLayer().pipe(
          Effect.provideService(DaemonSpawnerTag, spawner),
        )
        const client = yield* RpcClient.make(MagnitudeRpcs).pipe(
          Effect.provide(
            protocolLayer.pipe(
              Layer.provide(Layer.succeed(HttpClient.HttpClient, http)),
            )
          )
        )
        return yield* use(client)
      })
    )
  )

// ─── Wire message builders ───────────────────────────────────────────────────

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

const heartbeat = { _tag: "heartbeat" }

/**
 * Resident streams never end legitimately from the server, so tests
 * terminate them with a domain error and collect what arrived before it.
 */
const endOfStream = (id: string) =>
  exitMessage("WatchFile", id, Exit.fail(new SessionNotFound({ sessionId: "s" })))

const collectPaths = (acn: AcnClient) =>
  Stream.runCollect(
    acn.WatchFile({ cwd: "/project", path: "/watched" }).pipe(
      Stream.catchAll(() => Stream.empty)
    )
  ).pipe(
    Effect.map((events) => Array.from(events).flatMap((event) => "_tag" in event ? [] : [event.path]))
  )

// ─── Tests ───────────────────────────────────────────────────────────────────

describe("recovering client — operation contract", () => {
  it("shares one lazy resolver across independent builds of one protocol layer", async () => {
    const { spawner, discoverCalls, spawnCalls } = makeFakeSpawner({
      discover: [Option.some("http://daemon-1")],
    })
    const { client: http } = makeFakeHttp([
      { kind: "lines", make: (id) => [exitMessage("CheckFileExists", id, Exit.succeed(true))] },
    ])
    const protocolLayer = (await Effect.runPromise(
      makeRecoveringProtocolLayer().pipe(Effect.provideService(DaemonSpawnerTag, spawner)),
    )).pipe(
      Layer.provide(Layer.succeed(HttpClient.HttpClient, http)),
    )
    const call = Effect.scoped(
      RpcClient.make(MagnitudeRpcs).pipe(
        Effect.provide(protocolLayer),
        Effect.flatMap((acn) => acn.CheckFileExists({ cwd: "/project", path: "/x" })),
      ),
    )

    expect(await Effect.runPromise(call)).toBe(true)
    expect(await Effect.runPromise(call)).toBe(true)
    expect(discoverCalls()).toBe(1)
    expect(spawnCalls()).toBe(0)
  })

  it("dispatches optimistically against a discovered daemon (no spawn)", async () => {
    const { spawner, spawnCalls } = makeFakeSpawner({ discover: [Option.some("http://daemon-1")] })
    const { client, calls } = makeFakeHttp([
      { kind: "lines", make: (id) => [exitMessage("CheckFileExists", id, Exit.succeed(true))] },
    ])

    const result = await withClient(spawner, client, (acn) =>
      acn.CheckFileExists({ cwd: "/project", path: "/x" })
    )

    expect(result).toBe(true)
    expect(calls()).toBe(1)
    expect(spawnCalls()).toBe(0)
  })

  it("recovers a unary op across a daemon death: respawn + re-issue, caller never sees it", async () => {
    // The daemon was killed: first POST is refused. The engine must
    // invalidate, discover nothing, spawn a replacement, and re-issue the
    // SAME operation — the caller's promise just resolves.
    const { spawner, spawnCalls } = makeFakeSpawner({
      discover: [Option.some("http://dead-daemon"), Option.none()],
      spawnUrl: "http://fresh-daemon",
    })
    const { client, calls } = makeFakeHttp([
      { kind: "refuse" },
      { kind: "lines", make: (id) => [exitMessage("CheckFileExists", id, Exit.succeed(true))] },
    ])

    const result = await withClient(spawner, client, (acn) =>
      acn.CheckFileExists({ cwd: "/project", path: "/x" })
    )

    expect(result).toBe(true)
    expect(calls()).toBe(2)
    expect(spawnCalls()).toBe(1)
  })

  it("surfaces a fatal error after two consecutive failures without progress (crash loop)", async () => {
    const { spawner } = makeFakeSpawner({
      discover: [Option.some("http://zombie")],
    })
    const { client, calls } = makeFakeHttp([{ kind: "refuse" }])

    const outcome = await withClient(spawner, client, (acn) =>
      Effect.flip(acn.CheckFileExists({ cwd: "/project", path: "/x" }))
    )

    expect(outcome).toBeInstanceOf(TransportError)
    expect(calls()).toBe(2)
  })

  it("surfaces resolution failure as fatal with the DaemonError in cause", async () => {
    const { spawner } = makeFakeSpawner({ discover: [Option.none()] })
    const { client } = makeFakeHttp([{ kind: "refuse" }])

    const outcome = await withClient(spawner, client, (acn) =>
      Effect.flip(acn.CheckFileExists({ cwd: "/project", path: "/x" }))
    )

    expect(outcome).toBeInstanceOf(TransportError)
    expect(outcome.cause).toBeInstanceOf(DaemonSpawnFailed)
  })

  it("treats a stream body ending without an exit as death and re-issues invisibly", async () => {
    // First attempt streams one event then dies (EOF, no exit). The engine
    // re-issues the same wire request; the consumer sees one uninterrupted
    // stream with events from both attempts.
    const { spawner, spawnCalls } = makeFakeSpawner({
      discover: [Option.some("http://daemon-1"), Option.none()],
      spawnUrl: "http://daemon-2",
    })
    const { client, calls } = makeFakeHttp([
      { kind: "lines", make: (id) => [chunkMessage(id, [{ event: "changed", path: "first" }])] },
      {
        kind: "lines",
        make: (id) => [
          chunkMessage(id, [{ event: "changed", path: "second" }]),
          endOfStream(id),
        ],
      },
    ])

    const paths = await withClient(spawner, client, collectPaths)

    expect(paths).toEqual(["first", "second"])
    expect(calls()).toBe(2)
    expect(spawnCalls()).toBe(1)
  })

  it("treats a clean stream exit (graceful shutdown) as relinquishment and re-issues invisibly", async () => {
    // SIGTERM / idle timeout: the daemon drains and ends the display stream
    // with a clean protocol exit. A resident stream never legitimately ends,
    // so the engine must recover exactly as if the transport had died —
    // never complete the consumer's stream silently.
    const { spawner, spawnCalls } = makeFakeSpawner({
      discover: [Option.some("http://draining-daemon"), Option.none()],
      spawnUrl: "http://fresh-daemon",
    })
    const { client, calls } = makeFakeHttp([
      {
        kind: "lines",
        make: (id) => [
          chunkMessage(id, [{ event: "changed", path: "before-shutdown" }]),
          exitMessage("WatchFile", id, Exit.void),
        ],
      },
      {
        kind: "lines",
        make: (id) => [
          chunkMessage(id, [{ event: "changed", path: "after-respawn" }]),
          endOfStream(id),
        ],
      },
    ])

    const paths = await withClient(spawner, client, collectPaths)

    expect(paths).toEqual(["before-shutdown", "after-respawn"])
    expect(calls()).toBe(2)
    expect(spawnCalls()).toBe(1)
  })

  it("treats a server-side interrupt exit as relinquishment and re-issues invisibly", async () => {
    const { spawner, spawnCalls } = makeFakeSpawner({
      discover: [Option.some("http://draining-daemon"), Option.none()],
      spawnUrl: "http://fresh-daemon",
    })
    const { client, calls } = makeFakeHttp([
      {
        kind: "lines",
        make: (id) => [
          chunkMessage(id, [{ event: "changed", path: "before-interrupt" }]),
          exitMessage("WatchFile", id, Exit.interrupt(FiberId.none)),
        ],
      },
      {
        kind: "lines",
        make: (id) => [
          chunkMessage(id, [{ event: "changed", path: "after-respawn" }]),
          endOfStream(id),
        ],
      },
    ])

    const paths = await withClient(spawner, client, collectPaths)

    expect(paths).toEqual(["before-interrupt", "after-respawn"])
    expect(calls()).toBe(2)
    expect(spawnCalls()).toBe(1)
  })

  it("surfaces a domain error on a stream to the consumer without re-issuing", async () => {
    const { spawner, spawnCalls } = makeFakeSpawner({ discover: [Option.some("http://daemon-1")] })
    const { client, calls } = makeFakeHttp([
      { kind: "lines", make: (id) => [endOfStream(id)] },
    ])

    const outcome = await withClient(spawner, client, (acn) =>
      Effect.flip(Stream.runCollect(acn.WatchFile({ cwd: "/project", path: "/watched" })))
    )

    expect(outcome).toBeInstanceOf(SessionNotFound)
    expect(calls()).toBe(1)
    expect(spawnCalls()).toBe(0)
  })

  it("filters heartbeats so they never reach the consumer", async () => {
    const { spawner } = makeFakeSpawner({ discover: [Option.some("http://daemon-1")] })
    const { client } = makeFakeHttp([
      {
        kind: "lines",
        make: (id) => [
          chunkMessage(id, [heartbeat]),
          chunkMessage(id, [heartbeat, { event: "created", path: "real" }]),
          endOfStream(id),
        ],
      },
    ])

    const paths = await withClient(spawner, client, collectPaths)
    expect(paths).toEqual(["real"])
  })

  it("a stream that made progress recovers again on a later, separate death", async () => {
    // Death → recover → progress → death → recover. Only two consecutive
    // no-progress failures are fatal; separate deaths are routine.
    const { spawner, spawnCalls } = makeFakeSpawner({
      discover: [
        Option.some("http://daemon-1"),
        Option.none(),
        Option.none(),
      ],
      spawnUrl: "http://daemon-n",
    })
    const { client, calls } = makeFakeHttp([
      { kind: "lines", make: (id) => [chunkMessage(id, [{ event: "changed", path: "one" }])] },
      { kind: "lines", make: (id) => [chunkMessage(id, [{ event: "changed", path: "two" }])] },
      {
        kind: "lines",
        make: (id) => [
          chunkMessage(id, [{ event: "changed", path: "three" }]),
          endOfStream(id),
        ],
      },
    ])

    const paths = await withClient(spawner, client, collectPaths)

    expect(paths).toEqual(["one", "two", "three"])
    expect(calls()).toBe(3)
    expect(spawnCalls()).toBe(2)
  })
})
