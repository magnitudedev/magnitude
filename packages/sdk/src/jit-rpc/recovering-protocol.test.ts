import { describe, expect, it } from "vitest"
import { Rpc, RpcClient, RpcClientError, RpcGroup } from "@effect/rpc"
import * as HttpClient from "@effect/platform/HttpClient"
import * as HttpClientError from "@effect/platform/HttpClientError"
import * as HttpClientRequest from "@effect/platform/HttpClientRequest"
import * as HttpClientResponse from "@effect/platform/HttpClientResponse"
import { Effect, Exit, FiberId, Layer, Option, Schema, Stream } from "effect"
import { recoveringProtocolLayer } from "./recovering-protocol"
import { RecoveryExhausted, SubscriptionProtocolViolation } from "./errors"
import {
  isCleanOrInterruptedExit,
  isInterruptedExit,
  type RecoveringStreamProtocol,
} from "./recovering-stream-protocol"
import {
  AcnOwnerIdSchema,
  type AcnEndpoint,
} from "@magnitudedev/acn-protocol"

const { RpcClientError: TransportError } = RpcClientError

// ─── Fake RPC group ──────────────────────────────────────────────────────────

class FakeError extends Schema.TaggedError<FakeError>()("FakeError", {
  message: Schema.String,
}) {}

const Ping = Rpc.make("Ping", {
  payload: Schema.Struct({ value: Schema.String }),
  success: Schema.String,
  error: FakeError,
})

const Watch = Rpc.make("Watch", {
  payload: Schema.Struct({ path: Schema.String }),
  success: Schema.Struct({ event: Schema.String, path: Schema.String }),
  error: FakeError,
  stream: true,
})

const FakeRpcs = RpcGroup.make(Ping, Watch)

type FakeClient = RpcClient.FromGroup<typeof FakeRpcs, RpcClientError.RpcClientError>

// ─── Wire helpers ────────────────────────────────────────────────────────────

const getRpc = (tag: string) => {
  const rpc = FakeRpcs.requests.get(tag)
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

// ─── Fake access ───────────────────────────────────────────────────────────

const makeFakeEndpointAccess = (options: {
  readonly current: ReadonlyArray<Option.Option<string>>
  readonly startUrl?: string
}) => {
  let currentCalls = 0
  let startCalls = 0
  let selected: AcnEndpoint | null = null
  const fallbackStartUrl = options.current.find(Option.isSome)?.value ?? "http://started"
  const endpoint = (url: string, id: string = url): AcnEndpoint => ({
    id: AcnOwnerIdSchema.make(id),
    version: "1.0.0",
    url,
  })
  const readCurrent = () => Effect.sync(() => {
    const result = options.current[Math.min(currentCalls, options.current.length - 1)]
    currentCalls++
    return Option.map(result, (url) => endpoint(url))
  })
  const start = Effect.sync(() => {
    startCalls++
    const url = options.startUrl ?? fallbackStartUrl
    return endpoint(url, `${url}#start-${startCalls}`)
  })
  const access = {
    endpoint: Effect.suspend(() => {
      if (selected !== null) return Effect.succeed(selected)
      return readCurrent().pipe(
        Effect.flatMap(
          Option.match({
            onSome: Effect.succeed,
            onNone: () => start,
          }),
        ),
        Effect.tap((value) => Effect.sync(() => { selected = value })),
      )
    }),
    recover: (failed: AcnEndpoint) => Effect.suspend(() => {
      if (selected !== null && selected.id !== failed.id) {
        return Effect.succeed(selected)
      }
      selected = null
      return readCurrent().pipe(
        Effect.flatMap(
          Option.match({
            onSome: (current) => current.id !== failed.id
              ? Effect.succeed(current)
              : start,
            onNone: () => start,
          }),
        ),
        Effect.tap((value) => Effect.sync(() => { selected = value })),
      )
    }),
  }
  return { access, currentCalls: () => currentCalls, startCalls: () => startCalls }
}

// ─── Fake resident stream policy ─────────────────────────────────────────────

const FakeStreamWireItem = Schema.Union(
  Schema.Struct({ event: Schema.String, path: Schema.String }),
  Schema.TaggedStruct("keepalive", {}),
  Schema.TaggedStruct("terminated", {}),
)
const decodeFakeStreamWireItem = Schema.decodeUnknown(FakeStreamWireItem)

const fakeStreamProtocol: RecoveringStreamProtocol = {
  isStream: (tag) => tag === "Watch",
  decodeChunk: (values) => Effect.gen(function* () {
    const items = yield* Effect.forEach(values, (value) => decodeFakeStreamWireItem(value))
    const payloads: Array<{ readonly event: string; readonly path: string }> = []
    let terminated = false
    for (const item of items) {
      if (!("_tag" in item)) payloads.push(item)
      else if (item._tag === "terminated") terminated = true
    }
    return terminated
      ? { _tag: "Terminated" }
      : { _tag: "Continue", values: payloads, progressed: payloads.length > 0 }
  }),
  livenessTimeoutMs: 30_000,
  isExitWithoutTermination: isCleanOrInterruptedExit,
}

const classifyInfraError = (
  error: never,
): RpcClientError.RpcClientError =>
  new TransportError({ reason: "Unknown", message: "infra failure", cause: new Error(String(error)) })

// ─── Client under test ───────────────────────────────────────────────────────

const withClient = <A, E>(
  access: {
    readonly endpoint: Effect.Effect<AcnEndpoint>
    readonly recover: (failed: AcnEndpoint) => Effect.Effect<AcnEndpoint>
  },
  http: HttpClient.HttpClient,
  use: (client: FakeClient) => Effect.Effect<A, E>,
): Promise<A> =>
  Effect.runPromise(
    Effect.scoped(
      Effect.gen(function* () {
        const client = yield* RpcClient.make(FakeRpcs).pipe(
          Effect.provide(
            recoveringProtocolLayer<never>({
              endpoint: access.endpoint,
              recover: access.recover,
              rpcPath: "/rpc",
              streamProtocol: fakeStreamProtocol,
              isEndpointRetirementExit: isInterruptedExit,
              classifyInfraError,
            }).pipe(
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

const keepalive = { _tag: "keepalive" }

const endOfStream = (id: string) =>
  exitMessage("Watch", id, Exit.fail(new FakeError({ message: "stream ended" })))

const collectEvents = (client: FakeClient) =>
  Stream.runCollect(
    client.Watch({ path: "/watched" }).pipe(
      Stream.catchAll(() => Stream.empty)
    )
  ).pipe(
    Effect.map((events) => Array.from(events).map((event) => event.path))
  )

// ─── Tests ───────────────────────────────────────────────────────────────────

describe("recovering protocol — operation contract", () => {
  it("dispatches optimistically against a discovered daemon (no spawn)", async () => {
    const { access, startCalls } = makeFakeEndpointAccess({ current: [Option.some("http://daemon-1")] })
    const { client, calls } = makeFakeHttp([
      { kind: "lines", make: (id) => [exitMessage("Ping", id, Exit.succeed("pong"))] },
    ])

    const result = await withClient(access, client, (c) =>
      c.Ping({ value: "ping" })
    )

    expect(result).toBe("pong")
    expect(calls()).toBe(1)
    expect(startCalls()).toBe(0)
  })

  it("recovers a unary op across a daemon death: respawn + re-issue, caller never sees it", async () => {
    const { access, startCalls } = makeFakeEndpointAccess({
      current: [Option.some("http://dead-daemon"), Option.none()],
      startUrl: "http://fresh-daemon",
    })
    const { client, calls } = makeFakeHttp([
      { kind: "refuse" },
      { kind: "lines", make: (id) => [exitMessage("Ping", id, Exit.succeed("pong"))] },
    ])

    const result = await withClient(access, client, (c) =>
      c.Ping({ value: "ping" })
    )

    expect(result).toBe("pong")
    expect(calls()).toBe(2)
    expect(startCalls()).toBe(1)
  })

  it("reissues a unary demand when root retirement wins admission", async () => {
    const { access, startCalls } = makeFakeEndpointAccess({
      current: [Option.some("http://retiring-daemon"), Option.none()],
      startUrl: "http://fresh-daemon",
    })
    const { client, calls } = makeFakeHttp([
      {
        kind: "lines",
        make: (id) => [exitMessage("Ping", id, Exit.interrupt(FiberId.none))],
      },
      { kind: "lines", make: (id) => [exitMessage("Ping", id, Exit.succeed("pong"))] },
    ])

    const result = await withClient(access, client, (c) => c.Ping({ value: "ping" }))

    expect(result).toBe("pong")
    expect(calls()).toBe(2)
    expect(startCalls()).toBe(1)
  })

  it("surfaces a fatal error after two consecutive failures without progress (crash loop)", async () => {
    const { access } = makeFakeEndpointAccess({ current: [Option.some("http://zombie")] })
    const { client, calls } = makeFakeHttp([{ kind: "refuse" }])

    const outcome = await withClient(access, client, (c) =>
      Effect.flip(c.Ping({ value: "ping" }))
    )

    expect(outcome).toBeInstanceOf(TransportError)
    const rpcError = outcome as RpcClientError.RpcClientError
    expect(rpcError.cause).toBeInstanceOf(RecoveryExhausted)
    expect((rpcError.cause as RecoveryExhausted).attempts).toBe(2)
    expect(rpcError.reason).toBe("Unknown")
    expect(calls()).toBe(2)
  })

  it("surfaces a bad HTTP status as transport exhaustion with a typed cause", async () => {
    const { access } = makeFakeEndpointAccess({ current: [Option.some("http://broken-daemon")] })
    const statusClient = HttpClient.make((request) =>
      Effect.succeed(
        HttpClientResponse.fromWeb(
          request,
          new Response("internal error", { status: 500 }),
        ),
      ),
    )

    const outcome = await withClient(access, statusClient, (c) =>
      Effect.flip(c.Ping({ value: "ping" }))
    )

    expect(outcome).toBeInstanceOf(TransportError)
    const rpcError = outcome as RpcClientError.RpcClientError
    expect(rpcError.cause).toBeInstanceOf(RecoveryExhausted)
    expect((rpcError.cause as RecoveryExhausted).attempts).toBe(2)
  })

  it("surfaces resolution failure as fatal with the infra error in cause", async () => {
    const { access } = makeFakeEndpointAccess({ current: [Option.none()] })
    const { client } = makeFakeHttp([{ kind: "refuse" }])

    const outcome = await withClient(access, client, (c) =>
      Effect.flip(c.Ping({ value: "ping" }))
    )

    expect(outcome).toBeInstanceOf(TransportError)
  })

  it("treats a stream body ending without an exit as death and re-issues invisibly", async () => {
    const { access, startCalls } = makeFakeEndpointAccess({
      current: [Option.some("http://daemon-1"), Option.none()],
      startUrl: "http://daemon-2",
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

    const paths = await withClient(access, client, collectEvents)

    expect(paths).toEqual(["first", "second"])
    expect(calls()).toBe(2)
    expect(startCalls()).toBe(1)
  })

  it("surfaces a stream exit without the authoritative terminal control", async () => {
    const { access, startCalls } = makeFakeEndpointAccess({
      current: [Option.some("http://draining-daemon")],
      startUrl: "http://fresh-daemon",
    })
    const { client, calls } = makeFakeHttp([
      {
        kind: "lines",
        make: (id) => [
          chunkMessage(id, [{ event: "changed", path: "before-shutdown" }]),
          exitMessage("Watch", id, Exit.void),
        ],
      },
    ])

    const outcome = await withClient(access, client, (c) =>
      Effect.flip(Stream.runCollect(c.Watch({ path: "/watched" }))),
    )

    expect(outcome).toBeInstanceOf(TransportError)
    expect((outcome as RpcClientError.RpcClientError).cause).toBeInstanceOf(
      SubscriptionProtocolViolation,
    )
    expect(calls()).toBe(1)
    expect(startCalls()).toBe(0)
  })

  it("does not treat an unframed server interrupt as daemon retirement", async () => {
    const { access, startCalls } = makeFakeEndpointAccess({
      current: [Option.some("http://draining-daemon")],
      startUrl: "http://fresh-daemon",
    })
    const { client, calls } = makeFakeHttp([
      {
        kind: "lines",
        make: (id) => [
          chunkMessage(id, [{ event: "changed", path: "before-interrupt" }]),
          exitMessage("Watch", id, Exit.interrupt(FiberId.none)),
        ],
      },
    ])

    const outcome = await withClient(access, client, (c) =>
      Effect.flip(Stream.runCollect(c.Watch({ path: "/watched" }))),
    )

    expect(outcome).toBeInstanceOf(TransportError)
    expect((outcome as RpcClientError.RpcClientError).cause).toBeInstanceOf(
      SubscriptionProtocolViolation,
    )
    expect(calls()).toBe(1)
    expect(startCalls()).toBe(0)
  })

  it("surfaces a domain error on a stream to the consumer without re-issuing", async () => {
    const { access, startCalls } = makeFakeEndpointAccess({ current: [Option.some("http://daemon-1")] })
    const { client, calls } = makeFakeHttp([
      { kind: "lines", make: (id) => [endOfStream(id)] },
    ])

    const outcome = await withClient(access, client, (c) =>
      Effect.flip(Stream.runCollect(c.Watch({ path: "/watched" })))
    )

    expect(outcome).toBeInstanceOf(FakeError)
    expect(calls()).toBe(1)
    expect(startCalls()).toBe(0)
  })

  it("consumes keepalives so they never reach the consumer", async () => {
    const { access } = makeFakeEndpointAccess({ current: [Option.some("http://daemon-1")] })
    const { client } = makeFakeHttp([
      {
        kind: "lines",
        make: (id) => [
          chunkMessage(id, [keepalive]),
          chunkMessage(id, [keepalive, { event: "created", path: "real" }]),
          endOfStream(id),
        ],
      },
    ])

    const paths = await withClient(access, client, collectEvents)
    expect(paths).toEqual(["real"])
  })

  it("parks after authoritative termination and resumes on a discovered successor", async () => {
    const { access, startCalls } = makeFakeEndpointAccess({
      current: [
        Option.some("http://daemon-1"),
        Option.some("http://daemon-2"),
      ],
      startUrl: "http://must-not-spawn",
    })
    const { client, calls } = makeFakeHttp([
      {
        kind: "lines",
        make: (id) => [
          chunkMessage(id, [{ event: "changed", path: "before" }]),
          chunkMessage(id, [{ _tag: "terminated" }]),
        ],
      },
      {
        kind: "lines",
        make: (id) => [
          chunkMessage(id, [{ event: "changed", path: "after" }]),
          endOfStream(id),
        ],
      },
    ])

    const paths = await withClient(access, client, collectEvents)
    expect(paths).toEqual(["before", "after"])
    expect(calls()).toBe(2)
    expect(startCalls()).toBe(0)
  })

  it("a stream that made progress recovers again on a later, separate death", async () => {
    const { access, startCalls } = makeFakeEndpointAccess({
      current: [
        Option.some("http://daemon-1"),
        Option.none(),
        Option.none(),
      ],
      startUrl: "http://daemon-n",
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

    const paths = await withClient(access, client, collectEvents)

    expect(paths).toEqual(["one", "two", "three"])
    expect(calls()).toBe(3)
    expect(startCalls()).toBe(2)
  })
})
