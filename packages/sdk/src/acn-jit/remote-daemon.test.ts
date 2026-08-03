import { FetchHttpClient } from "@effect/platform"
import { Effect, Option, Stream } from "effect"
import { afterEach, describe, expect, it } from "vitest"
import { runDaemonLaunch } from "./daemon-launcher"
import { makeRemoteDaemonDiscovery, makeRemoteDaemonLauncher } from "./remote-daemon"

let server: ReturnType<typeof Bun.serve> | undefined

afterEach(() => {
  server?.stop(true)
  server = undefined
})

describe("remote daemon discovery", () => {
  it("preserves query failure instead of reporting authoritative absence", async () => {
    server = Bun.serve({
      port: 0,
      fetch: () => Response.json(
        { error: "registry unavailable" },
        { status: 500 },
      ),
    })

    const outcome = await makeRemoteDaemonDiscovery(
      `http://127.0.0.1:${server.port}`,
    ).pipe(
      Effect.flatMap((discovery) => discovery.current()),
      Effect.either,
      Effect.provide(FetchHttpClient.layer),
      Effect.runPromise,
    )

    expect(outcome).toMatchObject({
      _tag: "Left",
      left: {
        _tag: "DaemonDiscoveryFailed",
        reason: "registry unavailable",
      },
    })
  })
})

describe("remote daemon launcher", () => {
  it("forwards startup observations before returning the ACN", async () => {
    const encoder = new TextEncoder()
    server = Bun.serve({
      port: 0,
      fetch: () => {
        const stream = new ReadableStream<Uint8Array>({
          start(controller) {
            controller.enqueue(
              encoder.encode(
                `${JSON.stringify({
                  _tag: "Observation",
                  observation: {
                    _tag: "Installing",
                    phase: "DownloadingDaemon",
                    plan: {
                      daemonBytes: 100,
                      inferenceEngineBytes: 300,
                      inferenceEngineBytesExact: false,
                    },
                    progress: {
                      completed: 50,
                      totalBytes: 100,
                      unit: "Bytes",
                      attempt: 1,
                    },
                  },
                })}\n`,
              ),
            )
            controller.enqueue(
              encoder.encode(
                `${JSON.stringify({
                  _tag: "Ready",
                  endpoint: {
                    id: "acn-test",
                    version: "1.0.0",
                    url: "http://127.0.0.1:4567",
                  },
                })}\n`,
              ),
            )
            controller.close()
          },
        })
        return new Response(stream, {
          headers: { "content-type": "application/x-ndjson" },
        })
      },
    })
    const observations: string[] = []

    const proxyUrl = `http://127.0.0.1:${server.port}`
    const endpoint = await makeRemoteDaemonLauncher(proxyUrl).pipe(
      Effect.flatMap((launcher) =>
        runDaemonLaunch(
          launcher.launch(Option.none()).pipe(
            Stream.tap((event) =>
              event._tag === "Observation"
                ? Effect.sync(() => {
                    observations.push(event.observation._tag)
                  })
                : Effect.void
            ),
          ),
        ),
      ),
      Effect.provide(FetchHttpClient.layer),
      Effect.runPromise,
    )

    expect(observations).toEqual(["Installing"])
    expect(endpoint).toEqual({
      id: "acn-test",
      version: "1.0.0",
      url: proxyUrl,
    })
  })
})
