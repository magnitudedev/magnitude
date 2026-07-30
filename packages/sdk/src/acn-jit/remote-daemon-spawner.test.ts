import { FetchHttpClient } from "@effect/platform"
import { Effect, Option, Stream } from "effect"
import { afterEach, describe, expect, it } from "vitest"
import { runDaemonSpawn } from "./daemon-spawner"
import { makeRemoteDaemonSpawner } from "./remote-daemon-spawner"

let server: ReturnType<typeof Bun.serve> | undefined

afterEach(() => {
  server?.stop(true)
  server = undefined
})

describe("remote daemon spawner", () => {
  it("forwards streamed startup observations before returning the daemon", async () => {
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
                  url: "http://127.0.0.1:4567",
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

    const url = await makeRemoteDaemonSpawner(
      `http://127.0.0.1:${server.port}`,
    ).pipe(
      Effect.flatMap((spawner) =>
        runDaemonSpawn(
          spawner.spawn(Option.none()).pipe(
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
    expect(url).toBe("http://127.0.0.1:4567")
  })
})
