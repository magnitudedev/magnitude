import * as HttpClient from "@effect/platform/HttpClient"
import * as HttpClientResponse from "@effect/platform/HttpClientResponse"
import { AcnReady } from "@magnitudedev/acn-protocol"
import { Effect, Layer, Option, Stream } from "effect"
import { describe, expect, it } from "vitest"
import { makeRemoteAcnProcessManager } from "./remote-acn-process-manager"

const failure = { _tag: "DaemonSpawnFailed", reason: "coordinated failure" }

const makeManager = (respond: (url: string) => Response) => {
  const client = HttpClient.make((request) =>
    Effect.succeed(HttpClientResponse.fromWeb(request, respond(request.url))),
  )
  return Effect.runPromise(
    makeRemoteAcnProcessManager("http://host").pipe(
      Effect.provide(Layer.succeed(HttpClient.HttpClient, client)),
    ),
  )
}

const instance = {
  id: "acn-1" as never,
  identity: "1.0.0" as never,
  url: "http://host/acn/acn-1",
  pid: 42,
  processStartIdentity: "start-42" as never,
  lifecycle: new AcnReady({}),
}

describe("RemoteAcnProcessManager", () => {
  it("preserves typed current-observation errors", async () => {
    const manager = await makeManager(() => Response.json({ error: failure }, { status: 500 }))
    const error = await Effect.runPromise(Effect.flip(manager.observeCurrent))
    expect(error).toMatchObject(failure)
  })

  it("preserves typed launch-stream errors", async () => {
    const manager = await makeManager(() => new Response(
      `${JSON.stringify({ _tag: "Failed", error: failure })}\n`,
      { status: 200 },
    ))
    const error = await Effect.runPromise(Effect.flip(Stream.runDrain(manager.launch({
      identity: "1.0.0" as never,
      replace: Option.none(),
      command: Option.none(),
    }))))
    expect(error).toMatchObject(failure)
  })

  it("preserves typed exact-termination errors", async () => {
    const manager = await makeManager(() => Response.json({ error: failure }, { status: 500 }))
    const error = await Effect.runPromise(Effect.flip(manager.terminate(instance)))
    expect(error).toMatchObject(failure)
  })
})
