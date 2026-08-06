import * as HttpClient from "@effect/platform/HttpClient"
import * as HttpClientResponse from "@effect/platform/HttpClientResponse"
import {
  AcnInstanceIdSchema,
  AcnReady,
  ProcessStartIdentitySchema,
} from "@magnitudedev/acn-protocol"
import { Chunk, Effect, Layer, Option, Stream } from "effect"
import { describe, expect, it } from "vitest"
import type { ReadyAcn } from "./acn-ensurer"
import { AcnEnsuranceFailed } from "./errors"
import { makeRemoteAcnEnsurer } from "./remote-acn-ensurer"

const instance: ReadyAcn = {
  id: AcnInstanceIdSchema.make("acn-1"),
  identity: "1.0.0" as never,
  url: "http://127.0.0.1:5000",
  pid: 42,
  processStartIdentity: ProcessStartIdentitySchema.make("start-42"),
  lifecycle: new AcnReady({}),
}

const makeEnsurer = (respond: (url: string) => Response) => {
  const client = HttpClient.make((request) =>
    Effect.succeed(HttpClientResponse.fromWeb(request, respond(request.url))),
  )
  return Effect.runPromise(makeRemoteAcnEnsurer("http://host").pipe(
    Effect.provide(Layer.succeed(HttpClient.HttpClient, client)),
  ))
}

describe("RemoteAcnEnsurer", () => {
  it("preserves a typed HTTP ensurance error", async () => {
    const failure = new AcnEnsuranceFailed({ reason: "coordinated failure" })
    const ensurer = await makeEnsurer(() => Response.json({ error: failure }, { status: 500 }))
    const error = await Effect.runPromise(Effect.flip(Stream.runDrain(ensurer.ensure({
      minimumIdentity: "1.0.0" as never,
    }))))
    expect(error).toMatchObject({ _tag: "AcnEnsuranceFailed", reason: "coordinated failure" })
  })

  it("preserves streamed failures", async () => {
    const failure = new AcnEnsuranceFailed({ reason: "streamed failure" })
    const ensurer = await makeEnsurer(() => new Response(
      `${JSON.stringify({ _tag: "Failed", error: failure })}\n`,
      { status: 200 },
    ))
    const error = await Effect.runPromise(Effect.flip(Stream.runDrain(ensurer.ensure({
      minimumIdentity: "1.0.0" as never,
    }))))
    expect(error).toMatchObject({ _tag: "AcnEnsuranceFailed", reason: "streamed failure" })
  })

  it("rewrites only the ready endpoint to the exact proxy route", async () => {
    const ensurer = await makeEnsurer(() => new Response(
      `${JSON.stringify({ _tag: "Ready", instance })}\n`,
      { status: 200 },
    ))
    const events = await Effect.runPromise(Stream.runCollect(ensurer.ensure({
      minimumIdentity: "1.0.0" as never,
    })))
    expect(Option.getOrThrow(Chunk.head(events))).toMatchObject({
      _tag: "Ready",
      instance: { id: instance.id, url: "http://host/acn/acn-1" },
    })
  })
})
