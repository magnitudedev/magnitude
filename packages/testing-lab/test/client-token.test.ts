import { HttpClient, HttpClientResponse } from "@effect/platform"
import { Effect, Layer, Redacted, TestClock, TestContext } from "effect"
import { expect, test } from "vitest"
import { githubClientToken } from "../src/client-token"
import { LabClient, labClientLayer } from "../src/client"

const token = (exp: number, generation: number) => `${Buffer.from('{}').toString('base64url')}.${Buffer.from(JSON.stringify({ exp, generation })).toString('base64url')}.signature`
test("renews cached GitHub credentials before subsequent coordinator requests", async () => {
  let issued = 0
  const received: string[] = []
  const transport = Layer.succeed(HttpClient.HttpClient, HttpClient.make(request => Effect.sync(() => {
    if (request.url.startsWith("https://pipelines.actions.githubusercontent.com/")) {
      expect(new URL(request.url).searchParams.get("audience")).toBe("lab audience")
      expect(request.headers.authorization).toBe("Bearer request-secret")
      issued++
      return HttpClientResponse.fromWeb(request, Response.json({ value: token(3600, issued) }))
    }
    received.push(request.headers.authorization!)
    return HttpClientResponse.fromWeb(request, Response.json({ owner: "github:123:789:1", trust: "untrusted-ci" }))
  })))
  await Effect.runPromise(Effect.gen(function* () {
    const credential = yield* githubClientToken("https://pipelines.actions.githubusercontent.com/token?api-version=2", Redacted.make("request-secret"), "lab audience")
    yield* Effect.gen(function* () {
      const client = yield* LabClient
      yield* client.identity()
      yield* client.identity()
      expect(issued).toBe(1)
      yield* TestClock.adjust("61 seconds")
      yield* client.identity()
      expect(issued).toBe(2)
      expect(received[0]).toBe(received[1])
      expect(received[2]).not.toBe(received[0])
    }).pipe(Effect.provide(labClientLayer("https://lab.example", credential)))
  }).pipe(Effect.provide(Layer.merge(transport, TestContext.TestContext))))
})
test("rejects token endpoints outside the Actions HTTPS host before sending credentials", async () => {
  for (const url of ["http://pipelines.actions.githubusercontent.com/token", "https://actions.githubusercontent.com.attacker.example/token", "https://user@pipelines.actions.githubusercontent.com/token"]) {
    const result = await Effect.runPromise(githubClientToken(url, Redacted.make("secret"), "lab").pipe(
      Effect.provide(Layer.succeed(HttpClient.HttpClient, HttpClient.make(() => Effect.die("Must not send credentials")))), Effect.either))
    expect(result._tag).toBe("Left")
  }
})
test("rejects malformed and nearly expired tokens without publishing a credential", async () => {
  for (const value of ["invalid", token(30, 1)]) {
    const transport = Layer.succeed(HttpClient.HttpClient, HttpClient.make(request => Effect.succeed(
      HttpClientResponse.fromWeb(request, Response.json({ value })),
    )))
    const result = await Effect.runPromise(githubClientToken("https://pipelines.actions.githubusercontent.com/token", Redacted.make("secret"), "lab").pipe(
      Effect.flatten, Effect.provide(Layer.merge(transport, TestContext.TestContext)), Effect.either,
    ))
    expect(result._tag).toBe("Left")
  }
})
