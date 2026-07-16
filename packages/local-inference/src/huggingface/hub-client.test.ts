import * as FetchHttpClient from "@effect/platform/FetchHttpClient"
import { afterAll, beforeAll, describe, expect, it } from "vitest"
import { Effect, Option, Redacted } from "effect"
import { HuggingFaceRepositoryId, HuggingFaceRevision } from "./identity"
import { makeHuggingFaceHubClient } from "./hub-client"

let server: ReturnType<typeof Bun.serve>
const authorization: string[] = []

beforeAll(() => {
  server = Bun.serve({
    port: 0,
    fetch: (request) => {
      const url = new URL(request.url)
      authorization.push(request.headers.get("authorization") ?? "")
      if (url.pathname.endsWith("/tree/main") && url.searchParams.get("page") === "2") {
        return Response.json([{ path: "weights-2.gguf", type: "file", lfs: { oid: `sha256:${"b".repeat(64)}`, size: 20 } }])
      }
      if (url.pathname.endsWith("/tree/main")) {
        return Response.json(
          [{ path: "weights-1.gguf", type: "file", lfs: { oid: `sha256:${"a".repeat(64)}`, size: 10 } }],
          { headers: { link: `<http://127.0.0.1:${server.port}${url.pathname}?page=2>; rel="next"` } },
        )
      }
      if (url.pathname.endsWith("/tree/cycle")) {
        return Response.json([], { headers: { link: `<http://127.0.0.1:${server.port}${url.pathname}>; rel="next"` } })
      }
      if (url.pathname.endsWith("/tree/invalid")) return Response.json([{ path: 42, type: "file" }])
      if (url.pathname.includes("/revision/rejected")) return Response.json({}, { status: 403 })
      if (url.pathname.includes("/revision/")) return Response.json({ sha: "c".repeat(40) })
      return new Response(null, { status: 404 })
    },
  })
})

afterAll(() => server.stop(true))

const client = () => makeHuggingFaceHubClient({
  apiOrigin: Option.some(new URL(`http://127.0.0.1:${server.port}`)),
  token: Option.some(Redacted.make("hub-secret")),
}).pipe(Effect.provide(FetchHttpClient.layer))

const repository = HuggingFaceRepositoryId.make("owner/model")

describe("Hugging Face Hub client", () => {
  it("follows Link pagination, preserves LFS facts, and authenticates every request", async () => {
    authorization.length = 0
    const files = await Effect.runPromise(client().pipe(
      Effect.flatMap((hub) => hub.listFiles(repository, HuggingFaceRevision.make("main"))),
    ))
    expect(files.map(({ path }) => path)).toEqual(["weights-1.gguf", "weights-2.gguf"])
    expect(Option.getOrNull(files[0]?.lfs ?? Option.none())?.sizeBytes).toBe(10)
    expect(authorization).toEqual(["Bearer hub-secret", "Bearer hub-secret"])
  })

  it("rejects pagination cycles instead of looping", async () => {
    const result = await Effect.runPromiseExit(client().pipe(
      Effect.flatMap((hub) => hub.listFiles(repository, HuggingFaceRevision.make("cycle"))),
    ))
    expect(result._tag).toBe("Failure")
  })

  it("distinguishes invalid response data from rejected status", async () => {
    const invalid = await Effect.runPromiseExit(client().pipe(
      Effect.flatMap((hub) => hub.listFiles(repository, HuggingFaceRevision.make("invalid"))),
    ))
    const rejected = await Effect.runPromiseExit(client().pipe(
      Effect.flatMap((hub) => hub.resolveRevision(repository, HuggingFaceRevision.make("rejected"))),
    ))
    expect(invalid._tag).toBe("Failure")
    expect(rejected._tag).toBe("Failure")
  })
})
