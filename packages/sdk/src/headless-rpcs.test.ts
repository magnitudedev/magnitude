import { HeadlessRpcs } from "./headless-rpcs"
import { Schema } from "effect"
import { describe, expect, it } from "vitest"

const metadata = {
  sessionId: "session-1",
  title: "Headless test",
  cwd: "/repo",
  createdAt: 1,
  updatedAt: 1,
  messageCount: 1,
  lastMessage: "test prompt",
}

const successSchema = (tag: "CreateSession" | "GetSession"): Schema.Schema<any, any, never> => {
  const rpc = HeadlessRpcs.requests.get(tag)
  if (rpc === undefined) throw new Error(`Missing ${tag}`)
  return rpc.successSchema as Schema.Schema<any, any, never>
}

describe("HeadlessRpcs strict external decoding", () => {
  it("rejects top-level excess properties before RPC normalization", () => {
    const decoded = Schema.decodeUnknownEither(successSchema("GetSession"))({
      ...metadata,
      forged: true,
    })

    expect(decoded._tag).toBe("Left")
  })

  it("rejects nested excess properties before RPC normalization", () => {
    const decoded = Schema.decodeUnknownEither(successSchema("CreateSession"))({
      _tag: "created",
      metadata: {
        ...metadata,
        forged: true,
      },
    })

    expect(decoded._tag).toBe("Left")
  })

  it("rejects excess properties on daemon error results", () => {
    const rpc = HeadlessRpcs.requests.get("GetSession")
    if (rpc === undefined) throw new Error("Missing GetSession")
    const decoded = Schema.decodeUnknownEither(
      rpc.errorSchema as Schema.Schema<any, any, never>,
    )({
      _tag: "SessionNotFound",
      sessionId: "session-1",
      forged: true,
    })

    expect(decoded._tag).toBe("Left")
  })

  it("accepts the canonical encoded result", () => {
    const decoded = Schema.decodeUnknownEither(successSchema("CreateSession"))({
      _tag: "created",
      metadata,
    })

    expect(decoded._tag).toBe("Right")
  })
})
