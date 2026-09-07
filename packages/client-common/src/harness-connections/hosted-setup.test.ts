import { Schema } from "effect"
import { describe, expect, it } from "vitest"
import { HostedSetupCapability, HostedSetupResult, hostedSetupFailure } from "./hosted-setup"

describe("hosted setup protocol", () => {
  const decode = Schema.decodeUnknownSync(HostedSetupResult)
  it.each([
    { _tag: "Completed", protocolVersion: 1, modelId: "qwen:gguf:q6" },
    { _tag: "Cancelled", protocolVersion: 1 },
    { _tag: "Failed", protocolVersion: 1, message: "failure" },
  ])("round trips $_tag", result => {
    expect(Schema.decodeUnknownSync(Schema.parseJson(HostedSetupResult))(
      Schema.encodeSync(Schema.parseJson(HostedSetupResult))(decode(result)),
    )).toEqual(result)
  })
  it.each([
    {}, { _tag: "Completed", protocolVersion: 1 },
    { _tag: "Cancelled", protocolVersion: 2 },
    { _tag: "Failed", protocolVersion: 1, message: 7 },
    { _tag: "Failed", protocolVersion: 1, message: "x".repeat(4097) },
    { _tag: "Unknown", protocolVersion: 1 },
  ])("rejects invalid result %j", value => {
    expect(() => decode(value)).toThrow()
  })
  it.each(["a", "é", "😀"])("bounds failure diagnostics in UTF-8 (%s)", character => {
    const result = hostedSetupFailure(character.repeat(10000))
    expect(() => decode(result)).not.toThrow()
    if (result._tag === "Failed") expect(new TextEncoder().encode(result.message).length).toBeLessThanOrEqual(4096)
  })
  it("rejects an incompatible capability", () => {
    expect(() => Schema.decodeUnknownSync(HostedSetupCapability)({ protocolVersion: 2 })).toThrow()
  })
})
