import { describe, expect, it } from "vitest"
import { Option, Schema } from "effect"
import { ProviderModelIdSchema } from "@magnitudedev/sdk"
import { HarnessIdSchema } from "../harness-connections/service"
import { DesktopConnectRequest } from "./connections"

describe("desktop connection boundary", () => {
  it("omits an ordinary connection's model across the structured-clone boundary", () => {
    const input = { harness: HarnessIdSchema.make("opencode"), model: Option.none() }
    const encoded = structuredClone(Schema.encodeSync(DesktopConnectRequest)(input))
    expect(encoded).toEqual({ harness: "opencode" })
    expect(Schema.decodeUnknownSync(DesktopConnectRequest)(encoded)).toEqual(input)
  })

  it("preserves the exact model selection across the structured-clone boundary", () => {
    const input = { harness: HarnessIdSchema.make("opencode"), model: Option.some(ProviderModelIdSchema.make("catalog/model-Q8")) }
    const encoded = structuredClone(Schema.encodeSync(DesktopConnectRequest)(input))
    expect(encoded).toEqual({ harness: "opencode", model: "catalog/model-Q8" })
    expect(Schema.decodeUnknownSync(DesktopConnectRequest)(encoded)).toEqual(input)
  })
})
