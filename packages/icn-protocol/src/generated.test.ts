import { describe, expect, it } from "vitest"
import { Schema } from "effect"
import {
  IcnInstallationDeclaration,
  IcnStartupRecord,
  ModelLoadPlan,
} from "./generated/schemas.js"

describe("generated ICN bootstrap protocol", () => {
  it("decodes a load plan only with its execution device", () => {
    const decode = Schema.decodeUnknownSync(Schema.parseJson(ModelLoadPlan))
    const plan = decode(JSON.stringify({
      contextWindowTokens: 262_144,
      requiredMemoryBytes: 3_204_000_000,
      device: { id: "metal:0000000100000abc", backend: "metal" },
    }))
    expect(plan.device.backend).toBe("metal")
    expect(() => decode(JSON.stringify({
      contextWindowTokens: 262_144,
      requiredMemoryBytes: 3_204_000_000,
    }))).toThrow()
  })

  it("preserves the native readiness and installation field names", () => {
    const startup = Schema.decodeUnknownSync(
      Schema.parseJson(IcnStartupRecord),
    )(JSON.stringify({
      type: "icn_ready",
      protocolVersion: 1,
      origin: "http://127.0.0.1:3000",
      instanceId: "instance",
      pid: 1,
      apiVersion: 1,
      nativeBuild: "native",
    }))
    expect(startup.type).toBe("icn_ready")

    const installation = Schema.encodeSync(
      Schema.parseJson(IcnInstallationDeclaration),
    )({
      schemaVersion: 1,
      nativeBuild: "native",
    })
    expect(JSON.parse(installation)).toEqual({
      schemaVersion: 1,
      nativeBuild: "native",
    })
  })
})
