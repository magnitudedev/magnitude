import { describe, expect, it } from "vitest"
import { Schema } from "effect"
import {
  BackendEligibilityReport,
  IcnInstallationDeclaration,
  IcnStartupRecord,
} from "./generated/schemas.js"

describe("generated ICN bootstrap protocol", () => {
  it("decodes the Rust eligibility discriminator and rejects the former duplicate", () => {
    const decode = Schema.decodeUnknownSync(
      Schema.parseJson(BackendEligibilityReport),
    )
    const report = decode(JSON.stringify({
      schemaVersion: 1,
      cuda: {
        state: "usable",
        driverApi: 12_000,
        architectures: ["90"],
        driverLibrary: "/host/libcuda.so.1",
      },
      vulkan: { state: "absent", diagnostic: "unavailable" },
      metal: { state: "absent", diagnostic: "unavailable" },
    }))
    expect(report.cuda.state).toBe("usable")
    expect(() => decode(JSON.stringify({
      schemaVersion: 1,
      cuda: {
        _tag: "usable",
        driverApi: 12_000,
        architectures: ["90"],
        driverLibrary: "/host/libcuda.so.1",
      },
      vulkan: { _tag: "absent", diagnostic: "unavailable" },
      metal: { _tag: "absent", diagnostic: "unavailable" },
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
      backend: "cpu",
      nativeBuild: "native",
      backendModuleAbi: "abi",
    })
    expect(JSON.parse(installation)).toEqual({
      schemaVersion: 1,
      backend: "cpu",
      nativeBuild: "native",
      backendModuleAbi: "abi",
    })
  })
})
