import { describe, expect, it } from "vitest"
import { BackendEligibilityReport } from "@magnitudedev/icn-protocol"
import { Schema } from "effect"
import { finalOutputRecord } from "./release-installation.js"

describe("release ICN command output", () => {
  it("decodes the final backend eligibility record after native driver chatter", () => {
    const output = [
      " dllPath = /usr/lib/wsl/drivers/iigd_dch.inf_amd64/libigdgmm_w.so.12",
      " IsWddmLinux = 1, dllWslName = /usr/lib/wsl/drivers/iigd_dch.inf_amd64/libigdgmm_w.so.12 flags = 2",
      '{"schemaVersion":1,"cuda":{"state":"usable","driverApi":13010,"architectures":["89"],"driverLibrary":"/usr/lib/wsl/lib/libcuda.so.1"},"vulkan":{"state":"usable","loaderApi":4206867},"metal":{"state":"absent","diagnostic":"Metal requires Apple Silicon"}}',
      "",
    ].join("\n")

    const report = Schema.decodeUnknownSync(
      Schema.parseJson(BackendEligibilityReport),
    )(finalOutputRecord(output))

    expect(report.schemaVersion).toBe(1)
    expect(report.cuda).toMatchObject({
      state: "usable",
      driverApi: 13010,
      architectures: ["89"],
    })
  })
})
