import { describe, expect, it } from "vitest"
import { makeHardware, makeInstalledCatalogModel } from "./fixtures/inference"
import { renderModelDetail } from "./models-status-runtime"

describe("a model that failed to load", () => {
  it("says what the failure means and gives the engine's own message as its reason", () => {
    const model = makeInstalledCatalogModel({
      acquisitionState: {
        _tag: "Installed",
        installation: { _tag: "Resolved", installedBytes: 1, primaryPath: "/models/catalog-model.gguf", ownership: "Magnitude" },
        residencyState: { _tag: "Failed", failure: { code: "worker_exited", message: "worker exited with status 134", retryable: true } },
      },
    })
    const detail = renderModelDetail({ model, models: [model], hardware: makeHardware(), drafters: { _tag: "NotApplicable" }, hidden: false })
    expect(detail).toMatch(/Runtime\s+Failed - The model’s engine stopped unexpectedly/)
    expect(detail).toMatch(/Reason\s+worker exited with status 134/)
  })
})
