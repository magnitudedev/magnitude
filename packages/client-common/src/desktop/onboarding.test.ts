import { describe, expect, it } from "vitest"
import { Option } from "effect"
import type { CatalogLocalModel } from "@magnitudedev/sdk"
import { desktopSetupModelOutcome } from "./onboarding"

const installed = (residencyState: Extract<CatalogLocalModel["acquisitionState"], { _tag: "Installed" }>["residencyState"]): Pick<CatalogLocalModel, "acquisitionState"> => ({
  acquisitionState: {
    _tag: "Installed",
    installation: { _tag: "Resolved", primaryPath: "/models/test.gguf", installedBytes: 10, ownership: "Magnitude" },
    residencyState,
  },
})

describe("desktop onboarding observations", () => {
  it("does not advance an admitted but incomplete download", () => {
    expect(desktopSetupModelOutcome({ acquisitionState: {
      _tag: "Installing", progress: { stage: "downloading", completedBytes: 5, totalBytes: 10, bytesPerSecond: Option.none() },
    } }, "Installing")).toEqual({ _tag: "Waiting" })
    expect(desktopSetupModelOutcome(installed({ _tag: "Unloaded" }), "Installing")).toEqual({ _tag: "Ready" })
  })

  it("retains actionable disk guidance and does not load a failed installation", () => {
    expect(desktopSetupModelOutcome({ acquisitionState: {
      _tag: "InstallFailed", failure: { _tag: "InsufficientDiskSpace", requiredBytes: 37_923_968_128, availableBytes: 33_440_665_600 },
    } }, "Installing")).toEqual({ _tag: "Failed", message: "Not enough disk space. Free at least 4.48 GB and try again." })
  })

  it("does not treat cancellation as installation completion", () => {
    expect(desktopSetupModelOutcome({ acquisitionState: { _tag: "NotInstalled" } }, "Installing")._tag).toBe("Failed")
  })

  it("requires loaded residency and rejects a stopped or removed model", () => {
    expect(desktopSetupModelOutcome(installed({ _tag: "Unloaded" }), "Loading")._tag).toBe("Failed")
    expect(desktopSetupModelOutcome({ acquisitionState: { _tag: "NotInstalled" } }, "Loading")._tag).toBe("Failed")
    expect(desktopSetupModelOutcome(installed({ _tag: "Ready", allocation: {
      contextWindowTokens: 4096, parallelSequences: 1, physicalContextTokens: 4096, memoryDomains: [],
    } }), "Loading")).toEqual({ _tag: "Ready" })
  })
})
