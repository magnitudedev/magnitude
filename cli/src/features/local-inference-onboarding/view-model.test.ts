import { describe, expect, test } from "vitest"
import type { LocalInferenceOnboardingSnapshot } from "@magnitudedev/sdk"
import {
  buildLocalInferenceSelections,
  formatBytes,
  formatContext,
  formatModelSize,
  selectionFidelity,
  selectionMetadata,
  shouldShowLocalInferenceOnboarding,
} from "./view-model"

const snapshot: LocalInferenceOnboardingSnapshot = {
  schemaVersion: 2,
  onboarding: { required: true },
  configuration: { usable: false },
  runtime: { status: "ready", canDownload: true, canActivate: true },
  running: [{
    choiceId: "running-1",
    source: "running",
    displayName: "Running model",
    providerModelId: "running",
    contextTokens: 32_768,
    fitClass: "unknown",
    managed: false,
    compatible: true,
    explanation: "healthy",
  }],
  downloaded: [
    {
      choiceId: "incompatible",
      source: "downloaded",
      displayName: "Too-small context",
      providerModelId: "small",
      contextTokens: 4_096,
      fitClass: "unknown",
      managed: true,
      compatible: false,
      explanation: "not compatible",
    },
    {
      choiceId: "cached-1",
      source: "downloaded",
      displayName: "Cached model",
      providerModelId: "cached",
      contextTokens: 65_536,
      fitClass: "cpu_or_unified",
      managed: true,
      compatible: true,
      explanation: "verified",
    },
  ],
  recommendations: [{
    configurationId: "config-1",
    catalogModelId: "catalog-1",
    badge: "recommended",
    displayName: "Download model",
    family: "test",
    architecture: "dense",
    quantization: {
      format: "UD-Q5_K_XL",
      bitsClass: "q5",
      quantAwareCheckpoint: false,
      fidelityLabel: "High fidelity with only minor quality loss",
      fidelityEvidence: "fidelity evidence",
      fidelitySourceUrl: "https://example.com/evidence",
    },
    repo: "org/repo",
    revision: "a".repeat(40),
    quantTag: "UD-Q5_K_XL",
    files: [],
    totalDownloadBytes: 10,
    sourcePageUrl: "https://example.com/model",
    license: { id: "apache-2.0", url: "https://example.com/license", acknowledgementRequired: false },
    contextTokens: 32_768,
    modelMaximumContextTokens: 262_144,
    estimatedRuntimeBytes: 20,
    stableCapacityBudgetBytes: 30,
    fitMarginBytes: 10,
    fitClass: "cpu_or_unified",
    constrainedContext: false,
    explanation: "fits",
  }],
  warnings: [],
}

describe("local inference onboarding view model", () => {
  test("orders compatible running, downloaded, then recommended choices", () => {
    expect(buildLocalInferenceSelections(snapshot).map((selection) => selection.id)).toEqual([
      "running-1",
      "cached-1",
      "config-1",
    ])
  })

  test("formats capacities and contexts for terminal cards", () => {
    expect(formatBytes(6_743_680_224)).toBe("6.28 GiB")
    expect(formatModelSize(6_743_680_224)).toBe("6.74 GB")
    expect(formatContext(65_536)).toBe("64K")
    expect(formatContext(200_192)).toBe("200K")
  })

  test("puts quant, size, parameters, and context on one metadata line", () => {
    const [running, , recommendation] = buildLocalInferenceSelections(snapshot)
    expect(running && selectionMetadata(running)).toBe("Quant unavailable · Size unavailable · 32K context")
    expect(running && selectionFidelity(running)).toBeNull()
    expect(recommendation && selectionMetadata(recommendation)).toBe("UD-Q5_K_XL · 0.00 GB · 32K context")
    expect(recommendation && selectionFidelity(recommendation)).toBe("High fidelity with only minor quality loss")
  })

  test("distinguishes Gemma effective parameters from MoE active parameters", () => {
    const recommendation = buildLocalInferenceSelections({
      ...snapshot,
      recommendations: [{
        ...snapshot.recommendations[0]!,
        displayName: "Gemma 4 E2B",
        totalParametersBillions: 5.1,
        effectiveParametersBillions: 2.3,
      }],
    }).find((selection) => selection.kind === "recommendation")

    expect(recommendation && selectionMetadata(recommendation)).toBe(
      "UD-Q5_K_XL · 0.00 GB · 5.1B total / 2.3B effective · 32K context",
    )
  })

  test("gates only on versioned walkthrough completion or an explicit rerun", () => {
    expect(shouldShowLocalInferenceOnboarding(snapshot, false)).toBe(true)
    expect(shouldShowLocalInferenceOnboarding({
      ...snapshot,
      onboarding: { required: false, completedVersion: 2 },
    }, false)).toBe(false)
    expect(shouldShowLocalInferenceOnboarding({
      ...snapshot,
      onboarding: { required: false, completedVersion: 2 },
    }, true)).toBe(true)
  })
})
