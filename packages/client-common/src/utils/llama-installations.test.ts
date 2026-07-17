import { Option } from "effect"
import { describe, expect, it } from "vitest"
import type { LocalInferenceState } from "@magnitudedev/sdk"
import {
  deriveLlamaCppInstallationChatNotice,
  deriveLlamaCppInstallationManagementView,
} from "./llama-installations"

const state = (builds: readonly number[], selectedIndex: Option.Option<number>): LocalInferenceState => ({
  usage: null,
  activeBinding: null,
  llamaCpp: {
    minimumBuild: 8868,
    recommendedBuild: 10011,
    installations: builds.map((build, index) => ({
      id: `installation-${index}`,
      executables: {
        serverPath: `/bin/llama-server-${index}`,
        fitParamsPath: `/bin/llama-fit-params-${index}`,
      },
      build,
      ownership: "user" as const,
      discoveries: [{ _tag: "Path" as const, requestedPath: `/bin/llama-${index}`, priority: index }],
    })),
    selectedInstallationId: Option.map(selectedIndex, (index) => `installation-${index}`),
    activeManagedInstallationId: Option.none(),
    managedInstall: {
      availability: { _tag: "Available", build: 10011 },
      operation: { _tag: "Idle" },
    },
    diagnostics: [],
  },
  host: { _tag: "Unavailable", message: "not needed" },
  choices: [],
  operations: [],
  recommendations: [],
  warnings: [],
})

describe("llama.cpp installation presentation", () => {
  it("derives ready exclusively from the registry selection", () => {
    const view = deriveLlamaCppInstallationManagementView(state([8680, 9000], Option.some(1)))
    expect(view.status).toBe("ready")
    expect(Option.getOrThrow(view.selected).build).toBe(9000)
  })

  it("shows the best detected outdated build and exact requirement", () => {
    const notice = deriveLlamaCppInstallationChatNotice(state([8000, 8680], Option.none()), true)
    expect(Option.getOrThrow(notice).prefix).toContain("b8680 is outdated")
    expect(Option.getOrThrow(notice).prefix).toContain("requires b8868")
    expect(Option.getOrThrow(notice).actionLabel).toBe("/settings")
  })

  it("only warns about a missing installation when local inference is relevant", () => {
    expect(Option.isNone(deriveLlamaCppInstallationChatNotice(state([], Option.none()), false))).toBe(true)
    expect(Option.getOrThrow(deriveLlamaCppInstallationChatNotice(state([], Option.none()), true)).kind).toBe("missing")
  })
})
