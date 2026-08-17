import { renderToStaticMarkup } from "react-dom/server"
import { Option } from "effect"
import { Result } from "@effect-atom/atom-react"
import { describe, expect, it } from "vitest"
import type { useOnboardingModelSetup } from "@magnitudedev/client-common"
import type {
  LocalModelRecommendationProgressStep,
  ModelFailure,
} from "@magnitudedev/sdk"
import { LocalModelOnboarding } from "./local-model-onboarding"

type Setup = ReturnType<typeof useOnboardingModelSetup>

const setupWithProgress = (
  progress: readonly LocalModelRecommendationProgressStep[],
  discoveryFailure: ModelFailure | null = null
): Setup => ({
  hardware: Result.initial(false),
  view: Result.success({
    _tag: "Open",
    exitKind: "Skip",
    notice: Option.none(),
    content: {
      _tag: "Preparation",
      progress,
      discoveryFailure,
    },
  }),
  retry: () => {},
  open: () => {},
  select: () => {},
  cancel: () => {},
  exit: () => {},
})

describe("LocalModelOnboarding preparation", () => {
  it("shows the active authoritative step with count, ETA, and progress", () => {
    const html = renderToStaticMarkup(
      <LocalModelOnboarding
        setup={setupWithProgress([
          {
            id: "hardware",
            status: {
              _tag: "Completed",
              startedAtMs: 0,
              durationMs: 120,
              cached: false,
            },
            completedItems: Option.none(),
            totalItems: Option.none(),
            estimatedRemainingMs: Option.none(),
          },
          {
            id: "assessment",
            status: { _tag: "Running", startedAtMs: 120 },
            completedItems: Option.some(3),
            totalItems: Option.some(12),
            estimatedRemainingMs: Option.some(5_000),
          },
        ])}
      />
    )

    expect(html).toContain("Assessing models for this machine")
    expect(html).toContain("3 of 12 models · About 5s remaining")
    expect(html).toContain('aria-valuenow="25"')
    expect(html).toContain("Detected hardware")
    expect(html).not.toContain(">Hardware<")
    expect(html).not.toContain(">Assessment<")
  })

  it("shows the exact discovery failure instead of indefinite activity", () => {
    const failure: ModelFailure = {
      code: "inventory_unavailable",
      message: "Inventory database is unavailable",
      retryable: true,
    }
    const setup = setupWithProgress(
      [
        {
          id: "inventory",
          status: {
            _tag: "Failed",
            startedAtMs: 0,
            durationMs: 200,
            failure,
          },
          completedItems: Option.none(),
          totalItems: Option.none(),
          estimatedRemainingMs: Option.none(),
        },
      ],
      failure
    )

    const html = renderToStaticMarkup(<LocalModelOnboarding setup={setup} />)

    expect(html).toContain("Inventory database is unavailable")
    expect(html).toContain("Checking downloaded models")
  })
})
