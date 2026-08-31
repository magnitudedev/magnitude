import { renderToStaticMarkup } from "react-dom/server"
import { Option } from "effect"
import { Result } from "@effect-atom/atom-react"
import { describe, expect, it } from "vitest"
import type { useOnboardingModelSetup } from "@magnitudedev/client-common"
import { LocalModelOnboarding } from "./local-model-onboarding"

type Setup = ReturnType<typeof useOnboardingModelSetup>

const preparingSetup = (): Setup => ({
  hardware: Result.initial(false),
  view: Result.success({
    _tag: "Open",
    exitKind: "Skip",
    notice: Option.none(),
    content: {
      _tag: "Preparation",
    },
  }),
  retry: () => {},
  open: () => {},
  select: () => {},
  setRankingControls: () => {},
  cancel: () => {},
  chooseAnother: () => {},
  back: () => {},
  continueWithHarness: () => {},
  exit: () => {},
})

describe("LocalModelOnboarding preparation", () => {
  it("renders five labeled Fast-to-Smart points without a memory control", () => {
    const setup = {
      hardware: Result.success({
        productName: Option.none(),
        processor: Option.some("Test CPU"),
        logicalCores: 8,
        totalSystemMemoryBytes: 64,
        accelerators: [],
        memoryDomains: [{ totalBytes: 64 }],
      }),
      view: Result.success({
        _tag: "Open",
        exitKind: "Skip",
        notice: Option.none(),
        content: {
          _tag: "Chooser",
          options: [],
          rankingControls: { fastToSmart: 0.5 },
          operation: Option.none(),
        },
      }),
      retry: () => {},
      open: () => {},
      select: () => {},
      setRankingControls: () => {},
      cancel: () => {},
      back: () => {},
      continueWithHarness: () => {},
      exit: () => {},
    } as unknown as Setup

    const html = renderToStaticMarkup(<LocalModelOnboarding setup={setup} />)

    expect(html).toContain('aria-label="Fast to Smart"')
    expect(html).toContain('max="4"')
    for (const label of ["Fastest", "Faster", "Balanced", "Smarter", "Smartest"]) {
      expect(html).toContain(`>${label}<`)
    }
    expect(html).not.toContain('aria-label="Memory budget"')
  })

  it("shows the explicit reconciliation activity", () => {
    const html = renderToStaticMarkup(
      <LocalModelOnboarding setup={preparingSetup()} />
    )

    expect(html).toContain("Reconciling local models")
    expect(html).toContain("Checking the model catalog and installed Hugging Face cache entries.")
    expect(html).toContain("animate-spin")
  })
})
