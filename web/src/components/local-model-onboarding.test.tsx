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
      preparation: {
        discovery: { complete: false, modelsFound: 5 },
        assessment: { complete: false, settledModels: 12, totalModels: 18 },
      },
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

  it("shows the fixed live discovery and assessment rows", () => {
    const html = renderToStaticMarkup(
      <LocalModelOnboarding setup={preparingSetup()} />
    )

    expect(html).toContain("Discovering existing models")
    expect(html).toContain("· 5 found")
    expect(html).toContain("Assessing models · 12 of 18")
    expect(html.match(/animate-spin/g)).toHaveLength(2)
    expect(html).not.toContain("Discovery complete")
    expect(html).not.toContain("currently known")
  })

  it("omits the found count while discovery has found nothing", () => {
    const setup = preparingSetup()
    const html = renderToStaticMarkup(
      <LocalModelOnboarding
        setup={{
          ...setup,
          view: Result.success({
            notice: Option.none(),
            content: {
              _tag: "Preparation",
              preparation: {
                discovery: { complete: false, modelsFound: 0 },
                assessment: { complete: false, settledModels: 0, totalModels: 0 },
              },
            },
          }),
        }}
      />
    )

    expect(html).toContain("Discovering existing models")
    expect(html).toContain("Assessing models")
    expect(html).not.toContain("found")
    expect(html).not.toContain("Assessing models ·")
  })
})
