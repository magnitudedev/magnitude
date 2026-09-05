import { renderToStaticMarkup } from "react-dom/server"
import { Option } from "effect"
import { describe, expect, it, vi } from "vitest"
import type { ServiceLifecycleState } from "@magnitudedev/client-common"
import { AcnBootstrapScreen } from "./acn-bootstrap-screen"

const render = (state: Exclude<ServiceLifecycleState, { readonly _tag: "Ready" }>) =>
  renderToStaticMarkup(
    <AcnBootstrapScreen state={state} onRetry={() => {}} />
  )

describe("AcnBootstrapScreen", () => {
  it.each([
    ["PreparingAcn", "Preparing background server"],
    ["WaitingForOwner", "Waiting for previous Magnitude process"],
    ["ResolvingLocalInference", "Preparing local inference"],
    ["LaunchingLocalInference", "Starting local inference"],
  ] as const)("renders the %s startup phase", (phase, label) => {
    const html = render({ _tag: "Starting", phase })
    expect(html).toContain(label)
  })

  it("renders the transparent Magnitude mark without a manufactured container", () => {
    const html = render({ _tag: "Starting", phase: "PreparingAcn" })
    expect(html).toContain('<img src="data:image/svg+xml')
    expect(html).toContain('aria-hidden="true"')
    expect(html).toContain('class="mx-auto mb-6 h-auto w-[82px]"')
  })

  it("renders backend preparation with the hardware label", () => {
    expect(
      render({
        _tag: "Starting",
        phase: {
          _tag: "PreparingBackend",
          backend: { _tag: "Metal", hardwareLabel: "Apple M4 Max" },
        },
      })
    ).toContain("Preparing Metal backend for Apple M4 Max")
  })

  it("renders real installation progress and exact byte detail", () => {
    const html = render({
      _tag: "Installing",
      phase: "DownloadingInferenceEngine",
      overallProgress: 0.887,
      detailIsExact: true,
      detail: Option.some({
        completed: 19 * 1024 * 1024,
        totalBytes: 20 * 1024 * 1024,
        unit: "Bytes",
        attempt: Option.some(1),
      }),
    })

    expect(html).toContain("Installing Magnitude")
    expect(html).toContain("Downloading inference engine")
    expect(html).toContain('aria-valuenow="88"')
    expect(html).toContain("19.0 MiB of 20.0 MiB")
  })

  it("does not present estimated byte detail as exact", () => {
    const html = render({
      _tag: "Installing",
      phase: "DownloadingInferenceEngine",
      overallProgress: 0.5,
      detailIsExact: false,
      detail: Option.some({
        completed: 10,
        totalBytes: 20,
        unit: "Bytes",
        attempt: Option.none(),
      }),
    })

    expect(html).not.toContain("MiB of")
  })

  it("distinguishes installation failure and respects available actions", () => {
    const onRetry = vi.fn()
    const onQuit = vi.fn()
    const html = renderToStaticMarkup(
      <AcnBootstrapScreen
        state={{
          _tag: "Failed",
          stage: "PrepareLocalInference",
          message: "Inference engine download failed",
          retryable: true,
        }}
        onRetry={onRetry}
        onQuit={onQuit}
      />
    )

    expect(html).toContain("Magnitude failed to install")
    expect(html).toContain("Inference engine download failed")
    expect(html).toContain("Retry")
    expect(html).toContain("Quit")
  })

  it("does not offer retry for a non-retryable failure", () => {
    const html = render({
      _tag: "Failed",
      stage: "Connect",
      message: "Connection rejected",
      retryable: false,
    })

    expect(html).toContain("Magnitude failed to start")
    expect(html).not.toContain(">Retry<")
  })
})
