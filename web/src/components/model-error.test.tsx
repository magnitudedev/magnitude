import { renderToStaticMarkup } from "react-dom/server"
import { describe, expect, it } from "vitest"
import { LowMemoryModelInstanceFailureSchema, LocalModelMutationFailed } from "@magnitudedev/sdk"
import { Option, Schema } from "effect"
import { ModelLoadFailureIndicator, ModelLoadNotice, downloadNotice, modelCommandNotice, modelRemovalNotice } from "./model-error"
import { ErrorNotice, NoticeAction } from "./error-notice"

const memory = Schema.decodeUnknownSync(LowMemoryModelInstanceFailureSchema)({
  _tag: "LowMemory", code: "low_memory", message: "PRIVATE worker diagnostics 32358673408 bytes", retryable: true,
  requiredMemoryBytes: 32358673408, systemReserveBytes: 6871947673,
  allocationHeadroomBytes: 33741848576, loadBoundaryBytes: 26869900903,
  minimumAdditionalAvailableBytes: 5488772506,
})
describe("desktop failure presentation", () => {
  it("uses authoritative memory quantities and rounds required memory upward", () => {
    const html = renderToStaticMarkup(<ModelLoadNotice failure={memory} />)
    expect(html).toContain("Free up 5.2 GB")
    expect(html).toContain("30.2 GB")
    expect(html).toContain("6.4 GB")
    expect(html).toContain("31.4 GB")
    expect(html).toContain("Memory breakdown")
    expect(html).not.toMatch(/PRIVATE|bytes|32358673408|title=/)
    expect(html.match(/role="alert"/g)).toHaveLength(1)
  })
  it("marks a failed row with a labelled amber indicator instead of an alert", () => {
    const html = renderToStaticMarkup(<ModelLoadFailureIndicator failure={memory} />)
    expect(html).toContain('aria-label="Not enough memory to load this model"')
    expect(html).toContain("text-amber-500")
    expect(html).not.toMatch(/role="alert"|PRIVATE|bytes|32358673408/)
  })
  it("does not expose diagnostics from unknown load, download, or command failures", () => {
    const html = renderToStaticMarkup(<ModelLoadNotice failure={{ code: "worker_lost", message: "PRIVATE stack trace", retryable: true }} />)
    expect(html).toContain("This model couldn’t start")
    expect(html).not.toContain("PRIVATE")
    expect(JSON.stringify(downloadNotice({ _tag: "Internal", message: "PRIVATE" }))).not.toContain("PRIVATE")
    const rejection = new LocalModelMutationFailed({ code: "unknown", message: "PRIVATE", retryable: false })
    expect(JSON.stringify(modelCommandNotice({ operation: "remove", rejection: Option.some(rejection) }))).not.toContain("PRIVATE")
  })
  it("explains retained files and an interrupted load without inventing remediation", () => {
    expect(modelRemovalNotice({ code: "model_removal_retained_external", message: "PRIVATE", retryable: false }).description).toContain("external model cache")
    expect(modelRemovalNotice({ code: "model_removal_retained_shared", message: "PRIVATE", retryable: false }).description).toContain("Another model")
    expect(modelCommandNotice({ operation: "load", rejection: Option.some(new LocalModelMutationFailed({ code: "model_instance_stopped", message: "PRIVATE", retryable: true })) })).toMatchObject({ severity: "info", title: "Loading was stopped" })
  })
  it("keeps disk space separate from RAM and preserves button semantics for text actions", () => {
    expect(downloadNotice({ _tag: "InsufficientDiskSpace", requiredBytes: 10_000_000_000, availableBytes: 1_600_000_000 }).description).toContain("8.4 GB on the drive")
    const html = renderToStaticMarkup(<ErrorNotice title="Couldn’t load" actions={<NoticeAction disabled onClick={() => {}}>Load again</NoticeAction>} />)
    expect(html).toContain("<button")
    expect(html).toContain('disabled=""')
    expect(html).toContain("focus-visible:")
    expect(html).not.toContain('href=')
    expect(html).not.toContain("truncate")
  })
})
