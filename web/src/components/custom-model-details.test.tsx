import { renderToStaticMarkup } from "react-dom/server"
import { describe, expect, it } from "vitest"
import { Option } from "effect"
import type { AttachedDrafterAvailability } from "@magnitudedev/sdk"
import { CustomModelDetails, DrafterNotices, DrafterOffer, customModelSource, drafterUpdateAvailable } from "./custom-model-details"
import { added, attachedDrafter, catalogModel, customModel, drafterCandidate, foundIn, imported, measured, repository } from "../../test/models/fixtures"

const attached = (availability: AttachedDrafterAvailability, updateAvailable = false) => Option.some(attachedDrafter(availability, { updateAvailable }))
const downloading: AttachedDrafterAvailability = { _tag: "Downloading", progress: { stage: "downloading", completedBytes: 450, totalBytes: 900, bytesPerSecond: Option.none() } }
const details = (model: ReturnType<typeof customModel>) => renderToStaticMarkup(<CustomModelDetails model={model} variantOfName={Option.some("Qwen3.6 35B-A3B")} disabled={false} onChangeDrafter={() => {}} onTurnOffDrafter={() => {}} onAddDrafter={() => {}} />)

const notices = (model: ReturnType<typeof customModel>) => renderToStaticMarkup(<DrafterNotices model={model} disabled={false} onChangeDrafter={() => {}} onTurnOffDrafter={() => {}} onRetry={() => {}} onCancel={() => {}} onDismiss={() => {}} />)

describe("custom model details", () => {
  it("names the source by origin: a pinned repository, an imported path, or where it was found", () => {
    expect(customModelSource(customModel(added))).toEqual({ label: `${repository} · pinned to 4f1c2e9`, href: Option.some(`https://huggingface.co/${repository}`) })
    expect(customModelSource(customModel(imported)).label).toBe("/Users/me/models/qwen.gguf")
    expect(customModelSource(customModel(foundIn({ _tag: "Ollama" }))).label).toBe("Ollama · /Users/me/.ollama/models/blobs/sha256-1")
  })
  it("shows Source, Variant of, and the drafter with measured acceptance, Change, and Turn off", () => {
    const html = details(customModel(added, {
      drafter: attached({ _tag: "Present" }),
      measuredAcceptance: Option.some(measured(0.58)),
    }))
    expect(html).toContain(`href="https://huggingface.co/${repository}"`)
    expect(html).toContain("Variant of")
    expect(html).toContain("Qwen3.6 35B-A3B")
    expect(html).toContain("Magnitude DFlash · 58% of drafted tokens accepted")
    expect(html).toContain(">Change<")
    expect(html).toContain(">Turn off<")
  })
  it("offers the recommended drafter when a model has none, and Change either way", () => {
    const offered = details(customModel(foundIn({ _tag: "Ollama" }), { recommendedDrafter: Option.some(drafterCandidate()) }))
    expect(offered).toContain("Faster with a drafter · Add (900 MB)")
    expect(offered).toContain(">Change<")
    expect(offered).not.toContain(">Turn off<")
    const none = details(customModel(imported))
    expect(none).toContain(">None<")
    expect(none).toContain(">Change<")
  })
  it("puts the drafter offer on the card only when one is recommended and none is attached", () => {
    expect(renderToStaticMarkup(<DrafterOffer model={customModel(added, { recommendedDrafter: Option.some(drafterCandidate({ onThisComputer: true })) })} disabled={false} onAdd={() => {}} />)).toContain("Faster with a drafter · Add<")
    expect(renderToStaticMarkup(<DrafterOffer model={customModel(added)} disabled={false} onAdd={() => {}} />)).toBe("")
    expect(renderToStaticMarkup(<DrafterOffer model={customModel(added, { drafter: attached({ _tag: "Present" }), recommendedDrafter: Option.some(drafterCandidate()) })} disabled={false} onAdd={() => {}} />)).toBe("")
  })
  it("reports a slowing drafter with Change and Turn off, and a drafter-only download's progress and failure", () => {
    const slowing = notices(customModel(added, { drafter: attached({ _tag: "Present" }), measuredAcceptance: Option.some(measured(0.1, true)) }))
    expect(slowing).toContain("This drafter is slowing responses")
    expect(slowing).toContain(">Change<")
    expect(slowing).toContain(">Turn off<")
    expect(notices(customModel(added, { drafter: attached(downloading) }))).toContain("Downloading the drafter for Qwen3.6 35B A3B Uncensored (Q4_K_P) · 50%")
    const failed = notices(customModel(added, { drafter: attached({ _tag: "DownloadFailed", failure: { _tag: "NetworkUnavailable" } }) }))
    expect(failed).toContain("Can’t reach the model source")
    expect(failed).toContain("Retry drafter download")
    expect(notices(customModel(added))).toBe("")
  })
  it("offers Cancel while the drafter downloads, and Retry and Dismiss once it fails", () => {
    const running = notices(customModel(imported, { drafter: attached(downloading) }))
    expect(running).toContain('role="status"')
    expect(running).toContain(">Cancel<")
    expect(running).not.toContain("Retry drafter download")
    const failed = notices(customModel(imported, { drafter: attached({ _tag: "DownloadFailed", failure: { _tag: "NetworkUnavailable" } }) }))
    expect(failed).toContain(">Retry drafter download<")
    expect(failed).toContain(">Dismiss<")
    expect(failed).not.toContain(">Cancel<")
    expect(notices(customModel(imported, { drafter: attached({ _tag: "Present" }) }))).toBe("")
  })
  it("shows Update for any custom model whose catalog drafter has a newer revision, once nothing is downloading it", () => {
    expect(drafterUpdateAvailable(customModel(foundIn({ _tag: "Ollama" }), { drafter: attached({ _tag: "Present" }, true) }))).toBe(true)
    expect(drafterUpdateAvailable(customModel(added, { drafter: attached(downloading, true) }))).toBe(false)
    expect(drafterUpdateAvailable(customModel(added, { drafter: attached({ _tag: "Present" }) }))).toBe(false)
    expect(drafterUpdateAvailable(customModel(added))).toBe(false)
    expect(drafterUpdateAvailable(catalogModel())).toBe(false)
    expect(details(customModel(imported, { drafter: attached({ _tag: "Present" }, true) }))).toContain("Magnitude DFlash · Update available")
  })
})
