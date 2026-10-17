import { renderToStaticMarkup } from "react-dom/server"
import { describe, expect, it } from "vitest"
import { Option } from "effect"
import { HuggingFaceArtifactSelectorSchema, HuggingFaceRepositoryIdSchema } from "@magnitudedev/sdk"
import { DrafterList, attachedDrafterSelection, drafterChoice, drafterKey, selectedDrafterCandidate, type DrafterSelection } from "./drafter-list"
import { added, attachedDrafter, customModel, drafterCandidate } from "../../test/models/fixtures"

const recommended = drafterCandidate()
const builtIn = drafterCandidate({ category: "BuiltIn", reference: { _tag: "Embedded" }, name: "MTP heads", label: "Built into this model", sizeBytes: 0, onThisComputer: true })
const rejected = drafterCandidate({
  category: "NextToModel",
  reference: { _tag: "HuggingFace", repository: HuggingFaceRepositoryIdSchema.make("z-lab/Qwen3.6-35B-A3B-DFlash"), commit: "abc", path: HuggingFaceArtifactSelectorSchema.make("dflash.gguf") },
  name: "z-lab/Qwen3.6-35B-A3B-DFlash/dflash.gguf",
  label: "In this repository",
  verdict: { _tag: "Rejected", reasons: [{ _tag: "TargetMismatch", detail: "shape" }], detail: "PRIVATE" },
})
const untested = drafterCandidate({ category: "Other", reference: { _tag: "Local", path: "/models/draft.gguf" }, name: "draft.gguf", label: "Untested", lineage: "None", sizeBytes: 500_000_000, onThisComputer: true })
const candidates = [rejected, recommended, builtIn, untested]
const render = (selection: DrafterSelection) => renderToStaticMarkup(<DrafterList candidates={candidates} selection={selection} onSelect={() => {}} onCheckOther={() => {}} />)

describe("drafter list", () => {
  it("lists candidates in order by name with their note once, then No drafter and Use a different drafter…", () => {
    const html = render({ _tag: "Auto" })
    const order = ["z-lab/Qwen3.6-35B-A3B-DFlash/dflash.gguf", "Magnitude DFlash for Qwen3.6 35B-A3B", "MTP heads", "draft.gguf", "No drafter"].map(text => html.indexOf(text))
    expect(order.every((index, position) => index >= 0 && (position === 0 || index > order[position - 1]!))).toBe(true)
    expect(html).toContain(">Trained for Qwen3.6 35B-A3B<")
    expect(html).toContain(">MTP heads<")
    expect(html).toContain(">Built into this model<")
    expect(html).toContain(">In this repository<")
    expect(html).toContain(">Untested<")
    expect(html.match(/Trained for/g)).toHaveLength(1)
    expect(html.match(/>Recommended</g)).toHaveLength(1)
    expect(html.indexOf(">Recommended<")).toBeLessThan(html.indexOf("MTP heads"))
    expect(html).toContain("900 MB")
    expect(html).toContain("No download")
    expect(html).toContain("Responses are generated without speculative decoding")
    expect(html).toContain("Use a different drafter…")
  })
  it("disables a drafter the engine rejects and says why", () => {
    const html = render({ _tag: "Auto" })
    expect(html).toContain("Its drafter wasn&#x27;t built for this model")
    expect(html).not.toContain("PRIVATE")
    expect(html.match(/type="radio"[^>]*disabled=""/g)).toHaveLength(1)
  })
  it("preselects the first drafter the engine binds", () => {
    expect(render({ _tag: "Auto" })).toMatch(/type="radio"[^>]*checked=""[^>]*\/><span class="min-w-0 flex-1"><span class="flex min-w-0 items-center gap-2"><span class="truncate text-sm font-medium"[^>]*>Magnitude DFlash/)
    expect(selectedDrafterCandidate({ _tag: "Auto" }, [rejected, builtIn])).toEqual(Option.some(builtIn))
    expect(selectedDrafterCandidate({ _tag: "Auto" }, [rejected])).toEqual(Option.none())
    expect(selectedDrafterCandidate({ _tag: "Auto" }, [rejected, untested])).toEqual(Option.none())
  })
  it("never resolves to a rejected or stale pick and sends Auto, None, or the chosen reference", () => {
    expect(selectedDrafterCandidate({ _tag: "Candidate", key: drafterKey(rejected) }, candidates)).toEqual(Option.some(recommended))
    expect(selectedDrafterCandidate({ _tag: "Candidate", key: "gone" }, candidates)).toEqual(Option.some(recommended))
    expect(drafterChoice({ _tag: "Auto" }, candidates)).toEqual({ _tag: "Auto" })
    expect(drafterChoice({ _tag: "None" }, candidates)).toEqual({ _tag: "None" })
    expect(drafterChoice({ _tag: "Candidate", key: drafterKey(untested) }, candidates)).toEqual({ _tag: "Reference", reference: untested.reference })
    expect(render({ _tag: "None" })).toMatch(/checked=""[^>]*\/><span class="min-w-0 flex-1"><span class="block text-sm font-medium">No drafter/)
  })
  it("opens Change on the attached drafter, or on the recommendation when there is none", () => {
    const attached = customModel(added, { drafter: Option.some(attachedDrafter({ _tag: "Present" }, { reference: builtIn.reference, name: "MTP heads", label: "Built into this model", sizeBytes: 0, verdict: builtIn.verdict })) })
    expect(attachedDrafterSelection(attached, candidates)).toEqual({ _tag: "Candidate", key: drafterKey(builtIn) })
    expect(attachedDrafterSelection(customModel(added), candidates)).toEqual({ _tag: "Auto" })
  })
  it("opens Change on No drafter for a model whose drafter was turned off", () => {
    expect(attachedDrafterSelection(customModel(added, { drafterChoice: { _tag: "None" }, recommendedDrafter: Option.some(recommended) }), candidates)).toEqual({ _tag: "None" })
  })
})
