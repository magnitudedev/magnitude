import { renderToStaticMarkup } from "react-dom/server"
import { describe, expect, it } from "vitest"
import { Option } from "effect"
import { HuggingFaceArtifactSelectorSchema, HuggingFaceRepositoryIdSchema } from "@magnitudedev/sdk"
import { DrafterList, attachedDrafterSelection, drafterCandidateAcceptance, drafterCandidateSlower, drafterChoice, drafterKey, selectedDrafterCandidate, type DrafterSelection } from "./drafter-list"
import { added, attachedDrafter, catalogModel, customModel, drafterCandidate, measured } from "../../test/models/fixtures"

const recommended = drafterCandidate()
const builtIn = drafterCandidate({ category: "BuiltIn", reference: { _tag: "Embedded" }, label: "MTP heads", sizeBytes: 0, onThisComputer: true })
const rejected = drafterCandidate({
  category: "HuggingFace",
  reference: { _tag: "HuggingFace", repository: HuggingFaceRepositoryIdSchema.make("z-lab/Qwen3.6-35B-A3B-DFlash"), commit: "abc", path: HuggingFaceArtifactSelectorSchema.make("dflash.gguf") },
  label: "z-lab DFlash",
  verdict: { _tag: "Rejected", reasons: [{ _tag: "TargetMismatch", detail: "shape" }], detail: "PRIVATE" },
})
const untested = drafterCandidate({ category: "Other", reference: { _tag: "Local", path: "/models/draft.gguf" }, label: "draft.gguf", lineage: "None", sizeBytes: 500_000_000, onThisComputer: true })
const candidates = [recommended, builtIn, rejected, untested]
const render = (selection: DrafterSelection) => renderToStaticMarkup(<DrafterList candidates={candidates} models={[catalogModel()]} selection={selection} onSelect={() => {}} onCheckOther={() => {}} />)

describe("drafter list", () => {
  it("lists candidates in order with their labels, then No drafter and Use a different drafter…", () => {
    const html = render({ _tag: "Auto" })
    const order = ["Magnitude DFlash for Qwen3.6 35B-A3B", "MTP heads", "z-lab DFlash", "draft.gguf", "No drafter"].map(text => html.indexOf(text))
    expect(order.every((index, position) => index >= 0 && (position === 0 || index > order[position - 1]!))).toBe(true)
    expect(html).toContain("Recommended · Trained for Qwen3.6 35B-A3B")
    expect(html).toContain("Built into this model")
    expect(html).toContain("z-lab · z-lab/Qwen3.6-35B-A3B-DFlash")
    expect(html).toContain("900 MB")
    expect(html).toContain("No download")
    expect(html).toContain("Responses are generated without speculative decoding")
    expect(html).toContain("Use a different drafter…")
  })
  it("disables a drafter the engine rejects and says why, and warns about an untested one", () => {
    const html = render({ _tag: "Auto" })
    expect(html).toContain("Its drafter wasn&#x27;t built for this model")
    expect(html).not.toContain("PRIVATE")
    expect(html).toContain("Untested with this model")
    expect(html.match(/type="radio"[^>]*disabled=""/g)).toHaveLength(1)
  })
  it("preselects the first drafter the engine binds", () => {
    expect(render({ _tag: "Auto" })).toMatch(/type="radio"[^>]*checked=""[^>]*\/><span class="min-w-0 flex-1"><span class="block truncate text-sm font-medium">Magnitude DFlash/)
    expect(selectedDrafterCandidate({ _tag: "Auto" }, [rejected, builtIn])).toEqual(Option.some(builtIn))
    expect(selectedDrafterCandidate({ _tag: "Auto" }, [rejected])).toEqual(Option.none())
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
    const attached = customModel(added, { drafter: Option.some(attachedDrafter({ _tag: "Present" }, { reference: builtIn.reference, label: "MTP heads", sizeBytes: 0, verdict: builtIn.verdict })) })
    expect(attachedDrafterSelection(attached, candidates)).toEqual({ _tag: "Candidate", key: drafterKey(builtIn) })
    expect(attachedDrafterSelection(customModel(added), candidates)).toEqual({ _tag: "Auto" })
  })
  it("shows each candidate's measured acceptance, and warns when it made responses slower", () => {
    const fast = drafterCandidate({ measuredAcceptance: Option.some(measured(0.58)) })
    const slow = drafterCandidate({ category: "BuiltIn", reference: { _tag: "Embedded" }, label: "MTP heads", sizeBytes: 0, onThisComputer: true, measuredAcceptance: Option.some(measured(0.124, true)) })
    expect(drafterCandidateAcceptance(fast)).toEqual(Option.some("58% accepted"))
    expect(drafterCandidateAcceptance(recommended)).toEqual(Option.none())
    expect(drafterCandidateSlower(slow)).toBe(true)
    expect(drafterCandidateSlower(fast)).toBe(false)
    const html = renderToStaticMarkup(<DrafterList candidates={[fast, slow, untested]} models={[catalogModel()]} selection={{ _tag: "Auto" }} onSelect={() => {}} onCheckOther={() => {}} />)
    expect(html).toContain("Recommended · Trained for Qwen3.6 35B-A3B · 58% accepted")
    expect(html).toContain("Built into this model · 12% accepted")
    expect(html.match(/Slower than no drafter on this computer/g)).toHaveLength(1)
    expect(html.match(/% accepted/g)).toHaveLength(2)
  })
})
