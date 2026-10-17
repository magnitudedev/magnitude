import { renderToStaticMarkup } from "react-dom/server"
import type { ReactNode } from "react"
import { describe, expect, it } from "vitest"
import { Option } from "effect"
import { Dialog } from "./ui/dialog"
import { FindStepView, InspectedRepository, LocalFileSummary, LocalFileView, NativeFileChoice, ggufOpenRequest, RepositoryFooter, RepositoryView, SearchResultsView, VerdictLabel, addVersionRequest, confirmAction, formatDownloads, initialVersion, parseModelReference, unselectableVersionNotice, unsupportedVersionLabel, versionVerdict, type FindTab } from "./add-model-dialog"
import { catalogId, catalogModel, drafterCandidate, inspection, repository, version, doesNotFit, unsupported, unreadable } from "../../test/models/fixtures"

const inDialog = (node: ReactNode) => renderToStaticMarkup(<Dialog open>{node}</Dialog>)
const models = [catalogModel()]
const find = (tab: FindTab, overrides: { results?: ReactNode; pasteInvalid?: boolean } = {}) => inDialog(<FindStepView tab={tab} onTab={() => {}} typed="qwen3.6 35b abliterated" onType={() => {}} onSearch={() => {}}
  results={overrides.results ?? null} pasted="not a link" pasteInvalid={overrides.pasteInvalid ?? false} onPaste={() => {}} onOpenPasted={() => {}} />)

describe("paste a link", () => {
  it("accepts a Hugging Face URL, owner/repo, and owner/repo:QUANT", () => {
    expect(parseModelReference(" https://huggingface.co/unsloth/Qwen3.6-27B-GGUF ")).toEqual(Option.some("https://huggingface.co/unsloth/Qwen3.6-27B-GGUF"))
    expect(parseModelReference("https://huggingface.co/unsloth/Qwen3.6-27B-GGUF/blob/main/Qwen3.6-27B-Q4_K_M.gguf")).toEqual(Option.some("https://huggingface.co/unsloth/Qwen3.6-27B-GGUF/blob/main/Qwen3.6-27B-Q4_K_M.gguf"))
    expect(parseModelReference("unsloth/Qwen3.6-27B-GGUF")).toEqual(Option.some("unsloth/Qwen3.6-27B-GGUF"))
    expect(parseModelReference("unsloth/Qwen3.6-27B-GGUF:Q4_K_M")).toEqual(Option.some("unsloth/Qwen3.6-27B-GGUF:Q4_K_M"))
  })
  it("refuses anything else before asking the service", () => {
    for (const text of ["", "qwen", "https://example.com/owner/repo", "owner/repo/extra", "owner/repo:"]) expect(parseModelReference(text)).toEqual(Option.none())
  })
})

describe("find step", () => {
  it("titles the dialog, explains what is accepted, and offers Search and Paste a link", () => {
    const html = find("search")
    expect(html).toContain("Add a model from Hugging Face")
    expect(html).toContain("other quantizations, finetunes, and uncensored versions")
    expect(html).toMatch(/role="tab"[^>]*aria-selected="true"[^>]*>.*Search/)
    expect(html).toContain("Paste a link")
    expect(html).toContain('value="qwen3.6 35b abliterated"')
    expect(html).toContain("Press Enter to search")
  })
  it("explains the accepted forms on the paste tab and flags a link it can't use", () => {
    expect(find("paste")).toContain("Accepts a Hugging Face link, owner/repo, or owner/repo:QUANT.")
    const invalid = find("paste", { pasteInvalid: true })
    expect(invalid).toContain('aria-invalid="true"')
    expect(invalid).toContain("That isn’t a Hugging Face link.")
  })
  it("lists results with uploader, lineage, and downloads, then counts the hidden ones", () => {
    const html = renderToStaticMarkup(<SearchResultsView models={models} onOpen={() => {}} onRetry={() => {}} state={{ _tag: "Loaded", response: {
      hiddenCount: 4,
      results: [{ repository, displayName: "Huihui Qwen3.6 35B A3B abliterated", uploader: "huihui-ai", variantOf: Option.some(catalogId), downloads: 172_100, architecture: Option.some("qwen35moe"), unsupported: Option.none() }],
    } }} />)
    expect(html).toContain("Huihui Qwen3.6 35B A3B abliterated")
    expect(html).toContain("huihui-ai · Variant of Qwen3.6 35B-A3B")
    expect(html).toContain(formatDownloads(172_100))
    expect(html).toContain("Qwen logo")
    expect(html).toContain("4 more results use architectures Magnitude doesn’t run")
  })
  it("distinguishes searching, no matches, and a failed search", () => {
    expect(renderToStaticMarkup(<SearchResultsView models={models} onOpen={() => {}} onRetry={() => {}} state={{ _tag: "Loading" }} />)).toContain('aria-busy="true"')
    const empty = renderToStaticMarkup(<SearchResultsView models={models} onOpen={() => {}} onRetry={() => {}} state={{ _tag: "Loaded", response: { results: [], hiddenCount: 1 } }} />)
    expect(empty).toContain("No supported GGUF repositories match this search.")
    expect(empty).toContain("1 more result uses an architecture Magnitude doesn’t run")
    const failed = renderToStaticMarkup(<SearchResultsView models={models} onOpen={() => {}} onRetry={() => {}} state={{ _tag: "Failed", notice: { title: "Couldn’t reach Hugging Face" } }} />)
    expect(failed).toContain("Couldn’t reach Hugging Face")
    expect(failed).toContain("Try again")
  })
})

describe("repository step", () => {
  it("gives each version one verdict from the engine", () => {
    expect(versionVerdict(version("Q4_K_P", { recommended: true }))).toMatchObject({ label: "Recommended", selectable: true })
    expect(versionVerdict(version("Q5_K_P"))).toMatchObject({ label: "Fits your machine", selectable: true })
    expect(versionVerdict(version("Q8_K_P", { assessment: Option.some(doesNotFit) }))).toMatchObject({ label: "Too large for this machine", selectable: false })
    expect(versionVerdict(version("IQ2_M", { assessment: Option.some(unsupported({ _tag: "WeightTypes", types: [{ ggmlType: "IQ2_S", tensorCount: 100 }] })) }))).toMatchObject({ label: "Not supported yet", detail: "Uses IQ2_S weights, which Magnitude doesn't support yet", selectable: false })
  })
  it("says Not supported only for an architecture without kernels, File is damaged when unreadable, and Not supported yet otherwise", () => {
    expect(unsupportedVersionLabel([{ _tag: "RopeScaling", kind: "yarn" }, { _tag: "TiedOutputHead" }])).toBe("Not supported yet")
    expect(unsupportedVersionLabel([{ _tag: "Backend", backend: "metal" }])).toBe("Not supported yet")
    expect(unsupportedVersionLabel([{ _tag: "ChatTemplate" }])).toBe("Not supported yet")
    expect(unsupportedVersionLabel([{ _tag: "Unknown", kind: "Sliding", detail: "x" }])).toBe("Not supported yet")
    expect(unsupportedVersionLabel([{ _tag: "Unknown", kind: "Sliding", detail: "x" }, { _tag: "ChatTemplate" }])).toBe("Not supported yet")
    expect(unsupportedVersionLabel([{ _tag: "Architecture", architecture: "deepseek4" }])).toBe("Not supported")
    expect(unsupportedVersionLabel([{ _tag: "WeightTypes", types: [{ ggmlType: "IQ1_M", tensorCount: 1 }] }, { _tag: "Architecture", architecture: "deepseek4" }])).toBe("Not supported")
    expect(unsupportedVersionLabel([{ _tag: "Unreadable" }, { _tag: "WeightTypes", types: [{ ggmlType: "IQ1_M", tensorCount: 1 }] }])).toBe("File is damaged")
    expect(versionVerdict(version("Q4_K_P", { assessment: Option.none() }))).toMatchObject({ label: "Not checked yet", selectable: true })
  })
  it("shows an unknown reason's detail behind the version's verdict", () => {
    const sliding = version("Q4_K_M", { assessment: Option.some(unsupported({ _tag: "Unknown", kind: "Sliding", detail: "sliding window 4096" })) })
    expect(versionVerdict(sliding)).toMatchObject({ label: "Not supported yet", detail: "Magnitude can't run this model (sliding window 4096)", selectable: false })
    const html = inDialog(<RepositoryView inspection={inspection({ versions: [version("Q4_K_P", { recommended: true }), sliding] })} models={models} selectedVersion={Option.some("Q4_K_P")} onSelectVersion={() => {}} />)
    expect(html).toContain('Not supported yet<span class="sr-only">: Magnitude can&#x27;t run this model (sliding window 4096)</span>')
  })
  it("puts a verdict's reason on a focusable trigger, so keyboard users reach it", () => {
    const html = renderToStaticMarkup(<VerdictLabel verdict={{ label: "Not supported yet", detail: "Uses IQ2_S weights, which Magnitude doesn't support yet", selectable: false, recommended: false }} />)
    expect(html).toMatch(/^<button type="button"[^>]*>Not supported yet<span class="sr-only">: Uses IQ2_S weights, which Magnitude doesn&#x27;t support yet<\/span><\/button>$/)
    expect(renderToStaticMarkup(<VerdictLabel verdict={{ label: "Fits your machine", selectable: true, recommended: false }} />)).not.toContain("<button")
  })
  it("starts on a linked file, then the pasted version, then the recommended one", () => {
    expect(initialVersion(inspection(), Option.some("Q8_K_P.gguf"))).toEqual(Option.some("Q8_K_P"))
    expect(initialVersion(inspection({ selectedVersion: Option.some("Q2_K_P") }), Option.none())).toEqual(Option.some("Q2_K_P"))
    expect(initialVersion(inspection(), Option.none())).toEqual(Option.some("Q4_K_P"))
    expect(initialVersion(inspection({ versions: [version("Q5_K_P")] }), Option.none())).toEqual(Option.some("Q5_K_P"))
  })
  it("totals the version's download (target and projector) with the chosen drafter, and only the drafter for a version already here", () => {
    const withProjector = inspection({ projector: Option.some({ path: "mmproj.gguf", sizeBytes: 1_000_000_000 }) })
    const toDownload = version("Q4_K_P", { downloadBytes: 24_400_000_000 })
    const drafter = Option.some(drafterCandidate())
    expect(confirmAction(withProjector, toDownload, drafter)).toEqual({ label: "Download (25.3 GB)", download: true })
    expect(confirmAction(withProjector, toDownload, Option.none())).toEqual({ label: "Download (24.4 GB)", download: true })
    expect(confirmAction(withProjector, toDownload, Option.some(drafterCandidate({ onThisComputer: true })))).toEqual({ label: "Download (24.4 GB)", download: true })
    const here = version("Q4_K_P", { onThisComputer: Option.some(catalogId), downloadBytes: 0 })
    expect(confirmAction(withProjector, here, drafter)).toEqual({ label: "Add (900 MB)", download: false })
    expect(confirmAction(withProjector, here, Option.none())).toEqual({ label: "Add", download: false })
    expect(confirmAction(inspection({ source: { _tag: "Local", path: "/models/a.gguf" } }), version("a"), Option.none()).label).toBe("Add")
  })
  it("shows the repository link, lineage, license, and every version with its verdict", () => {
    const html = inDialog(<RepositoryView inspection={inspection({ versions: [...inspection().versions, version("Q6_K_P", { onThisComputer: Option.some(catalogId) })] })} models={models} selectedVersion={Option.some("Q4_K_P")} onSelectVersion={() => {}} onBack={() => {}} drafters={<p>drafter list</p>} />)
    expect(html).toContain(`href="https://huggingface.co/${repository}"`)
    expect(html).toContain("Variant of Qwen3.6 35B-A3B")
    expect(html).toContain("apache-2.0")
    expect(html).toContain("Too large for this machine")
    expect(html).toContain("Recommended")
    expect(html).toContain('Not supported yet<span class="sr-only">: Uses Q2_K weights, which Magnitude doesn&#x27;t support yet</span></button>')
    expect(html).toContain("Already on this computer")
    expect(html).toContain("drafter list")
    expect(html).toContain('aria-label="Back to search"')
    expect(html.match(/type="radio"[^>]*disabled=""/g)).toHaveLength(2)
    expect(html).toMatch(/type="radio"[^>]*checked=""/)
  })
  it("shows an unreadable version, which has no serving profile, as damaged and unselectable", () => {
    const damaged = version("Q3_K_M", { assessment: Option.some(unreadable) })
    expect(versionVerdict(damaged)).toMatchObject({ label: "File is damaged", detail: "The file is incomplete or damaged", selectable: false })
    const html = inDialog(<RepositoryView inspection={inspection({ versions: [version("Q4_K_P", { recommended: true }), damaged] })} models={models} selectedVersion={Option.some("Q4_K_P")} onSelectVersion={() => {}} />)
    expect(html).toContain('File is damaged<span class="sr-only">: The file is incomplete or damaged</span></button>')
    expect(html.match(/type="radio"[^>]*disabled=""/g)).toHaveLength(1)
  })
  it("explains why when no version runs on this computer", () => {
    const nothingRuns = inspection({ versions: [
      version("Q8_K_P", { assessment: Option.some(doesNotFit) }),
      version("Q2_K", { assessment: Option.some(unsupported({ _tag: "WeightTypes", types: [{ ggmlType: "Q2_K", tensorCount: 2 }] })) }),
    ] })
    const html = inDialog(<RepositoryView inspection={nothingRuns} models={models} selectedVersion={Option.none()} onSelectVersion={() => {}} />)
    expect(html).toContain("No version of this model runs on this computer")
    expect(html).toContain("Some versions are too large for this machine, and Magnitude can&#x27;t run the others: Uses Q2_K weights, which Magnitude doesn&#x27;t support yet.")
    const tooLarge = inDialog(<RepositoryView inspection={inspection({ versions: [version("Q8_K_P", { assessment: Option.some(doesNotFit) })] })} models={models} selectedVersion={Option.none()} onSelectVersion={() => {}} />)
    expect(tooLarge).toContain("Every version is too large for this machine.")
    expect(inDialog(<RepositoryView inspection={inspection()} models={models} selectedVersion={Option.some("Q4_K_P")} onSelectVersion={() => {}} />))
      .not.toContain("No version of this model runs on this computer")
  })
  it("explains an architecture the engine can't run in place of the versions", () => {
    const html = inDialog(<RepositoryView inspection={inspection({ displayName: "DeepSeek V4 Flash 0731", versions: [], unsupportedArchitecture: Option.some([{ _tag: "Architecture", architecture: "deepseek4" }]) })} models={models} selectedVersion={Option.none()} onSelectVersion={() => {}} />)
    expect(html).toContain("Magnitude can’t run this model")
    expect(html).toContain("No kernels for the deepseek4 architecture")
    expect(html).not.toContain('type="radio"')
  })
  describe("a pasted or linked version that can't run here", () => {
    const inspected = (overrides: Parameters<typeof inspection>[0], file: Option.Option<string>) =>
      inDialog(<InspectedRepository inspection={inspection(overrides)} file={file} models={models} onAdded={() => {}} />)
    const checkedRow = (html: string) => html.match(/<label data-selected="true"[^>]*>.*?<\/label>/)?.[0] ?? ""
    it("highlights a pasted owner/repo:QUANT that is too large, with no Download, and asks for another version", () => {
      const html = inspected({ selectedVersion: Option.some("Q8_K_P") }, Option.none())
      expect(checkedRow(html)).toContain("Q8_K_P")
      expect(checkedRow(html)).toMatch(/type="radio"[^>]*checked=""/)
      expect(checkedRow(html)).toContain("Too large for this machine")
      expect(html).toContain("Q8_K_P is too large for this machine")
      expect(html).toContain("Pick another version to download.")
      expect(html).not.toContain("Download (")
      expect(html).not.toContain("Pinned to the current version")
    })
    it("highlights a deep-linked unsupported file with its reason, with no Download", () => {
      const html = inspected({}, Option.some("Q2_K_P.gguf"))
      expect(checkedRow(html)).toContain("Q2_K_P")
      expect(html).toContain("Magnitude can’t run Q2_K_P")
      expect(html).toContain("Uses Q2_K weights, which Magnitude doesn&#x27;t support yet. Pick another version to download.")
      expect(html).not.toContain("Download (")
    })
    it("shows the nothing-runs footer when no version can run", () => {
      const html = inspected({ versions: [version("Q8_K_P", { assessment: Option.some(doesNotFit) })] }, Option.some("Q8_K_P.gguf"))
      expect(checkedRow(html)).toContain("Q8_K_P")
      expect(html).toContain("No version of this model runs on this computer")
      expect(html).not.toContain("Pick another version")
      expect(html).not.toContain("Download (")
    })
    it("words each unselectable verdict", () => {
      expect(unselectableVersionNotice(version("Q8_K_P", { assessment: Option.some(doesNotFit) }))).toEqual({ title: "Q8_K_P is too large for this machine", description: "Pick another version to download." })
      expect(unselectableVersionNotice(version("Q3_K_M", { assessment: Option.some(unreadable) })).description).toBe("The file is incomplete or damaged. Pick another version to download.")
    })
  })
  it("pins Hugging Face downloads and keeps local files in place", () => {
    const pinned = inDialog(<RepositoryFooter local={false} action={{ label: "Download (24.3 GB)", download: true }} pending={false} failure={undefined} onConfirm={() => {}} />)
    expect(pinned).toContain("Pinned to the current version of this repository")
    expect(pinned).toContain("Download (24.3 GB)")
    const local = inDialog(<RepositoryFooter local action={{ label: "Add", download: false }} pending failure={{ title: "Couldn’t add this model" }} onConfirm={() => {}} />)
    expect(local).toContain("never moves or deletes it")
    expect(local).toContain("Couldn’t add this model")
    expect(local).toMatch(/<button[^>]*disabled=""/)
  })
})

describe("add GGUF file", () => {
  const localFile = (overrides: Parameters<typeof inspection>[0] = {}) => inspection({
    source: { _tag: "Local", path: "/Users/me/models/qwen.gguf" }, license: Option.none(), versions: [version("Q4_K_P")], ...overrides,
  })
  it("imports the chosen file by its own path, never the folder the inspection read it from", () => {
    expect(addVersionRequest({ _tag: "File", path: "/Users/me/gguf-import/MiniCPM5-2B-Q4_K_M.gguf" }, version("Q4_K_M"), { _tag: "Auto" }))
      .toEqual({ _tag: "Import", request: { path: "/Users/me/gguf-import/MiniCPM5-2B-Q4_K_M.gguf", drafter: { _tag: "Auto" } } })
    expect(addVersionRequest({ _tag: "Repository", repository, commit: "c0ffee" }, version("Q4_K_P"), { _tag: "None" }))
      .toEqual({ _tag: "Add", request: { repository, revision: Option.some("c0ffee"), version: Option.some("Q4_K_P"), drafter: { _tag: "None" }, force: false } })
  })
  it("opens on the file picker, in place of the version list, until a file is chosen", () => {
    const html = inDialog(<LocalFileView browser={<p>file browser</p>} file={Option.none()} />)
    expect(html).toContain("Add a GGUF file")
    expect(html).toContain("never moves or deletes it")
    expect(html).toMatch(/aria-label="File".*file browser/)
    expect(html).toContain("Choose a GGUF file to see whether it runs on this machine.")
    expect(html).not.toContain('aria-label="Version"')
  })
  it("shows the file the native dialog chose, with Choose another file… in place of the in-app browser", () => {
    const opened: unknown[] = []
    const html = inDialog(<LocalFileView browser={<NativeFileChoice file={Option.some("C:\\Users\\me\\qwen.gguf")} open={request => { opened.push(request) }} onChosen={() => {}} />} file={Option.some(<p>chosen file</p>)} />)
    expect(html).toContain("C:\\Users\\me\\qwen.gguf")
    expect(html).toContain("Choose another file…")
    expect(html).not.toContain('aria-label="Folder path"')
    expect(ggufOpenRequest).toEqual({ kind: "gguf", title: "Add a GGUF file", defaultPath: Option.none() })
  })
  it("shows the chosen file below the picker", () => {
    const html = inDialog(<LocalFileView browser={<p>file browser</p>} file={Option.some(<p>chosen file</p>)} />)
    expect(html.indexOf("file browser")).toBeLessThan(html.indexOf("chosen file"))
    expect(html).not.toContain("Choose a GGUF file to see")
  })
  it("gives the chosen file its engine verdict, lineage, and size, then its drafters and Add", () => {
    const html = inDialog(<LocalFileSummary inspection={localFile()} models={models} drafters={<p>drafter list</p>} footer={<p>add footer</p>} />)
    expect(html).toContain("Qwen3.6 35B A3B Uncensored HauhauCS Aggressive")
    expect(html).toContain("Variant of Qwen3.6 35B-A3B")
    expect(html).toContain("23.4 GB")
    expect(html).toContain("Fits your machine")
    expect(html.indexOf("drafter list")).toBeLessThan(html.indexOf("add footer"))
    expect(html).not.toContain('type="radio"')
  })
  it("explains a file the engine can't run or that doesn't fit, with nothing to add", () => {
    const tooLarge = inDialog(<LocalFileSummary inspection={localFile({ versions: [version("Q8_K_P", { assessment: Option.some(doesNotFit) })] })} models={models} />)
    expect(tooLarge).toContain("Too large for this machine")
    const unsupportedFile = inDialog(<LocalFileSummary inspection={localFile({ unsupportedArchitecture: Option.some([{ _tag: "Architecture", architecture: "deepseek4" }]) })} models={models} />)
    expect(unsupportedFile).toContain("Magnitude can’t run this model")
    expect(unsupportedFile).toContain("No kernels for the deepseek4 architecture")
    expect(unsupportedFile).not.toContain("Fits your machine")
  })
})
