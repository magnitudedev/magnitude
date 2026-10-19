import { useEffect, useId, useState, type ReactNode } from "react"
import { Cause, Exit, Option } from "effect"
import { Result, useAtomRefresh, useAtomSet, useAtomValue } from "@effect-atom/atom-react"
import { ArrowLeftIcon, ArrowUpRightIcon, DownloadSimpleIcon, LinkIcon, MagnifyingGlassIcon, PlusIcon } from "@phosphor-icons/react"
import {
  candidateDownloadBytes, catalogVariantOfName, formatStorageSize, noRunnableVersionReason, GATED_SEARCH_RESULT_LABEL, searchResultUnsupportedMessage, unsupportedReasonMessages, useAgentClient,
  type AddModelRequest,
} from "@magnitudedev/client-common"
import type { CandidateInspection, CandidateInspectionPhase, CandidateInspectionProgress, CandidateInspectionUpdate, CandidateVersion, CandidateVersionProgress, DrafterCandidate, DrafterChoice, HuggingFaceRepositoryId, HuggingFaceSearchResponse, HuggingFaceSearchResult, LocalModel, LocalModelAssessment, UnsupportedReason } from "@magnitudedev/sdk"
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from "./ui/dialog"
import { Button } from "./ui/button"
import { Input } from "./ui/input"
import { ErrorNotice, NoticeAction, type NoticeContent } from "./error-notice"
import { SkeletonLine } from "./page-skeletons"
import { HuggingFaceLogo, NamedLogo, variantLab } from "./model-logo"
import { DrafterList, drafterChoice, selectedDrafterCandidate, useDrafterCandidates, type DrafterSelection } from "./drafter-list"
import { candidateErrorNotice, type CandidateNotice } from "./candidate-errors"
import { CandidateErrorNotice } from "./hugging-face"
import { DirectoryBrowser, useDirectoryBrowser, useNativeOpenPath, type NativeOpenPath } from "./folder-picker"
import { ActionTooltip, TooltipProvider } from "./ui/tooltip"

/** What the dialog opens on: the app's own requests, plus Add GGUF file…, with the file the native dialog chose or none to pick in-app. */
export type AddDialogRequest = AddModelRequest | { readonly _tag: "LocalFile"; readonly file: Option.Option<string> }

type Step =
  | { readonly _tag: "Find"; readonly tab: FindTab; readonly query: string }
  | { readonly _tag: "Repository"; readonly reference: string; readonly file: Option.Option<string>; readonly preview: RepositoryPreview }
  | { readonly _tag: "LocalFile"; readonly file: Option.Option<string> }

export type FindTab = "search" | "paste"

const initialStep = (request: AddDialogRequest): Step => {
  switch (request._tag) {
    case "Search": return { _tag: "Find", tab: "search", query: request.query }
    case "HuggingFace": return { _tag: "Repository", reference: request.repository, file: request.file, preview: referencePreview(request.repository) }
    case "LocalFile": return { _tag: "LocalFile", file: request.file }
  }
}

const huggingFaceUrl = /^https?:\/\/(www\.)?huggingface\.co\/[\w.-]+\/[\w.-]+(\/\S*)?$/
const repositoryReference = /^[\w.-]+\/[\w.-]+(:[\w.-]+)?$/

/** A pasted Hugging Face URL, `owner/repo`, or `owner/repo:QUANT`; anything else is refused before asking the service. */
export const parseModelReference = (text: string): Option.Option<string> => {
  const value = text.trim()
  return huggingFaceUrl.test(value) || repositoryReference.test(value) ? Option.some(value) : Option.none()
}

/** What the repository step shows of a repository before its inspection arrives: from the search result, or read from the reference. */
export type RepositoryPreview = {
  readonly displayName: string
  readonly repository: Option.Option<string>
  readonly variantOf: HuggingFaceSearchResult["variantOf"]
  readonly versionCount: HuggingFaceSearchResult["versionCount"]
}

export const searchResultPreview = (result: HuggingFaceSearchResult): RepositoryPreview =>
  ({ displayName: result.displayName, repository: Option.some(result.repository), variantOf: result.variantOf, versionCount: result.versionCount })

/** A pasted link or reference names its repository; anything unrecognized is shown as typed. */
export const referencePreview = (reference: string): RepositoryPreview => {
  const repository = Option.fromNullable(/^(?:https?:\/\/(?:www\.)?huggingface\.co\/)?([\w.-]+\/[\w.-]+)/.exec(reference.trim())?.[1])
  return { displayName: Option.match(repository, { onNone: () => reference, onSome: id => id.slice(id.indexOf("/") + 1) }), repository, variantOf: Option.none(), versionCount: Option.none() }
}

const downloadsFormat = new Intl.NumberFormat(undefined, { notation: "compact", maximumFractionDigits: 1 })
export const formatDownloads = (count: number) => `${downloadsFormat.format(count)} ${count === 1 ? "download" : "downloads"}`

/** The Add from Hugging Face dialog: Find (Search or Paste a link), then the repository step. */
export function AddModelDialog({ request, models, onClose, onAdded }: {
  request: Option.Option<AddDialogRequest>
  models: readonly LocalModel[]
  onClose: () => void
  onAdded: () => void
}) {
  return <Dialog open={Option.isSome(request)} onOpenChange={open => { if (!open) onClose() }}>
    <DialogContent className="flex h-[min(720px,85vh)] flex-col overflow-hidden sm:max-w-xl">
      {Option.isSome(request) && <AddModelFlow key={JSON.stringify(request.value)} request={request.value} models={models} onAdded={onAdded} />}
    </DialogContent>
  </Dialog>
}

function AddModelFlow({ request, models, onAdded }: { request: AddDialogRequest; models: readonly LocalModel[]; onAdded: () => void }) {
  const [step, setStep] = useState(() => initialStep(request))
  const [find, setFind] = useState<{ tab: FindTab; query: string }>(() => step._tag === "Find" ? { tab: step.tab, query: step.query } : { tab: "search", query: "" })
  if (step._tag === "LocalFile") return <LocalFileStep initialFile={step.file} models={models} onAdded={onAdded} />
  if (step._tag === "Find") return <FindStep tab={find.tab} query={find.query} models={models}
    onChange={next => setFind(next)} onOpen={(reference, preview) => setStep({ _tag: "Repository", reference, file: Option.none(), preview })} />
  return <RepositoryStep reference={step.reference} preview={step.preview} file={step.file} models={models} onAdded={onAdded}
    onBack={() => setStep({ _tag: "Find", tab: find.tab, query: find.query })} />
}

/**
 * The dialog's middle, between its header and footer. The dialog has one fixed height for every step and
 * state, so whatever loads in here scrolls instead of resizing it.
 */
export function DialogBody({ children }: { children: ReactNode }) {
  return <div data-slot="dialog-body" className="-mx-4 flex min-h-0 flex-1 flex-col gap-4 overflow-y-auto px-4 pb-px">{children}</div>
}

/** A step's footer: held at the bottom of the dialog body whether the content above is short or scrolls beneath it. */
function StepFooter({ children }: { children: ReactNode }) {
  return <div data-slot="step-footer" className="sticky bottom-0 z-10 -mx-4 mt-auto space-y-3 border-t border-slate-200 bg-white px-4 pt-4 dark:border-slate-750 dark:bg-slate-850">{children}</div>
}

/** The footer's row: a note or status on the left, in a slot of fixed height, and the step's action on the right. */
function StepActions({ status, action }: { status: ReactNode; action: ReactNode }) {
  return <DialogFooter className="items-center sm:justify-between">
    <div className="flex min-h-6 w-full min-w-0 flex-col justify-center sm:w-auto sm:flex-1">{status}</div>
    {action}
  </DialogFooter>
}

/** The search's observed state, as the find step renders it. */
export type SearchState =
  | { readonly _tag: "Idle" }
  | { readonly _tag: "Loading" }
  | { readonly _tag: "Failed"; readonly notice: NoticeContent }
  | { readonly _tag: "Loaded"; readonly response: HuggingFaceSearchResponse }

function FindStep({ tab, query, models, onChange, onOpen }: {
  tab: FindTab
  query: string
  models: readonly LocalModel[]
  onChange: (next: { tab: FindTab; query: string }) => void
  onOpen: (reference: string, preview: RepositoryPreview) => void
}) {
  const [typed, setTyped] = useState(query)
  const [pasted, setPasted] = useState("")
  const [pasteInvalid, setPasteInvalid] = useState(false)
  return <FindStepView tab={tab} onTab={next => onChange({ tab: next, query })}
    typed={typed} onType={setTyped} onSearch={() => onChange({ tab, query: typed.trim() })}
    results={query.trim() ? <SearchResults query={query.trim()} models={models} onOpen={onOpen} /> : null}
    pasted={pasted} pasteInvalid={pasteInvalid} onPaste={value => { setPasted(value); setPasteInvalid(false) }}
    onOpenPasted={() => Option.match(parseModelReference(pasted), { onNone: () => setPasteInvalid(true), onSome: reference => onOpen(reference, referencePreview(reference)) })} />
}

function SearchResults({ query, models, onOpen }: { query: string; models: readonly LocalModel[]; onOpen: (reference: string, preview: RepositoryPreview) => void }) {
  const client = useAgentClient()
  const search = client.Models.SearchHuggingFaceModels({ query, limit: Option.none() })
  const result = useAtomValue(search).result
  const retry = useAtomRefresh(search)
  const state: SearchState = Result.isSuccess(result) ? { _tag: "Loaded", response: result.value }
    : Result.isFailure(result) ? { _tag: "Failed", notice: Option.match(Cause.failureOption(result.cause), { onNone: () => ({ title: "Couldn’t search Hugging Face", description: "Try again." }), onSome: candidateErrorNotice }) }
    : { _tag: "Loading" }
  return <SearchResultsView state={state} models={models} onOpen={result => onOpen(result.repository, searchResultPreview(result))} onRetry={retry} />
}

/** The find step: Search and Paste a link. */
export function FindStepView({ tab, onTab, typed, onType, onSearch, results, pasted, pasteInvalid, onPaste, onOpenPasted }: {
  tab: FindTab
  onTab: (tab: FindTab) => void
  typed: string
  onType: (value: string) => void
  onSearch: () => void
  results: ReactNode
  pasted: string
  pasteInvalid: boolean
  onPaste: (value: string) => void
  onOpenPasted: () => void
}) {
  const tabButton = (value: FindTab, label: string, icon: ReactNode) =>
    <Button role="tab" aria-selected={tab === value} variant={tab === value ? "secondary" : "ghost"} onClick={() => onTab(value)}>{icon}{label}</Button>
  return <>
    <DialogHeader>
      <DialogTitle className="flex items-center gap-2"><HuggingFaceLogo className="size-5" />Add a model from Hugging Face</DialogTitle>
      <DialogDescription>Any GGUF from a model family Magnitude supports: other quantizations, finetunes, and uncensored versions.</DialogDescription>
    </DialogHeader>
    <div role="tablist" aria-label="Find a model" className="flex items-center gap-1">
      {tabButton("search", "Search", <MagnifyingGlassIcon />)}
      {tabButton("paste", "Paste a link", <LinkIcon />)}
    </div>
    {tab === "search"
      ? <div role="tabpanel" aria-label="Search" className="flex min-h-0 flex-1 flex-col gap-3">
        <form onSubmit={event => { event.preventDefault(); onSearch() }}>
          <Input aria-label="Search Hugging Face" autoFocus placeholder="Search Hugging Face, for example qwen3.6 35b abliterated" value={typed} onChange={event => onType(event.target.value)} />
        </form>
        <DialogBody>{results ?? <p className="py-6 text-center text-sm text-slate-500">Search by model name, uploader, or variant. Press Enter to search.</p>}</DialogBody>
      </div>
      : <div role="tabpanel" aria-label="Paste a link">
        <form className="flex gap-2" onSubmit={event => { event.preventDefault(); onOpenPasted() }}>
          <Input aria-label="Hugging Face link" autoFocus aria-invalid={pasteInvalid || undefined} placeholder="https://huggingface.co/owner/repo" value={pasted} onChange={event => onPaste(event.target.value)} className="min-w-0 flex-1" />
          <Button type="submit" disabled={!pasted.trim()}>Open</Button>
        </form>
        <p className={`mt-2 text-xs ${pasteInvalid ? "text-red-600 dark:text-red-400" : "text-slate-500"}`}>{pasteInvalid ? "That isn’t a Hugging Face link. " : ""}Accepts a Hugging Face link, owner/repo, or owner/repo:QUANT.</p>
      </div>}
  </>
}

/** Repositories with uploader, the catalog model each is a variant of, and downloads; those Magnitude can't run are shown disabled with why. */
export function SearchResultsView({ state, models, onOpen, onRetry }: { state: SearchState; models: readonly LocalModel[]; onOpen: (result: HuggingFaceSearchResult) => void; onRetry: () => void }) {
  switch (state._tag) {
    case "Idle": return null
    case "Loading": return <div aria-busy="true" aria-label="Searching Hugging Face" className="space-y-2 py-1">{[0, 1, 2].map(index => <SkeletonLine key={index} className="h-10" width={index % 2 ? "80%" : "95%"} />)}</div>
    case "Failed": return <ErrorNotice {...state.notice} actions={<NoticeAction onClick={onRetry}>Try again</NoticeAction>} />
    case "Loaded": {
      const { results } = state.response
      return results.length === 0
        ? <p className="py-6 text-center text-sm text-slate-500">No GGUF repositories match this search.</p>
        : <ul aria-label="Search results" className="divide-y divide-slate-200 overflow-hidden rounded-lg border border-slate-200 dark:divide-slate-750 dark:border-slate-750">
          {results.map(result => <li key={result.repository}><SearchResultRow result={result} models={models} onOpen={() => onOpen(result)} /></li>)}
        </ul>
    }
  }
}

function SearchResultRow({ result, models, onOpen }: { result: HuggingFaceSearchResult; models: readonly LocalModel[]; onOpen: () => void }) {
  const variantOf = catalogVariantOfName(models, result.variantOf)
  const unsupported = Option.map(result.unsupported, searchResultUnsupportedMessage)
  const detail = Option.match(unsupported, {
    onSome: reason => reason,
    onNone: () => [result.uploader, ...(Option.isSome(variantOf) ? [`Variant of ${variantOf.value}`] : []), ...(result.gated ? [GATED_SEARCH_RESULT_LABEL] : [])].join(" · "),
  })
  return <button type="button" onClick={onOpen} disabled={Option.isSome(unsupported)} className="flex w-full items-center gap-3 px-3 py-2.5 text-left hover:bg-slate-50 focus-visible:outline-2 focus-visible:outline-blue-500 disabled:cursor-not-allowed disabled:opacity-60 disabled:hover:bg-transparent dark:hover:bg-slate-800 dark:disabled:hover:bg-transparent">
    <NamedLogo lab={variantLab(result.variantOf, result.displayName)} name={result.displayName} className="size-6" />
    <span className="min-w-0 flex-1">
      <span className="block truncate text-sm font-medium" title={result.repository}>{result.displayName}</span>
      <span className="block truncate text-xs text-slate-500">{detail}</span>
    </span>
    <span className="shrink-0 whitespace-nowrap text-xs tabular-nums text-slate-500">{formatDownloads(result.downloads)}</span>
  </button>
}

/** What an inspection shows: still reading (including while Retry runs), its progress, the failure with Retry, or the result. */
export type InspectionView =
  | { readonly _tag: "Reading" }
  | { readonly _tag: "Inspecting"; readonly progress: CandidateInspectionProgress }
  | { readonly _tag: "Failed"; readonly error: Option.Option<unknown> }
  | { readonly _tag: "Inspected"; readonly inspection: CandidateInspection }

export const inspectionView = (result: Result.Result<CandidateInspectionUpdate, unknown>): InspectionView =>
  Result.isFailure(result) ? result.waiting ? { _tag: "Reading" } : { _tag: "Failed", error: Cause.failureOption(result.cause) }
    : Result.isSuccess(result) ? result.value._tag === "Inspected" ? { _tag: "Inspected", inspection: result.value.inspection } : { _tag: "Inspecting", progress: result.value.progress }
    : { _tag: "Reading" }

/**
 * Inspects a reference as it goes; an inspection already made shows at once. A failure remembered
 * from an earlier opening is read again rather than shown stale.
 */
function useInspection(reference: string) {
  const client = useAgentClient()
  const inspect = client.Models.StreamModelCandidateInspection({ reference })
  const result = useAtomValue(inspect).result
  const retry = useAtomRefresh(inspect)
  useEffect(() => { if (Result.isFailure(result) && !result.waiting) retry() }, [inspect])
  return { view: inspectionView(result), retry }
}

function RepositoryStep({ reference, preview, file, models, onBack, onAdded }: { reference: string; preview: RepositoryPreview; file: Option.Option<string>; models: readonly LocalModel[]; onBack?: () => void; onAdded: () => void }) {
  const { view, retry } = useInspection(reference)
  if (view._tag === "Inspected") return <InspectedRepository inspection={view.inspection} file={file} models={models} onAdded={onAdded} {...(onBack ? { onBack } : {})} />
  const variantOf = catalogVariantOfName(models, preview.variantOf)
  const header = <RepositoryHeader displayName={preview.displayName} lab={variantLab(preview.variantOf, preview.displayName)} onBack={onBack}
    source={Option.match(preview.repository, { onNone: () => <span className="font-mono">{reference}</span>, onSome: repositoryLink })}
    facts={Option.toArray(Option.map(variantOf, name => `Variant of ${name}`))} />
  if (view._tag === "Failed") return <>{header}<DialogBody><InspectionFailure error={view.error} onRetry={retry} /></DialogBody></>
  return <InspectingRepositoryView header={header} placeholderRows={placeholderVersionRows(preview.versionCount)} progress={view._tag === "Inspecting" ? Option.some(view.progress) : Option.none()} />
}

/** How many version rows stand in for a repository's versions until its files are known, without a search result's count. */
export const PLACEHOLDER_VERSION_ROWS = 6
const MAX_PLACEHOLDER_VERSION_ROWS = 12

/** The placeholder rows for a repository the search counted `versionCount` versions in, at most a dialog's worth. */
export const placeholderVersionRows = (versionCount: Option.Option<number>): number =>
  Option.match(versionCount, { onNone: () => PLACEHOLDER_VERSION_ROWS, onSome: count => Math.min(count, MAX_PLACEHOLDER_VERSION_ROWS) })

/**
 * The repository step while it is inspected, laid out exactly as the inspected step: the header, the
 * version rows (placeholders until the files are known, then each filling in its verdict), and the
 * footer, where the status line and progress bar stand in for the pinned-version note.
 */
export function InspectingRepositoryView({ header, placeholderRows = PLACEHOLDER_VERSION_ROWS, progress }: { header: ReactNode; placeholderRows?: number; progress: Option.Option<CandidateInspectionProgress> }) {
  const versions = Option.filter(Option.map(progress, ({ versions }) => versions), versions => versions.length > 0)
  return <>
    {header}
    <DialogBody>
      <section aria-label="Version" aria-busy="true">
        <h3 className="mb-2 font-heading text-sm font-semibold">Version</h3>
        {Option.match(versions, {
          onNone: () => <div aria-label="Reading the repository" className={versionList}>
            {Array.from({ length: placeholderRows }, (_, index) => <VersionRowPlaceholder key={index} />)}
          </div>,
          onSome: versions => <InspectionVersionList versions={versions} />,
        })}
      </section>
      <StepFooter><StepActions status={<InspectionStatus progress={progress} reading="Reading the repository…" />} action={<Button className="shrink-0" disabled><DownloadSimpleIcon />Download</Button>} /></StepFooter>
    </DialogBody>
  </>
}

/** The status line for an inspection's phase, with how much of it is done. */
export const inspectionPhaseText = (phase: CandidateInspectionPhase): string => {
  const counted = (text: string) => phase.total > 0 ? `${text} · ${phase.completed} of ${phase.total}` : text
  switch (phase._tag) {
    case "ReadingHeaders": return counted("Reading file headers")
    case "CheckingVersions": return counted("Checking versions on this computer")
    case "CheckingDrafters": return counted(`Checking drafters for ${phase.version}`)
  }
}

/**
 * The inspection's status line over a thin progress bar. Its place is held from the first frame; the
 * text itself fades in after a moment, so an inspection that finishes at once (its results cached) never flashes it.
 */
export function InspectionStatus({ progress, reading }: { progress: Option.Option<CandidateInspectionProgress>; reading: string }) {
  const percent = Option.match(progress, { onNone: () => 0, onSome: ({ phase }) => phase.total > 0 ? Math.round(100 * phase.completed / phase.total) : 0 })
  const text = Option.match(progress, { onNone: () => reading, onSome: ({ phase }) => inspectionPhaseText(phase) })
  return <div className="space-y-1.5 animate-[fade-in_150ms_ease-out_400ms_both]">
    <p role="status" className="truncate text-xs tabular-nums text-slate-500">{text}</p>
    <div role="progressbar" aria-label="Inspection progress" aria-valuemin={0} aria-valuemax={100} aria-valuenow={percent} aria-valuetext={text} className="h-1 w-full overflow-hidden rounded-full bg-slate-100 dark:bg-slate-800">
      <div className="h-full rounded-full bg-blue-600 transition-[width] duration-300 dark:bg-blue-500" style={{ width: `${percent}%` }} />
    </div>
  </div>
}

const versionList = "divide-y divide-slate-200 overflow-hidden rounded-lg border border-slate-200 dark:divide-slate-750 dark:border-slate-750"

/** One version row's frame, shared by placeholders, rows being checked and the inspected rows, so a row keeps its height as it fills in. */
const versionRow = "flex min-h-10 items-center gap-3 px-3 py-2 text-sm"
const versionLabelColumn = "w-20 shrink-0 truncate font-medium sm:w-28"
const versionSizeColumn = "w-14 shrink-0 tabular-nums text-slate-500 sm:w-20"
const versionVerdictColumn = "flex min-w-0 flex-1 items-center justify-end gap-x-2 whitespace-nowrap text-right text-xs *:min-w-0 *:truncate"

/** Where a row's radio button goes before the version can be picked. */
function RadioPlaceholder() {
  return <span aria-hidden="true" className="size-[13px] shrink-0 rounded-full border border-slate-300 dark:border-slate-600" />
}

function VersionRowPlaceholder() {
  return <div aria-hidden="true" data-slot="version-placeholder" className={versionRow}>
    <RadioPlaceholder />
    <span className={versionLabelColumn}><SkeletonLine className="h-5" width="75%" /></span>
    <span className={versionSizeColumn}><SkeletonLine className="h-5" width="80%" /></span>
    <span className={versionVerdictColumn}><SkeletonLine className="h-5" width="6rem" /></span>
  </div>
}

/** The repository's versions while it is inspected: each row fills in its verdict once that version is checked. */
export function InspectionVersionList({ versions }: { versions: readonly CandidateVersionProgress[] }) {
  return <ul aria-label="Versions" className={versionList}>
    {versions.map((version, index) => <li key={`${index}:${version.label}`} className={versionRow}>
      <RadioPlaceholder />
      <span className={versionLabelColumn}>{version.label}</span>
      <span className={versionSizeColumn}>{formatStorageSize(version.sizeBytes)}</span>
      <span className={versionVerdictColumn}>
        {Option.match(version.assessment, {
          onNone: () => <span className="inline-flex items-center gap-1.5 text-slate-400"><span aria-hidden="true" className="size-1.5 rounded-full bg-current motion-safe:animate-pulse" />Checking…</span>,
          onSome: assessment => <VerdictLabel verdict={assessmentVerdict(assessment, false)} />,
        })}
      </span>
    </li>)}
  </ul>
}

/** The repository step's header: back, logo, name, and one line with the repository link and its facts. */
function RepositoryHeader({ displayName, lab, source, facts, onBack }: { displayName: string; lab: ReturnType<typeof variantLab>; source: ReactNode; facts: readonly string[]; onBack: (() => void) | undefined }) {
  return <DialogHeader>
    <div className="flex items-start gap-3 pr-8">
      {onBack && <BackButton onBack={onBack} />}
      <NamedLogo lab={lab} name={displayName} className="mt-0.5 size-7" />
      <div className="min-w-0">
        <DialogTitle className="truncate">{displayName}</DialogTitle>
        <DialogDescription className="truncate">
          {source}
          {facts.map(fact => <span key={fact}><span aria-hidden="true"> · </span>{fact}</span>)}
        </DialogDescription>
      </div>
    </div>
  </DialogHeader>
}

const repositoryLink = (repository: string) =>
  <a href={`https://huggingface.co/${repository}`} target="_blank" rel="noreferrer" className="hover:underline">{repository}<ArrowUpRightIcon aria-hidden="true" className="ml-0.5 inline size-3 align-[-1px]" /></a>

function InspectionFailure({ error, onRetry }: { error: Option.Option<unknown>; onRetry: () => void }) {
  const notice = Option.match(error, { onNone: (): CandidateNotice => ({ title: "Couldn’t read this model", description: "Try again.", retryable: true }), onSome: candidateErrorNotice })
  return <CandidateErrorNotice notice={notice} onRetry={onRetry} />
}

function BackButton({ onBack }: { onBack: () => void }) {
  return <Button variant="ghost" size="icon-sm" aria-label="Back to search" onClick={onBack}><ArrowLeftIcon /></Button>
}

/** The short verdict for versions the engine can't run: damaged, an architecture without kernels, or anything else not supported yet. */
export const unsupportedVersionLabel = (reasons: readonly UnsupportedReason[]): string =>
  reasons.some(reason => reason._tag === "Unreadable") ? "Can't read this file"
    : reasons.some(reason => reason._tag === "Architecture") ? "Not supported"
    : "Not supported yet"

type Verdict = { readonly label: string; readonly detail?: string; readonly selectable: boolean; readonly recommended: boolean }

/** The engine's verdict on a version; `detail` is the full reason behind a short label. */
export const assessmentVerdict = (assessment: LocalModelAssessment, recommended: boolean): Verdict => {
  switch (assessment._tag) {
    case "Unsupported": return { label: unsupportedVersionLabel(assessment.reasons), detail: unsupportedReasonMessages(assessment.reasons).join(". "), selectable: false, recommended: false }
    case "DoesNotFit": return { label: "Too large for this machine", selectable: false, recommended: false }
    case "Fits": return recommended
      ? { label: "Recommended", selectable: true, recommended: true }
      : { label: "Fits your machine", selectable: true, recommended: false }
  }
}

/** A version's engine verdict, never read from its file name. */
export const versionVerdict = (version: CandidateVersion): Verdict => Option.match(version.assessment, {
  onNone: (): Verdict => ({ label: "Not checked yet", selectable: true, recommended: false }),
  onSome: assessment => assessmentVerdict(assessment, version.recommended),
})

/** The version shown first: a linked file, the pasted QUANT, the recommended one, then the first that can run. */
export const initialVersion = (inspection: CandidateInspection, file: Option.Option<string>): Option.Option<string> => {
  const linked = Option.flatMap(file, path => Option.fromNullable(inspection.versions.find(version => version.files.includes(path))))
  if (Option.isSome(linked)) return Option.some(linked.value.label)
  if (Option.isSome(inspection.selectedVersion)) return inspection.selectedVersion
  const preferred = inspection.versions.find(version => version.recommended) ?? inspection.versions.find(version => versionVerdict(version).selectable)
  return Option.fromNullable(preferred?.label)
}

/** Download names everything fetched: the version's target and projector bytes still to download, plus the chosen drafter. */
export const confirmAction = (inspection: CandidateInspection, version: CandidateVersion, drafter: Option.Option<DrafterCandidate>): { readonly label: string; readonly download: boolean } => {
  const { totalBytes, drafterBytes } = candidateDownloadBytes(version, inspection.projector, drafter)
  if (inspection.source._tag === "Local" || totalBytes === drafterBytes) {
    return { label: drafterBytes > 0 ? `Add (${formatStorageSize(drafterBytes)})` : "Add", download: false }
  }
  return { label: `Download (${formatStorageSize(totalBytes)})`, download: true }
}

/** The inspected repository: a version that can run gets its drafters and Download; one that can't gets the reason in their place. */
export function InspectedRepository({ inspection, file, models, onBack, onAdded }: { inspection: CandidateInspection; file: Option.Option<string>; models: readonly LocalModel[]; onBack?: () => void; onAdded: () => void }) {
  const [picked, setPicked] = useState(() => initialVersion(inspection, file))
  // The version the dialog opened on is scrolled into view until the user picks one.
  const [revealed, setRevealed] = useState(picked)
  const version = Option.flatMap(picked, label => Option.fromNullable(inspection.versions.find(entry => entry.label === label)))
  const onSelectVersion = (label: string) => { setPicked(Option.some(label)); setRevealed(Option.none()) }
  const header = { inspection, models, selectedVersion: picked, revealedVersion: revealed, onSelectVersion, ...(onBack ? { onBack } : {}) }
  const source = inspection.source
  if (Option.isNone(version) || !versionVerdict(version.value).selectable || Option.isSome(inspection.unsupportedArchitecture) || source._tag !== "HuggingFace") return <RepositoryView {...header} />
  return <VersionWithDrafters key={version.value.label} version={version.value} header={header} onAdded={onAdded}
    target={{ _tag: "Repository", repository: source.repository, commit: source.commit }} />
}

function VersionWithDrafters({ version, header, target, onAdded }: {
  version: CandidateVersion
  header: Omit<Parameters<typeof RepositoryView>[0], "drafters" | "footer">
  target: AddTarget
  onAdded: () => void
}) {
  const { drafters, footer } = useVersionConfirmation(header.inspection, version, target, onAdded)
  return <RepositoryView {...header} drafters={drafters} footer={footer} />
}

/** What adding a version names: a Hugging Face repository pinned to the inspected commit, or the GGUF file the user chose. */
export type AddTarget =
  | { readonly _tag: "Repository"; readonly repository: HuggingFaceRepositoryId; readonly commit: string }
  | { readonly _tag: "File"; readonly path: string }

/** The command that adds a version: a Hugging Face repository is added at its version; a chosen file is imported by its own path. */
export const addVersionRequest = (target: AddTarget, version: CandidateVersion, drafter: DrafterChoice) =>
  target._tag === "File"
    ? { _tag: "Import" as const, request: { path: target.path, drafter } }
    : { _tag: "Add" as const, request: { repository: target.repository, revision: Option.some(target.commit), version: Option.some(version.label), drafter, force: false } }

/**
 * This confirmation's own Add or Import: the mutation's state is shared by every opening of the
 * dialog, so a pending or failed add from an earlier one shows nothing here until this one submits.
 */
export const submissionState = (submitted: boolean, state: Result.Result<unknown, unknown>): { readonly pending: boolean; readonly failure: CandidateNotice | undefined } => ({
  pending: submitted && state.waiting,
  failure: submitted && Result.isFailure(state) && !state.waiting
    ? Option.match(Cause.failureOption(state.cause), { onNone: (): CandidateNotice => ({ title: "Couldn’t add this model", description: "Try again.", retryable: true }), onSome: candidateErrorNotice })
    : undefined,
})

/** The drafter choices for one version and the footer that adds it with the chosen drafter. */
function useVersionConfirmation(inspection: CandidateInspection, version: CandidateVersion, target: AddTarget, onAdded: () => void) {
  const client = useAgentClient()
  const { candidates, otherPending, otherFailure, checkOther } = useDrafterCandidates({ _tag: "Candidate", source: inspection.source, files: version.files })
  const [selection, setSelection] = useState<DrafterSelection>({ _tag: "Auto" })
  const list = Result.isSuccess(candidates) ? candidates.value.candidates : []
  const findingDrafters = !Result.isSuccess(candidates) && !Result.isFailure(candidates)
  const add = useAtomSet(client.Models.AddHuggingFaceModel, { mode: "promiseExit" })
  const adding = useAtomValue(client.Models.AddHuggingFaceModel)
  const importFile = useAtomSet(client.Models.ImportModelFile, { mode: "promiseExit" })
  const importing = useAtomValue(client.Models.ImportModelFile)
  const [submitted, setSubmitted] = useState(false)
  const { pending, failure } = submissionState(submitted, target._tag === "File" ? importing : adding)
  const confirm = () => {
    setSubmitted(true)
    const done = (exit: Exit.Exit<unknown, unknown>) => { if (Exit.isSuccess(exit)) onAdded() }
    const command = addVersionRequest(target, version, drafterChoice(selection, list))
    if (command._tag === "Import") void importFile(command.request).then(done)
    else void add(command.request).then(done)
  }
  return {
    drafters: Result.isFailure(candidates) ? <ErrorNotice severity="warning" title="Couldn’t list drafters" description="The recommended drafter is still used when you add this model." />
      : findingDrafters ? <div aria-busy="true" aria-label="Finding drafters" className="space-y-2"><p className="text-xs text-slate-500">Finding drafters for this version…</p><SkeletonLine className="h-10" width="100%" /><SkeletonLine className="h-10" width="100%" /></div>
      : <DrafterList candidates={list} selection={selection} onSelect={setSelection} otherPending={otherPending} otherFailure={otherFailure} onCheckOther={checkOther} />,
    footer: <RepositoryFooter local={target._tag === "File"} action={confirmAction(inspection, version, selectedDrafterCandidate(selection, list))} pending={pending || findingDrafters} failure={failure} onConfirm={confirm} />,
  }
}

const localFileNote = "Magnitude uses this file where it is and never moves or deletes it."

export const ggufOpenRequest = { kind: "gguf", title: "Add a GGUF file", defaultPath: Option.none() } as const

/** Add GGUF file…: the repository step with the chosen file in place of the version list; the native dialog chooses it when the window has one. */
function LocalFileStep({ initialFile, models, onAdded }: { initialFile: Option.Option<string>; models: readonly LocalModel[]; onAdded: () => void }) {
  const native = useNativeOpenPath()
  const [file, setFile] = useState(initialFile)
  const inspection = Option.map(file, path => <LocalFileInspection key={path} path={path} models={models} onAdded={onAdded} />)
  return Option.match(native, {
    onSome: open => <LocalFileView browser={<NativeFileChoice file={file} open={open} onChosen={path => setFile(Option.some(path))} />} file={inspection} />,
    onNone: () => <BrowsedLocalFile file={file} onChosen={path => setFile(Option.some(path))} inspection={inspection} />,
  })
}

function BrowsedLocalFile({ file, onChosen, inspection }: { file: Option.Option<string>; onChosen: (path: string) => void; inspection: Option.Option<ReactNode> }) {
  const browser = useDirectoryBrowser("gguf", Option.none())
  return <LocalFileView browser={<DirectoryBrowser browser={browser} selectedFile={file} onSelectFile={onChosen} />} file={inspection} />
}

/** The file the native dialog chose, with a button that opens it again. */
export function NativeFileChoice({ file, open, onChosen }: { file: Option.Option<string>; open: NativeOpenPath; onChosen: (path: string) => void }) {
  return <div className="flex items-center gap-2">
    <span className="min-w-0 flex-1 truncate font-mono text-xs text-slate-600 dark:text-slate-300" title={Option.getOrUndefined(file)}>{Option.getOrElse(file, () => "No file chosen")}</span>
    <Button size="sm" variant="outline" className="shrink-0" onClick={() => open(ggufOpenRequest, onChosen)}>{Option.isSome(file) ? "Choose another file…" : "Choose a file…"}</Button>
  </div>
}

/** The Add GGUF file step: the file picker on top, then the chosen file's verdict, drafters, and Add. */
export function LocalFileView({ browser, file }: { browser: ReactNode; file: Option.Option<ReactNode> }) {
  return <>
    <DialogHeader>
      <DialogTitle>Add a GGUF file</DialogTitle>
      <DialogDescription>A GGUF file on the computer running Magnitude. {localFileNote}</DialogDescription>
    </DialogHeader>
    <DialogBody>
      <section aria-label="File" className="space-y-2">
        <h3 className="font-heading text-sm font-semibold">File</h3>
        {browser}
      </section>
      {Option.getOrElse(file, () => <>
        <p className="text-sm text-slate-500">Choose a GGUF file to see whether it runs on this machine.</p>
        <LocalFilePendingFooter status={<p className="text-xs text-slate-500">{localFileNote}</p>} />
      </>)}
    </DialogBody>
  </>
}

/** The local file footer before the file can be added: a note or the inspection's status, beside Add disabled. */
function LocalFilePendingFooter({ status }: { status: ReactNode }) {
  return <StepFooter><StepActions status={status} action={<Button className="shrink-0" disabled><PlusIcon />Add</Button>} /></StepFooter>
}

/** The chosen file's row while it is read, the same size as the row it becomes. */
function LocalFilePlaceholder() {
  return <section aria-label="Chosen file" aria-busy="true" className={chosenFileRow}>
    <SkeletonLine className="size-6" width="100%" />
    <span className="min-w-0 flex-1">
      <SkeletonLine className="h-5" width="60%" />
      <SkeletonLine className="h-4" width="40%" />
    </span>
    <SkeletonLine className="h-5 w-24" width="100%" />
  </section>
}

const chosenFileRow = "flex items-center gap-3 rounded-lg border border-slate-200 px-3 py-2.5 dark:border-slate-750"

function LocalFileInspection({ path, models, onAdded }: { path: string; models: readonly LocalModel[]; onAdded: () => void }) {
  const { view, retry } = useInspection(path)
  if (view._tag === "Failed") return <InspectionFailure error={view.error} onRetry={retry} />
  if (view._tag === "Reading" || view._tag === "Inspecting") return <>
    <LocalFilePlaceholder />
    <LocalFilePendingFooter status={<InspectionStatus progress={view._tag === "Inspecting" ? Option.some(view.progress) : Option.none()} reading="Reading the file…" />} />
  </>
  const { inspection } = view
  const version = inspection.versions[0]
  if (version === undefined || Option.isSome(inspection.unsupportedArchitecture) || !versionVerdict(version).selectable) {
    return <LocalFileSummary inspection={inspection} models={models} />
  }
  return <LocalFileVersion path={path} inspection={inspection} version={version} models={models} onAdded={onAdded} />
}

function LocalFileVersion({ path, inspection, version, models, onAdded }: { path: string; inspection: CandidateInspection; version: CandidateVersion; models: readonly LocalModel[]; onAdded: () => void }) {
  const { drafters, footer } = useVersionConfirmation(inspection, version, { _tag: "File", path }, onAdded)
  return <LocalFileSummary inspection={inspection} models={models} drafters={drafters} footer={footer} />
}

/** The chosen file's name, lineage, size and one engine verdict, then its drafters and Add. */
export function LocalFileSummary({ inspection, models, drafters, footer }: { inspection: CandidateInspection; models: readonly LocalModel[]; drafters?: ReactNode; footer?: ReactNode }) {
  const variantOf = catalogVariantOfName(models, inspection.variantOf)
  const version = inspection.versions[0]
  const verdict = version === undefined ? undefined : versionVerdict(version)
  const facts = [...(Option.isSome(variantOf) ? [`Variant of ${variantOf.value}`] : []), ...Option.toArray(inspection.license)]
  return <>
    <section aria-label="Chosen file" className={chosenFileRow}>
      <NamedLogo lab={variantLab(inspection.variantOf, inspection.displayName)} name={inspection.displayName} className="size-6" />
      <span className="min-w-0 flex-1">
        <span className="block truncate text-sm font-medium">{inspection.displayName}</span>
        <span className="block min-h-4 truncate text-xs text-slate-500">{facts.join(" · ")}</span>
      </span>
      {version !== undefined && <span className="shrink-0 text-xs tabular-nums text-slate-500">{formatStorageSize(version.sizeBytes)}</span>}
      {verdict !== undefined && Option.isNone(inspection.unsupportedArchitecture) && <span className="shrink-0 text-xs"><VerdictLabel verdict={verdict} /></span>}
    </section>
    {Option.isSome(inspection.unsupportedArchitecture)
      ? <ErrorNotice severity="info" title="Magnitude can’t run this model" description={`${unsupportedReasonMessages(inspection.unsupportedArchitecture.value).join(". ")}.`} />
      : version === undefined && <p className="text-sm text-slate-500">This file isn’t a GGUF model Magnitude can read.</p>}
    {drafters && <DrafterSection>{drafters}</DrafterSection>}
    {footer}
  </>
}

function DrafterSection({ children }: { children: ReactNode }) {
  return <section aria-label="Drafter">
    <h3 className="font-heading text-sm font-semibold">Drafter</h3>
    <p className="mb-2 text-xs text-slate-500">A small model that drafts tokens for this one to verify, for faster responses.</p>
    {children}
  </section>
}

/** Pinned-version note, any failure to add, and Download with its total or Add. */
export function RepositoryFooter({ local, action, pending, failure, onConfirm }: { local: boolean; action: { label: string; download: boolean }; pending: boolean; failure: CandidateNotice | undefined; onConfirm: () => void }) {
  return <StepFooter>
    {failure && <CandidateErrorNotice notice={failure} />}
    <StepActions status={<p className="text-xs text-slate-500">{local ? localFileNote : "Pinned to the current version of this repository"}</p>}
      action={<Button className="shrink-0" disabled={pending} onClick={onConfirm}>{action.download ? <DownloadSimpleIcon /> : <PlusIcon />}{action.label}</Button>} />
  </StepFooter>
}

/** Scrolls the version the dialog opened on into the middle of the dialog, as a linked version far down the list needs. */
export const revealVersionRow = (row: Pick<HTMLElement, "scrollIntoView"> | null) => {
  row?.scrollIntoView({ block: "center" })
}

/** Whether a version's row is scrolled into view: the selected one, while it is still the version the dialog opened on. */
export const revealsVersion = (label: string, selectedVersion: Option.Option<string>, revealedVersion: Option.Option<string>) =>
  Option.contains(selectedVersion, label) && Option.contains(revealedVersion, label)

/** The repository step: header, versions with verdicts, drafter choices, and the footer. */
export function RepositoryView({ inspection, models, selectedVersion, revealedVersion = Option.none(), onSelectVersion, onBack, drafters, footer }: {
  inspection: CandidateInspection
  models: readonly LocalModel[]
  selectedVersion: Option.Option<string>
  revealedVersion?: Option.Option<string>
  onSelectVersion: (label: string) => void
  onBack?: () => void
  drafters?: ReactNode
  footer?: ReactNode
}) {
  const name = useId()
  const variantOf = catalogVariantOfName(models, inspection.variantOf)
  const noRunnableVersion = noRunnableVersionReason(inspection.versions)
  const unselectable = Option.filter(
    Option.flatMap(selectedVersion, label => Option.fromNullable(inspection.versions.find(version => version.label === label))),
    version => !versionVerdict(version).selectable,
  )
  const source = inspection.source
  const facts = [...(Option.isSome(variantOf) ? [`Variant of ${variantOf.value}`] : []), ...Option.toArray(inspection.license)]
  return <>
    <RepositoryHeader displayName={inspection.displayName} lab={variantLab(inspection.variantOf, inspection.displayName)} facts={facts} onBack={onBack}
      source={source._tag === "HuggingFace" ? repositoryLink(source.repository) : <span className="font-mono" title={source.path}>{source.path}</span>} />
    <DialogBody>
      {Option.isSome(inspection.unsupportedArchitecture)
        ? <ErrorNotice severity="info" title="Magnitude can’t run this model" description={`${unsupportedReasonMessages(inspection.unsupportedArchitecture.value).join(". ")}. Nothing was downloaded.`} />
        : inspection.versions.length === 0
          ? <p className="py-4 text-sm text-slate-500">This repository has no GGUF files.</p>
          : <>
            <section aria-label="Version">
              <h3 className="mb-2 font-heading text-sm font-semibold">Version</h3>
              <div role="radiogroup" aria-label="Version" className={versionList}>
                {inspection.versions.map(version => {
                  const verdict = versionVerdict(version)
                  const selected = Option.contains(selectedVersion, version.label)
                  const revealed = revealsVersion(version.label, selectedVersion, revealedVersion)
                  return <label key={version.label} ref={revealed ? revealVersionRow : undefined} data-selected={selected || undefined} className={`${versionRow} ${verdict.selectable ? "cursor-pointer hover:bg-slate-50 dark:hover:bg-slate-800" : selected ? "" : "opacity-60"} ${selected ? "bg-blue-50 dark:bg-slate-800" : ""}`}>
                    <input type="radio" name={name} className="shrink-0 accent-blue-600" checked={selected} disabled={!verdict.selectable} onChange={() => onSelectVersion(version.label)} />
                    <span className={versionLabelColumn} title={version.files.join("\n")}>{version.label}</span>
                    <span className={versionSizeColumn}>{formatStorageSize(version.sizeBytes)}</span>
                    <span className={versionVerdictColumn}>
                      {Option.isSome(version.onThisComputer) && <span className="text-green-700 dark:text-green-400">Already on this computer</span>}
                      <VerdictLabel verdict={verdict} />
                    </span>
                  </label>
                })}
              </div>
            </section>
            {drafters && <DrafterSection>{drafters}</DrafterSection>}
          </>}
      {Option.isNone(inspection.unsupportedArchitecture) && Option.match(noRunnableVersion, {
        onNone: () => Option.match(unselectable, { onNone: () => null, onSome: version => <UnselectableVersionFooter version={version} /> }),
        onSome: reason => <NoRunnableVersionFooter reason={reason} />,
      })}
      {footer}
    </DialogBody>
  </>
}

/** A verdict; the reason behind a short label opens from a focusable trigger, so it reaches keyboard and screen reader users too. */
export function VerdictLabel({ verdict }: { verdict: ReturnType<typeof versionVerdict> }) {
  if (verdict.recommended) return <span className="shrink-0 rounded-full border border-blue-300 px-2 py-0.5 text-blue-700 dark:border-blue-700 dark:text-blue-400">Recommended</span>
  if (verdict.detail === undefined) return <span className="text-slate-500">{verdict.label}</span>
  return <TooltipProvider><ActionTooltip label={verdict.detail} trigger={
    <button type="button" className="max-w-full cursor-help truncate rounded text-slate-500 underline decoration-dotted underline-offset-2 focus-visible:outline-2 focus-visible:outline-blue-500">
      {verdict.label}<span className="sr-only">: {verdict.detail}</span>
    </button>} /></TooltipProvider>
}

/** What a linked or pasted version that can't run here says in place of Download, while another version can. */
export const unselectableVersionNotice = (version: CandidateVersion): NoticeContent => {
  const assessment = Option.getOrUndefined(version.assessment)
  return assessment?._tag === "Unsupported"
    ? { title: `Magnitude can’t run ${version.label}`, description: `${unsupportedReasonMessages(assessment.reasons).join(". ")}. Pick another version to download.` }
    : { title: `${version.label} is too large for this machine`, description: "Pick another version to download." }
}

/** In place of Download when the chosen version can't run here but another one can. */
export function UnselectableVersionFooter({ version }: { version: CandidateVersion }) {
  return <StepFooter><ErrorNotice severity="info" {...unselectableVersionNotice(version)} /></StepFooter>
}

/** In place of Download when nothing can run: what is wrong, from the versions' verdicts. */
export function NoRunnableVersionFooter({ reason }: { reason: string }) {
  return <StepFooter><ErrorNotice severity="info" title="No version of this model runs on this computer" description={reason} /></StepFooter>
}
