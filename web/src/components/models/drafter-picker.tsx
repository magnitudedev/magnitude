import { useState } from "react"
import { Cause, Option } from "effect"
import { Result, useAtomValue } from "@effect-atom/atom-react"
import { CaretDownIcon, CircleNotchIcon, LightningIcon, PlusIcon, WarningIcon, XCircleIcon } from "@phosphor-icons/react"
import type { CustomLocalModel, DrafterCandidate, DrafterChoice, DrafterReference, DrafterTarget } from "@magnitudedev/sdk"
import { drafterCandidateDownloadBytes, drafterCandidateSelectable, drafterCandidateWarning, formatStorageSize, preselectedDrafterCandidate, useAgentClient } from "@magnitudedev/client-common"
import { cn } from "@/lib/utils"
import { Button } from "../ui/button"
import { Input } from "../ui/input"
import { Dialog, DialogContent, DialogTitle } from "../ui/dialog"
import { candidateErrorNotice } from "../candidate-errors"
import { Badge, RepositoryLink } from "./parts"
import { methodName } from "./data"

/** The drafter the user picked: the preselected one, none, a listed candidate, or one they entered by reference. */
export type DrafterSelection =
  | { readonly _tag: "Auto" }
  | { readonly _tag: "None" }
  | { readonly _tag: "Candidate"; readonly candidate: DrafterCandidate }
  | { readonly _tag: "Entered"; readonly reference: string }

export const AUTO_DRAFTER: DrafterSelection = { _tag: "Auto" }

const referenceKey = (reference: DrafterReference) => JSON.stringify(reference)
const drafterKey = (candidate: DrafterCandidate) => `${candidate.category}:${referenceKey(candidate.reference)}`

/** A different drafter the user entered, as the engine's check of it stands. */
export type EnteredDrafter =
  | { readonly _tag: "Idle" }
  | { readonly _tag: "Checking" }
  | { readonly _tag: "Checked"; readonly entered: Option.Option<DrafterCandidate> }
  | { readonly _tag: "Failed"; readonly message: string }

/**
 * A target's drafters as a selection resolves them: the listed candidates (null until listed), the drafter entered by
 * reference with the engine's check, and the one the selection runs with. A pick that doesn't bind, and an entered
 * drafter not yet checked or refused, give way to the preselected one.
 */
export function useDrafters(target: DrafterTarget, selection: DrafterSelection) {
  const client = useAgentClient()
  const reference = selection._tag === "Entered" ? Option.some(selection.reference) : Option.none<string>()
  const listed = useAtomValue(client.Models.ListDrafterCandidates({ target, other: Option.none() })).result
  const checked = useAtomValue(client.Models.ListDrafterCandidates({ target, other: reference })).result
  const candidates = Result.isSuccess(listed) ? listed.value.candidates : null
  const entered: EnteredDrafter = Option.isNone(reference) ? { _tag: "Idle" }
    : Result.isSuccess(checked) && !checked.waiting ? { _tag: "Checked", entered: checked.value.entered }
    : Result.isFailure(checked) ? { _tag: "Failed", message: Option.match(Cause.failureOption(checked.cause), { onNone: () => "Check the reference, then try again.", onSome: error => candidateErrorNotice(error).description ?? "Check the reference, then try again." }) }
    : { _tag: "Checking" }
  const preselected = preselectedDrafterCandidate(candidates ?? [])
  const picked = selection._tag === "Candidate" ? Option.some(selection.candidate)
    : entered._tag === "Checked" ? entered.entered : Option.none<DrafterCandidate>()
  const chosen = selection._tag === "None" ? Option.none<DrafterCandidate>()
    : Option.isSome(picked) && drafterCandidateSelectable(picked.value) ? picked : preselected
  return { candidates, failed: Result.isFailure(listed), entered, chosen }
}

/** What to send for a selection: the preselection stays `Auto`, so the service keeps choosing it. */
export const drafterChoice = (selection: DrafterSelection, chosen: Option.Option<DrafterCandidate>): DrafterChoice =>
  selection._tag === "Auto" || selection._tag === "None" ? selection
    : Option.match(chosen, { onNone: () => ({ _tag: "None" }), onSome: candidate => ({ _tag: "Reference", reference: candidate.reference }) })

/** The selection showing a custom model's attached drafter, No drafter once it was turned off, else the preselected one. */
export const attachedSelection = (model: CustomLocalModel): DrafterSelection =>
  model.drafterChoice._tag === "None" ? { _tag: "None" }
    : model.drafterChoice._tag === "Reference" ? Option.match(model.drafter, {
      onNone: () => AUTO_DRAFTER,
      onSome: (drafter): DrafterSelection => ({ _tag: "Candidate", candidate: {
        category: "Other", reference: drafter.reference, name: drafter.name, label: drafter.label, sizeBytes: drafter.sizeBytes,
        lineage: "None", verdict: drafter.verdict, automatic: false, onThisComputer: drafter.availability._tag === "Present",
        method: drafter.verdict._tag === "Bound" ? Option.some(drafter.verdict.method) : Option.none(),
      } }),
    })
    : AUTO_DRAFTER

const methodOf = (candidate: DrafterCandidate) => Option.match(candidate.method, { onNone: () => candidate.name, onSome: method => methodName[method._tag] })

/** A drafter in one line's words: its method, and whether it is built in. */
export const drafterTitle = (candidate: DrafterCandidate) =>
  candidate.reference._tag === "Embedded" ? `built-in ${methodOf(candidate)}` : `${methodOf(candidate)} drafter`

function Radio({ checked }: { checked: boolean }) {
  return <span className={cn("mt-0.5 flex size-4 shrink-0 items-center justify-center rounded-full border", checked ? "border-blue-600 dark:border-blue-400" : "border-slate-300 dark:border-slate-600")}>
    {checked && <span className="size-2 rounded-full bg-blue-600 dark:bg-blue-400" />}
  </span>
}

/** One drafter: its method, where it comes from, and what it costs; a download says its size, one already here the memory it adds. */
function CandidateRow({ candidate, checked, recommended, onSelect }: { candidate: DrafterCandidate; checked: boolean; recommended: boolean; onSelect: () => void }) {
  const usable = drafterCandidateSelectable(candidate)
  const download = drafterCandidateDownloadBytes(candidate)
  const memory = usable && download === 0 && candidate.verdict._tag === "Bound" ? candidate.verdict.memoryBytes : 0
  const rejection = drafterCandidateWarning(candidate)
  return <button type="button" disabled={!usable} onClick={onSelect}
    className={cn("flex w-full items-start gap-3 px-3 py-2.5 text-left text-[13px]", checked ? "bg-blue-50/60 dark:bg-slate-800" : "hover:bg-slate-50 dark:hover:bg-slate-800/50", !usable && "cursor-not-allowed")}>
    <Radio checked={checked} />
    <span className="min-w-0 flex-1">
      <span className={cn("flex items-center gap-2 font-medium", !usable && "text-slate-500")}>
        <LightningIcon weight="fill" className={cn("size-3.5 shrink-0", usable ? "text-amber-500" : "text-slate-400")} />
        <span className="truncate">{methodOf(candidate)}</span>
        {recommended && <Badge tone="blue">Recommended</Badge>}
        {candidate.onThisComputer && candidate.reference._tag !== "Embedded" && <Badge>Downloaded</Badge>}
      </span>
      <span className="mt-0.5 flex min-w-0 flex-wrap items-center gap-x-1.5 text-xs text-slate-500">
        <span>{candidate.label}</span>
        {candidate.reference._tag === "HuggingFace" && <>·<RepositoryLink repository={candidate.reference.repository}>{candidate.reference.repository}</RepositoryLink></>}
        {memory > 0 && <>· <span>+{formatStorageSize(memory)} memory</span></>}
      </span>
      {usable && candidate.lineage === "Fingerprint" && <span className="mt-1 flex items-center gap-1 text-xs text-orange-700 dark:text-orange-300"><WarningIcon className="size-3.5 shrink-0" />Not trained for this model. It may guess right less often.</span>}
      {Option.isSome(rejection) && <span className="mt-1 flex items-center gap-1 text-xs text-red-600 dark:text-red-400"><XCircleIcon className="size-3.5 shrink-0" />{rejection.value}</span>}
    </span>
    {usable && <span className="shrink-0 text-xs tabular-nums text-slate-500">{download > 0 ? formatStorageSize(download) : "No download"}</span>}
  </button>
}

/** A box for a drafter the list didn't find: a Hugging Face repository or link, or a file on this computer. */
function OtherDrafter({ other, check }: { other: EnteredDrafter; check: (reference: string) => void }) {
  const [open, setOpen] = useState(false)
  const [entry, setEntry] = useState("")
  if (!open) return <button type="button" onClick={() => setOpen(true)}
    className="mt-2 inline-flex items-center gap-1 text-xs font-medium text-blue-700 hover:underline dark:text-blue-400"><PlusIcon className="size-3" />Use a different drafter…</button>
  const checking = other._tag === "Checking"
  return <div className="mt-2">
    <form className="flex gap-2" onSubmit={event => { event.preventDefault(); if (entry.trim()) check(entry.trim()) }}>
      <Input autoFocus aria-label="Different drafter" value={entry} onChange={event => setEntry(event.target.value)} disabled={checking} placeholder="Hugging Face repo or link (owner/repo, owner/repo/file.gguf), or a file path" className="h-8 flex-1 text-xs" />
      <Button size="sm" type="submit" disabled={checking || entry.trim() === ""}>{checking ? <><CircleNotchIcon className="motion-safe:animate-spin" />Checking…</> : "Check"}</Button>
      <Button size="sm" variant="ghost" type="button" onClick={() => setOpen(false)}>Cancel</Button>
    </form>
    {other._tag === "Failed"
      ? <p className="mt-1.5 text-[11px] text-red-600 dark:text-red-400">{other.message}</p>
      : other._tag === "Checked" && Option.isNone(other.entered)
        ? <p className="mt-1.5 text-[11px] text-red-600 dark:text-red-400">Magnitude couldn't find a drafter there.</p>
        : <p className="mt-1.5 text-[11px] text-slate-500">Magnitude loads it with this model to check it pairs, and how much memory it adds.</p>}
  </div>
}

/**
 * The drafters that could pair with a target, as one choice: those found for it, in the order an automatic choice
 * prefers them, then the drafter the user entered, no drafter, and a way to enter another.
 */
export function DrafterPicker({ target, selection, onSelect }: { target: DrafterTarget; selection: DrafterSelection; onSelect: (selection: DrafterSelection) => void }) {
  const { candidates, entered, chosen } = useDrafters(target, selection)
  if (candidates === null) return <p role="status" className="flex items-center gap-1.5 text-xs text-slate-500"><CircleNotchIcon aria-hidden="true" className="size-3.5 motion-safe:animate-spin" />Finding drafters…</p>
  const extra = selection._tag === "Candidate" ? [selection.candidate] : entered._tag === "Checked" ? Option.toArray(entered.entered) : []
  const listed = [...candidates, ...extra.filter(candidate => !candidates.some(listedCandidate => drafterKey(listedCandidate) === drafterKey(candidate)))]
  const chosenKey = Option.map(chosen, drafterKey)
  const recommended = Option.map(preselectedDrafterCandidate(candidates), drafterKey)
  return <div>
    <div role="radiogroup" aria-label="Drafter" className="divide-y divide-slate-200 overflow-hidden rounded-lg border border-slate-200 bg-white dark:divide-slate-750 dark:border-slate-750 dark:bg-slate-850">
      {listed.map(candidate => <CandidateRow key={drafterKey(candidate)} candidate={candidate} checked={Option.contains(chosenKey, drafterKey(candidate))}
        recommended={Option.contains(recommended, drafterKey(candidate))} onSelect={() => onSelect({ _tag: "Candidate", candidate })} />)}
      <button type="button" onClick={() => onSelect({ _tag: "None" })} className={cn("flex w-full items-start gap-3 px-3 py-2.5 text-left text-[13px]", Option.isNone(chosen) ? "bg-blue-50/60 dark:bg-slate-800" : "hover:bg-slate-50 dark:hover:bg-slate-800/50")}>
        <Radio checked={Option.isNone(chosen)} />
        <span className="flex-1"><span className="block font-medium">No drafter</span><span className="mt-0.5 block text-xs text-slate-500">Generates one token at a time, without speculative decoding</span></span>
      </button>
    </div>
    <OtherDrafter other={entered} check={reference => onSelect({ _tag: "Entered", reference })} />
  </div>
}

/**
 * A quant's drafter as one line beside its download, naming the one it runs with; opening it lays every drafter that
 * could pair across the row below. Its parts are the row's own items, so the list takes the row's full width.
 */
export function DrafterSection({ target, selection, onSelect }: { target: DrafterTarget; selection: DrafterSelection; onSelect: (selection: DrafterSelection) => void }) {
  const [open, setOpen] = useState(false)
  const { candidates, failed, chosen } = useDrafters(target, selection)
  if (failed) return <span />
  if (candidates === null) return <span role="status" className="flex items-center gap-1.5 text-xs text-slate-500"><CircleNotchIcon aria-hidden="true" className="size-3.5 motion-safe:animate-spin" />Finding a drafter…</span>
  const download = Option.match(chosen, { onNone: () => 0, onSome: drafterCandidateDownloadBytes })
  return <>
    <button type="button" aria-expanded={open} onClick={() => setOpen(value => !value)}
      title="A drafter is a small model that guesses ahead, so this one writes several tokens per step"
      className="group inline-flex items-center gap-1.5 text-xs text-slate-600 dark:text-slate-400">
      <LightningIcon weight="fill" className={cn("size-3.5", Option.isNone(chosen) ? "text-slate-400" : "text-amber-500")} />
      <span>{Option.isSome(chosen) && "Using "}<span className="font-medium text-slate-800 underline decoration-slate-300 decoration-dotted underline-offset-2 group-hover:text-blue-700 group-hover:decoration-blue-700 dark:text-slate-200 dark:decoration-slate-600 dark:group-hover:text-blue-400">{Option.match(chosen, { onNone: () => "No drafter", onSome: drafterTitle })}</span></span>
      {download > 0 && <span className="text-slate-500">· {formatStorageSize(download)}</span>}
      <CaretDownIcon className={cn("size-3 text-slate-400 transition-transform group-hover:text-slate-600", open && "rotate-180")} />
    </button>
    {open && <div className="order-last basis-full"><DrafterPicker target={target} selection={selection} onSelect={onSelect} /></div>}
  </>
}

/** What confirming a drafter choice for a model already here does, as its button says it. */
const dialogAction = (model: CustomLocalModel, selection: DrafterSelection, chosen: Option.Option<DrafterCandidate>) => {
  const attached = Option.map(model.drafter, drafter => referenceKey(drafter.reference))
  return Option.match(chosen, {
    onNone: () => Option.isSome(attached) ? "Turn off drafter" : "Keep without a drafter",
    onSome: candidate => {
      if (Option.contains(attached, referenceKey(candidate.reference)) && selection._tag !== "None") return "Keep this drafter"
      const download = drafterCandidateDownloadBytes(candidate)
      return download > 0 ? `Download ${formatStorageSize(download)} and use` : "Use this drafter"
    },
  })
}

/** The Drafter row's picker on a custom model already here: the same list, then a button that says what choosing does. */
export function DrafterDialog({ model, bits, onClose, onChoose }: {
  model: CustomLocalModel; bits: number; onClose: () => void; onChoose: (choice: DrafterChoice) => void
}) {
  const target: DrafterTarget = { _tag: "Model", modelId: model.modelId }
  const [selection, setSelection] = useState(() => attachedSelection(model))
  const { chosen, candidates } = useDrafters(target, selection)
  const turningOff = Option.isNone(chosen) && Option.isSome(model.drafter)
  return <Dialog open onOpenChange={open => { if (!open) onClose() }}>
    <DialogContent className="max-h-[85vh] overflow-y-auto sm:max-w-xl">
      <DialogTitle className="font-heading text-base font-semibold">Drafter for {model.presentation.displayName}{bits > 0 && <span className="font-normal text-slate-500"> · {bits}-bit</span>}</DialogTitle>
      <p className="-mt-2 text-[13px] text-slate-600 dark:text-slate-400">A small model that guesses ahead, so this one writes several tokens per step.</p>
      <DrafterPicker target={target} selection={selection} onSelect={setSelection} />
      <div className="flex justify-end gap-2">
        <Button variant="ghost" onClick={onClose}>Cancel</Button>
        <Button disabled={candidates === null} variant={turningOff ? "outline" : "default"} onClick={() => { onChoose(drafterChoice(selection, chosen)); onClose() }}>{dialogAction(model, selection, chosen)}</Button>
      </div>
    </DialogContent>
  </Dialog>
}
