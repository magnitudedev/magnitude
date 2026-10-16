import { useId, useState } from "react"
import { Cause, Option } from "effect"
import { Result, useAtomValue } from "@effect-atom/atom-react"
import { WarningIcon } from "@phosphor-icons/react"
import {
  drafterCandidateDownloadBytes, drafterCandidateLabel, drafterCandidateSelectable, drafterCandidateWarning,
  formatStorageSize, preselectedDrafterCandidate, useAgentClient,
} from "@magnitudedev/client-common"
import type { CustomLocalModel, DrafterCandidate, DrafterChoice, DrafterReference, DrafterTarget, LocalModel } from "@magnitudedev/sdk"
import { Button } from "./ui/button"
import { Input } from "./ui/input"
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from "./ui/dialog"
import { ErrorNotice } from "./error-notice"
import { SkeletonLine } from "./page-skeletons"
import { candidateErrorNotice } from "./candidate-errors"

/** The drafter the user picked: the preselected one, none, or a specific listed candidate. */
export type DrafterSelection =
  | { readonly _tag: "Auto" }
  | { readonly _tag: "None" }
  | { readonly _tag: "Candidate"; readonly key: string }

export const drafterReferenceKey = (reference: DrafterReference) => JSON.stringify(reference)
export const drafterKey = (candidate: DrafterCandidate) => `${candidate.category}:${drafterReferenceKey(candidate.reference)}`

/** The candidate a selection resolves to; a stale or unbindable pick falls back to the preselected one. */
export const selectedDrafterCandidate = (selection: DrafterSelection, candidates: readonly DrafterCandidate[]): Option.Option<DrafterCandidate> => {
  if (selection._tag === "None") return Option.none()
  if (selection._tag === "Candidate") {
    const picked = candidates.find(candidate => drafterKey(candidate) === selection.key && drafterCandidateSelectable(candidate))
    if (picked) return Option.some(picked)
  }
  return preselectedDrafterCandidate(candidates)
}

/** What to send for a selection: the engine's own preselection stays `Auto`. */
export const drafterChoice = (selection: DrafterSelection, candidates: readonly DrafterCandidate[]): DrafterChoice => {
  if (selection._tag === "None") return { _tag: "None" }
  if (selection._tag === "Auto") return { _tag: "Auto" }
  return Option.match(selectedDrafterCandidate(selection, candidates), {
    onNone: () => ({ _tag: "None" }),
    onSome: candidate => ({ _tag: "Reference", reference: candidate.reference }),
  })
}

/** The selection that shows a model's attached drafter, or the recommendation when it has none. */
export const attachedDrafterSelection = (model: CustomLocalModel, candidates: readonly DrafterCandidate[]): DrafterSelection =>
  Option.match(model.drafter, {
    onNone: () => ({ _tag: "Auto" }),
    onSome: drafter => {
      const match = candidates.find(candidate => drafterReferenceKey(candidate.reference) === drafterReferenceKey(drafter.reference))
      return match ? { _tag: "Candidate", key: drafterKey(match) } : { _tag: "Auto" }
    },
  })

const rowClass = "flex items-start gap-3 px-3 py-2.5"

/** "58% accepted" once this model has served requests with the candidate. */
export const drafterCandidateAcceptance = (candidate: DrafterCandidate): Option.Option<string> =>
  Option.map(candidate.measuredAcceptance, ({ acceptanceRate }) => `${Math.round(acceptanceRate * 100)}% accepted`)

/** A measured candidate that made responses slower than running without a drafter. */
export const drafterCandidateSlower = (candidate: DrafterCandidate): boolean =>
  Option.exists(candidate.measuredAcceptance, ({ slower }) => slower)

/** The drafter choices in preselection order, then "Use a different drafter…", then No drafter. */
export function DrafterList({ candidates, models, selection, onSelect, otherPending = false, otherFailure, onCheckOther }: {
  candidates: readonly DrafterCandidate[]
  models: readonly LocalModel[]
  selection: DrafterSelection
  onSelect: (selection: DrafterSelection) => void
  otherPending?: boolean
  otherFailure?: string | undefined
  onCheckOther: (reference: string) => void
}) {
  const name = useId()
  const [otherOpen, setOtherOpen] = useState(false)
  const [other, setOther] = useState("")
  const selected = selectedDrafterCandidate(selection, candidates)
  const selectedKey = Option.match(selected, { onNone: () => "none", onSome: drafterKey })
  return <div>
    <div role="radiogroup" aria-label="Drafter" className="divide-y divide-slate-200 overflow-hidden rounded-lg border border-slate-200 dark:divide-slate-750 dark:border-slate-750">
      {candidates.map(candidate => {
        const key = drafterKey(candidate)
        const selectable = drafterCandidateSelectable(candidate)
        const warning = drafterCandidateWarning(candidate)
        const bytes = drafterCandidateDownloadBytes(candidate)
        const acceptance = drafterCandidateAcceptance(candidate)
        return <label key={key} className={`${rowClass} ${selectable ? "cursor-pointer hover:bg-slate-50 dark:hover:bg-slate-800" : "opacity-60"}`}>
          <input type="radio" name={name} className="mt-1 accent-blue-600" checked={selectedKey === key} disabled={!selectable} onChange={() => onSelect({ _tag: "Candidate", key })} />
          <span className="min-w-0 flex-1">
            <span className="block truncate text-sm font-medium">{candidate.label}</span>
            <span className="block truncate text-xs text-slate-500">{drafterCandidateLabel(candidate, models)}{candidate.reference._tag === "HuggingFace" ? ` · ${candidate.reference.repository}` : ""}{Option.isSome(acceptance) ? ` · ${acceptance.value}` : ""}</span>
            {drafterCandidateSlower(candidate) && <span className="mt-0.5 flex items-center gap-1 text-xs text-orange-600 dark:text-orange-400"><WarningIcon aria-hidden="true" className="size-3.5 shrink-0" />Slower than no drafter on this computer</span>}
            {Option.isSome(warning) && <span className={`mt-0.5 flex items-center gap-1 text-xs ${selectable ? "text-orange-600 dark:text-orange-400" : "text-slate-500"}`}>{selectable && <WarningIcon aria-hidden="true" className="size-3.5 shrink-0" />}{warning.value}</span>}
          </span>
          <span className="shrink-0 text-xs tabular-nums text-slate-500">{bytes > 0 ? formatStorageSize(bytes) : "No download"}</span>
        </label>
      })}
      <label className={`${rowClass} cursor-pointer hover:bg-slate-50 dark:hover:bg-slate-800`}>
        <input type="radio" name={name} className="mt-1 accent-blue-600" checked={selectedKey === "none"} onChange={() => onSelect({ _tag: "None" })} />
        <span className="min-w-0 flex-1">
          <span className="block text-sm font-medium">No drafter</span>
          <span className="block text-xs text-slate-500">Responses are generated without speculative decoding</span>
        </span>
      </label>
    </div>
    {otherOpen
      ? <form className="mt-3 flex gap-2" onSubmit={event => { event.preventDefault(); if (other.trim()) onCheckOther(other.trim()) }}>
        <Input aria-label="Different drafter" autoFocus placeholder="Hugging Face link, owner/repo/file.gguf, or a file path" value={other} onChange={event => setOther(event.target.value)} className="min-w-0 flex-1" />
        <Button type="submit" variant="outline" disabled={otherPending || !other.trim()}>{otherPending ? "Checking…" : "Check"}</Button>
      </form>
      : <Button variant="link" size="unstyled" className="mt-3 text-xs" onClick={() => setOtherOpen(true)}>Use a different drafter…</Button>}
    {otherFailure && <ErrorNotice className="mt-2" title="Couldn’t check this drafter" description={otherFailure} />}
  </div>
}

/** The drafter list for one target, read from the service, with the user's different reference checked by the engine. */
export function useDrafterCandidates(target: DrafterTarget) {
  const client = useAgentClient()
  const [other, setOther] = useState(Option.none<string>())
  const base = useAtomValue(client.Models.ListDrafterCandidates({ target, other: Option.none() })).result
  const checked = useAtomValue(client.Models.ListDrafterCandidates({ target, other })).result
  const current = Option.isSome(other) && Result.isSuccess(checked) ? checked : base
  const otherPending = Option.isSome(other) && !Result.isSuccess(checked) && !Result.isFailure(checked)
  const otherFailure = Option.isSome(other) && Result.isFailure(checked)
    ? Option.match(Cause.failureOption(checked.cause), { onNone: () => "Check the reference, then try again.", onSome: error => candidateErrorNotice(error).description })
    : undefined
  return { candidates: current, otherPending, otherFailure, checkOther: (reference: string) => setOther(Option.some(reference)) }
}

/** Change for a custom model's drafter, from the same list as the repository step. */
export function DrafterPickerDialog({ model, models, open, onOpenChange, onChoose }: {
  model: CustomLocalModel
  models: readonly LocalModel[]
  open: boolean
  onOpenChange: (open: boolean) => void
  onChoose: (choice: DrafterChoice) => void
}) {
  return <Dialog open={open} onOpenChange={onOpenChange}>
    <DialogContent className="max-h-[85vh] overflow-y-auto sm:max-w-lg">
      <DialogHeader>
        <DialogTitle>Drafter</DialogTitle>
        <DialogDescription>A small model that drafts tokens for this one to verify, for faster responses.</DialogDescription>
      </DialogHeader>
      {open && <DrafterPicker model={model} models={models} onCancel={() => onOpenChange(false)} onChoose={choice => { onOpenChange(false); onChoose(choice) }} />}
    </DialogContent>
  </Dialog>
}

function DrafterPicker({ model, models, onCancel, onChoose }: { model: CustomLocalModel; models: readonly LocalModel[]; onCancel: () => void; onChoose: (choice: DrafterChoice) => void }) {
  const { candidates, otherPending, otherFailure, checkOther } = useDrafterCandidates({ _tag: "Model", modelId: model.modelId })
  const [picked, setPicked] = useState(Option.none<DrafterSelection>())
  const list = Result.isSuccess(candidates) ? candidates.value.candidates : []
  const selection = Option.getOrElse(picked, () => attachedDrafterSelection(model, list))
  const choice = drafterChoice(selection, list)
  return <>
    {Result.isFailure(candidates) ? <ErrorNotice title="Couldn’t list drafters" description="Check your connection, then try again." />
      : !Result.isSuccess(candidates) ? <div className="space-y-2"><SkeletonLine className="h-10" width="100%" /><SkeletonLine className="h-10" width="100%" /></div>
      : <DrafterList candidates={list} models={models} selection={selection} onSelect={next => setPicked(Option.some(next))} otherPending={otherPending} otherFailure={otherFailure} onCheckOther={checkOther} />}
    <DialogFooter>
      <Button variant="outline" onClick={onCancel}>Cancel</Button>
      <Button disabled={!Result.isSuccess(candidates)} onClick={() => onChoose(choice)}>{choice._tag === "None" ? "Turn off drafter" : "Use this drafter"}</Button>
    </DialogFooter>
  </>
}
