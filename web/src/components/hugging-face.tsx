import { useState } from "react"
import { Cause, Effect, Exit, Option } from "effect"
import { Result, useAtomSet, useAtomValue } from "@effect-atom/atom-react"
import { ArrowUpRightIcon, CheckIcon, CopyIcon } from "@phosphor-icons/react"
import { huggingFaceAccountSummary, useAgentClient } from "@magnitudedev/client-common"
import {
  HuggingFaceSignInFailed, huggingFaceRepositoryUrl,
  type HuggingFaceAccessRequired, type HuggingFaceAccount, type HuggingFaceDeviceSignIn, type HuggingFaceSignInAttemptId, type HuggingFaceSignInMethod,
} from "@magnitudedev/sdk"
import { openExternalUrl } from "../lib/external-url"
import { writeClipboardText } from "../lib/clipboard"
import { useSession } from "../session"
import { Button } from "./ui/button"
import { ErrorNotice, NoticeAction, type NoticeContent } from "./error-notice"
import { SettingsRow } from "./settings-row"
import { HuggingFaceLogo } from "./model-logo"
import { SkeletonLine } from "./page-skeletons"
import type { CandidateNotice } from "./candidate-errors"

const failureNotice = (cause: Cause.Cause<unknown>, title: string): NoticeContent => Option.match(Cause.failureOption(cause), {
  onSome: error => ({ title, description: error instanceof HuggingFaceSignInFailed ? error.message : "Couldn’t reach the Magnitude service. Try again." }),
  onNone: () => ({ title, description: "Try again." }),
})

/** A device sign-in this window started: the code to enter, the page to enter it on, and when the code expires. */
export interface DeviceCode {
  readonly attemptId: HuggingFaceSignInAttemptId
  readonly userCode: string
  readonly verificationUrl: string
  readonly expiresAt: number
}

const deviceCode = (start: HuggingFaceDeviceSignIn, startedAt: number): DeviceCode => ({
  attemptId: start.attemptId,
  userCode: start.userCode,
  verificationUrl: Option.getOrElse(start.verificationUriComplete, () => start.verificationUri),
  expiresAt: startedAt + start.expiresIn * 1000,
})

/** The code to show: only while the account is waiting for a device sign-in and this window's code hasn't expired. */
export const visibleDeviceCode = (account: HuggingFaceAccount, code: Option.Option<DeviceCode>, now: number): Option.Option<DeviceCode> =>
  account._tag === "SigningIn" && account.method === "Device" ? Option.filter(code, current => current.expiresAt > now) : Option.none()

/**
 * Starts a Hugging Face sign-in the way this window can finish it. Hugging Face redirects a browser
 * sign-in to the viewer's own loopback address, which reaches ACN only when both share a computer.
 * The desktop app always does, so it opens the browser; a browser window can't know where ACN runs
 * (an SSH tunnel serves it from a loopback origin too), so it shows a code to enter on Hugging Face.
 */
export function useHuggingFaceSignIn() {
  const client = useAgentClient()
  const method: HuggingFaceSignInMethod = Option.isSome(useSession().clientWindow) ? "Browser" : "Device"
  const startBrowser = useAtomSet(client.HuggingFace.StartHuggingFaceSignIn, { mode: "promiseExit" })
  const startDevice = useAtomSet(client.HuggingFace.StartHuggingFaceDeviceSignIn)
  const browser = useAtomValue(client.HuggingFace.StartHuggingFaceSignIn)
  const device = useAtomValue(client.HuggingFace.StartHuggingFaceDeviceSignIn)
  const starting: Result.Result<unknown, unknown> = method === "Browser" ? browser : device
  const code = Result.isSuccess(device) && !device.waiting ? Option.some(deviceCode(device.value, device.timestamp)) : Option.none<DeviceCode>()
  return {
    method,
    signIn: method === "Browser"
      ? () => void startBrowser({}).then(exit => { if (Exit.isSuccess(exit)) openExternalUrl(exit.value.authorizationUrl) })
      : () => startDevice({}),
    starting: starting.waiting,
    failure: Result.isFailure(starting) && !starting.waiting ? failureNotice(starting.cause, "Couldn’t start signing in to Hugging Face") : undefined,
    deviceCode: code,
    copyCode: (): Promise<boolean> => Option.match(code, {
      onNone: () => Promise.resolve(false),
      onSome: current => Effect.runPromise(writeClipboardText(current.userCode).pipe(Effect.as(true), Effect.orElseSucceed(() => false))),
    }),
    openVerification: () => Option.match(code, { onNone: () => {}, onSome: current => openExternalUrl(current.verificationUrl) }),
  }
}

const expiryTime = (expiresAt: number) => new Date(expiresAt).toLocaleTimeString([], { hour: "numeric", minute: "2-digit" })

/** The code to enter on Hugging Face, with Copy and a button that opens the page to enter it on. */
export function DeviceCodePanel({ code, onCopy, onOpen }: { code: DeviceCode; onCopy: () => Promise<boolean>; onOpen: () => void }) {
  const [copied, setCopied] = useState<"idle" | "copied" | "failed">("idle")
  return <div className="mt-3 rounded-md border border-slate-200 bg-slate-50 px-3 py-2.5 dark:border-slate-700 dark:bg-slate-900">
    <p className="m-0 text-xs text-slate-600 dark:text-slate-400">Enter this code on Hugging Face</p>
    <div className="mt-1.5 flex flex-wrap items-center gap-2">
      <code aria-label="Hugging Face sign-in code" className="mr-1 select-all font-mono text-2xl font-semibold tracking-[0.15em] text-slate-900 dark:text-slate-100">{code.userCode}</code>
      <Button size="sm" variant="outline" onClick={() => void onCopy().then(ok => setCopied(ok ? "copied" : "failed"))}>
        {copied === "copied" ? <CheckIcon aria-hidden="true" className="text-green-600" /> : <CopyIcon aria-hidden="true" />}{copied === "copied" ? "Copied" : "Copy"}
      </Button>
      <Button size="sm" onClick={onOpen}>Open huggingface.co<ArrowUpRightIcon aria-hidden="true" /></Button>
    </div>
    <p className="mt-1.5 mb-0 text-xs text-slate-500">The code expires at {expiryTime(code.expiresAt)}.</p>
    {copied === "failed" && <p role="alert" className="mt-1 mb-0 text-xs text-red-700 dark:text-red-300">Couldn’t copy the code. Select it and copy it yourself.</p>}
  </div>
}

/** The account line: who is signed in, with their avatar, a sign-in in progress, or why it is signed out. */
function AccountHint({ account }: { account: HuggingFaceAccount }) {
  if (account._tag !== "SignedIn" || Option.isNone(account.avatarUrl)) return <>{huggingFaceAccountSummary(account)}</>
  return <span className="flex items-center gap-1.5">
    <img src={account.avatarUrl.value} alt="" className="size-4 shrink-0 rounded-full" />
    <span>{huggingFaceAccountSummary(account)}</span>
  </span>
}

const huggingFaceLabel = <span className="inline-flex items-center gap-1.5">Hugging Face<HuggingFaceLogo className="size-4" /></span>

/** Settings → Hugging Face: the account, with Sign in, Cancel (and the code to enter) while a sign-in waits, or Sign out. */
export function HuggingFaceRowView({ account, pending, notice, deviceCode, onSignIn, onCancel, onSignOut, onCopyCode, onOpenVerification }: {
  account: Result.Result<HuggingFaceAccount, unknown>
  pending: boolean
  notice: NoticeContent | undefined
  deviceCode: Option.Option<DeviceCode>
  onSignIn: () => void
  onCancel: () => void
  onSignOut: () => void
  onCopyCode: () => Promise<boolean>
  onOpenVerification: () => void
}) {
  if (!Result.isSuccess(account)) return <SettingsRow label={huggingFaceLabel}
    hint={Result.isFailure(account) ? undefined : <SkeletonLine className="h-4 text-xs" width="220px" />}
    alert={Result.isFailure(account) ? <ErrorNotice title="Couldn’t read your Hugging Face account" description="Check the service in Settings, then try again." /> : undefined} />
  const current = account.value
  const code = visibleDeviceCode(current, deviceCode, Date.now())
  return <SettingsRow label={huggingFaceLabel} hint={<AccountHint account={current} />} alert={notice && <ErrorNotice {...notice} />}
    control={current._tag === "SignedOut" ? <Button size="sm" variant="outline" disabled={pending} onClick={onSignIn}>Sign in</Button>
      : current._tag === "SigningIn" ? <Button size="sm" variant="outline" disabled={pending} onClick={onCancel}>Cancel</Button>
      : <Button size="sm" variant="outline" disabled={pending} onClick={onSignOut}>Sign out</Button>}>
    {Option.isSome(code) && <DeviceCodePanel key={code.value.attemptId} code={code.value} onCopy={onCopyCode} onOpen={onOpenVerification} />}
  </SettingsRow>
}

export function HuggingFaceRow() {
  const client = useAgentClient()
  const account = useAtomValue(client.HuggingFace.GetHuggingFaceAccount({})).result
  const { signIn, starting, failure, deviceCode, copyCode, openVerification } = useHuggingFaceSignIn()
  const cancel = useAtomSet(client.HuggingFace.CancelHuggingFaceSignIn)
  const cancelling = useAtomValue(client.HuggingFace.CancelHuggingFaceSignIn)
  const signOut = useAtomSet(client.HuggingFace.SignOutOfHuggingFace)
  const signingOut = useAtomValue(client.HuggingFace.SignOutOfHuggingFace)
  const tag = Result.isSuccess(account) ? Option.some(account.value._tag) : Option.none()
  const notice = Option.match(tag, {
    onNone: () => undefined,
    onSome: current => current === "SignedOut" ? failure
      : current === "SigningIn" && Result.isFailure(cancelling) && !cancelling.waiting ? { title: "Couldn’t cancel signing in", description: "Try again." }
      : current === "SignedIn" && Result.isFailure(signingOut) && !signingOut.waiting ? failureNotice(signingOut.cause, "Couldn’t sign out of Hugging Face")
      : undefined,
  })
  return <HuggingFaceRowView account={account} pending={starting || cancelling.waiting || signingOut.waiting} notice={notice} deviceCode={deviceCode}
    onSignIn={signIn} onCancel={() => cancel({ attemptId: Option.none() })} onSignOut={() => signOut({})} onCopyCode={copyCode} onOpenVerification={openVerification} />
}

const linkAction = "inline-flex min-h-6 items-center gap-0.5 py-0.5 text-xs font-medium text-inherit underline-offset-4 hover:underline"

/**
 * What a gated repository offers: Sign in until the account is signed in, or the repository's page to
 * accept its terms; Try again either way, once access is granted.
 */
export function HuggingFaceAccessActions({ error, account, starting, onSignIn, onRetry }: {
  error: HuggingFaceAccessRequired
  account: Option.Option<HuggingFaceAccount>
  starting: boolean
  onSignIn: () => void
  onRetry: (() => void) | undefined
}) {
  const retry = onRetry && <NoticeAction onClick={onRetry}>Try again</NoticeAction>
  if (error.access === "AcceptTerms") return <>
    <a href={huggingFaceRepositoryUrl(error.repository)} target="_blank" rel="noreferrer" className={linkAction}>Open on Hugging Face<ArrowUpRightIcon aria-hidden="true" className="size-3" /></a>
    {retry}
  </>
  const state = Option.map(account, current => current._tag)
  return <>
    {Option.isSome(account) && account.value._tag === "SigningIn" ? <NoticeAction disabled onClick={onSignIn}>{huggingFaceAccountSummary(account.value)}</NoticeAction>
      : !Option.contains(state, "SignedIn") && <NoticeAction disabled={starting} onClick={onSignIn}>Sign in</NoticeAction>}
    {retry}
  </>
}

function HuggingFaceAccessNotice({ notice, error, onRetry }: { notice: CandidateNotice; error: HuggingFaceAccessRequired; onRetry: (() => void) | undefined }) {
  const client = useAgentClient()
  const account = useAtomValue(client.HuggingFace.GetHuggingFaceAccount({})).result
  const { signIn, starting, failure, deviceCode, copyCode, openVerification } = useHuggingFaceSignIn()
  const code = error.access === "SignIn" && Result.isSuccess(account) ? visibleDeviceCode(account.value, deviceCode, Date.now()) : Option.none<DeviceCode>()
  return <ErrorNotice {...notice} actions={<HuggingFaceAccessActions error={error} account={Result.isSuccess(account) ? Option.some(account.value) : Option.none()}
    starting={starting} onSignIn={signIn} onRetry={onRetry} />}>
    {error.access === "SignIn" && failure && <p className="mt-1 text-xs leading-5">{failure.title}. {failure.description}</p>}
    {Option.isSome(code) && <DeviceCodePanel key={code.value.attemptId} code={code.value} onCopy={copyCode} onOpen={openVerification} />}
  </ErrorNotice>
}

/** A candidate failure with Try again when it may clear, and Hugging Face's sign-in or terms when a gated repository needs them. */
export function CandidateErrorNotice({ notice, onRetry }: { notice: CandidateNotice; onRetry?: () => void }) {
  if (notice.access) return <HuggingFaceAccessNotice notice={notice} error={notice.access} onRetry={onRetry} />
  return <ErrorNotice {...notice} actions={notice.retryable && onRetry ? <NoticeAction onClick={onRetry}>Try again</NoticeAction> : undefined} />
}
