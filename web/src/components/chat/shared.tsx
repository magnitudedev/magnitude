import { useMemo, useRef, useState } from "react"
import { Atom, Result, useAtomMount, useAtomValue } from "@effect-atom/atom-react"
import { Cause, Effect, Fiber, Option, Stream } from "effect"
import { CheckIcon, CopyIcon } from "@phosphor-icons/react"
import { writeClipboardText } from "../../lib/clipboard"
export const none = Option.none
export const value = <A,>(option: Option.Option<A>) =>
  Option.getOrUndefined(option)
export const number = (n: number) =>
  n.toLocaleString(undefined, { maximumFractionDigits: 1 })
/** A count of tokens: "1 token", "20 tokens". */
export const tokens = (n: number) => `${number(n)} ${n === 1 ? "token" : "tokens"}`
export const seconds = (ms: number) =>
  `${(ms / 1000).toFixed(ms < 1000 ? 2 : 1)}s`

const tick = Atom.make(Stream.tick("1 second").pipe(Stream.map(() => Date.now())))
/** The time now, updated every second while a component uses it. */
export const useNow = () => {
  const clock = useAtomValue(tick)
  return Result.isSuccess(clock) ? clock.value : Date.now()
}

/** How long something has run: "45s", "1m 15s", then "2h 5m". */
export const duration = (ms: number) => {
  const total = Math.max(0, Math.floor(ms / 1000))
  if (total < 60) return `${total}s`
  if (total < 3600) return `${Math.floor(total / 60)}m ${total % 60}s`
  return `${Math.floor(total / 3600)}h ${Math.floor((total % 3600) / 60)}m`
}

/** How long the model has been thinking: "Thinking" for its first second, then "Thinking for 12s". */
export const thinkingLabel = (ms: number) => ms < 1000 ? "Thinking" : `Thinking for ${duration(ms)}`

/** "Thinking for 12s", counting up from when the thought began. */
export function ThinkingFor({ started }: { started: number }) {
  return <>{thinkingLabel(useNow() - started)}</>
}

/** When a message was sent, as people say it: "just now", "5m ago", "2hr ago", then its date and time once a day old. */
export const sentAt = (time: number, now: number) => {
  const minutes = Math.floor((now - time) / 60_000)
  if (minutes < 1) return "just now"
  if (minutes < 60) return `${minutes}m ago`
  const hours = Math.floor(minutes / 60)
  if (hours < 24) return `${hours}hr ago`
  const date = new Date(time)
  const day = date.toLocaleDateString(undefined, { month: "short", day: "numeric", ...(date.getFullYear() === new Date(now).getFullYear() ? {} : { year: "numeric" }) })
  const clock = date.toLocaleTimeString(undefined, { hour: "numeric", minute: "2-digit" }).replace(/\s?([AaPp])\.?[Mm]\.?$/, (_, half: string) => `${half.toLowerCase()}m`)
  return `${day}, ${clock}`
}

/** `sentAt` for a message, kept current while it shows. */
export function SentAt({ time }: { time: number }) {
  const now = useNow()
  return <time dateTime={new Date(time).toISOString()} title={new Date(time).toLocaleString()}>{sentAt(time, now)}</time>
}

/** Copying a message's text, shown with its time when the message is hovered. */
export function CopyText({ text, label, notify }: { text: string; label: string; notify: (text: string) => void }) {
  const [copied, setCopied] = useState(false)
  const copying = useRef<Fiber.RuntimeFiber<void, never> | null>(null)
  useAtomMount(useMemo(() => Atom.make(Effect.addFinalizer(() =>
    copying.current ? Fiber.interrupt(copying.current).pipe(Effect.asVoid) : Effect.void)), []))
  const copy = () => {
    if (copying.current) Effect.runFork(Fiber.interrupt(copying.current))
    copying.current = Effect.runFork(writeClipboardText(text).pipe(
      Effect.tap(() => Effect.sync(() => setCopied(true))),
      Effect.zipRight(Effect.sleep("1500 millis")),
      Effect.tap(() => Effect.sync(() => setCopied(false))),
      Effect.catchAll(() => Effect.sync(() => {
        setCopied(false)
        notify("Couldn’t copy to the clipboard.")
      })),
    ))
  }
  return (
    <button type="button" className="chat-message-action" aria-label={copied ? "Copied" : label} title={label} disabled={text.length === 0} onClick={copy}>
      {copied ? <CheckIcon /> : <CopyIcon />}
    </button>
  )
}
export const titleCase = (text: string) =>
  text === "none" || text === "off"
    ? "Off"
    : text[0]?.toUpperCase() + text.slice(1)
const unspecified = "This action could not be completed. Please try again."
export const errorText = (result: Result.Result<unknown, unknown>) =>
  Result.isFailure(result)
    ? Option.match(Cause.failureOption(result.cause), {
        onNone: () => unspecified,
        onSome: (error) =>
          error && typeof error === "object" && "message" in error
            ? String(error.message)
            : unspecified,
      })
    : null
