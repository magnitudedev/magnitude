import { memo, useMemo, useRef, useState } from "react"
import { Atom, useAtomMount } from "@effect-atom/atom-react"
import { Effect, Fiber } from "effect"
import { CheckIcon, CopyIcon } from "@phosphor-icons/react"
import { Button } from "../ui/button"
import { writeClipboardText } from "../../lib/clipboard"
import { codeLanguageLabel, highlightCode } from "./code-highlight"

export const ChatCodeBlock = memo(function ChatCodeBlock({
  code,
  language,
  notify,
}: {
  code: string
  language: string
  notify: (text: string) => void
}) {
  const [copiedText, setCopiedText] = useState<string | null>(null)
  const copying = useRef<Fiber.RuntimeFiber<void, never> | null>(null)
  useAtomMount(
    useMemo(
      () =>
        Atom.make(
          Effect.addFinalizer(() =>
            copying.current
              ? Fiber.interrupt(copying.current).pipe(Effect.asVoid)
              : Effect.void
          )
        ),
      []
    )
  )
  const highlighted = useMemo(
    () => highlightCode(code, language),
    [code, language]
  )
  const copied = copiedText === code
  const copy = () => {
    if (copying.current) Effect.runFork(Fiber.interrupt(copying.current))
    copying.current = Effect.runFork(
      writeClipboardText(code).pipe(
        Effect.tap(() => Effect.sync(() => setCopiedText(code))),
        Effect.zipRight(Effect.sleep("3 seconds")),
        Effect.tap(() => Effect.sync(() => setCopiedText(null))),
        Effect.catchAll(() =>
          Effect.sync(() => {
            setCopiedText(null)
            notify(
              "Couldn’t copy the code. Select it and copy manually, or try again."
            )
          })
        )
      )
    )
  }
  return (
    <div className="chat-code-block overflow-hidden rounded-lg border border-slate-200 bg-slate-100 dark:border-slate-750 dark:bg-slate-850">
      <div className="flex items-center justify-between gap-3 border-b border-slate-200 px-3 py-1.5 dark:border-slate-750">
        <span className="min-w-0 truncate font-sans text-xs font-medium text-slate-500 dark:text-slate-400">
          {codeLanguageLabel(language)}
        </span>
        <Button
          type="button"
          variant="ghost"
          size="xs"
          aria-label={copied ? "Code copied" : "Copy code"}
          onClick={copy}
        >
          {copied ? (
            <CheckIcon aria-hidden="true" />
          ) : (
            <CopyIcon aria-hidden="true" />
          )}
          {copied ? "Copied" : "Copy"}
        </Button>
        <span role="status" className="sr-only">
          {copied ? "Code copied to clipboard" : ""}
        </span>
      </div>
      <pre tabIndex={0} aria-label={`${codeLanguageLabel(language)} code`}>
        <code>{highlighted}</code>
      </pre>
    </div>
  )
})
