import { Effect, Fiber } from "effect"
import { Atom, useAtomMount } from "@effect-atom/atom-react"
import { useMemo, useRef, useState } from "react"
import { CopyIcon, CheckIcon } from "@phosphor-icons/react"

/** A command shown inline; the whole control copies it and highlights on hover. */
export function CopyCommand({ command, label }: { command: string; label: string }) {
  const [copied, setCopied] = useState(false)
  const [failed, setFailed] = useState(false)
  const copying = useRef<Fiber.RuntimeFiber<void, never> | null>(null)
  useAtomMount(useMemo(() => Atom.make(Effect.addFinalizer(() => copying.current ? Fiber.interrupt(copying.current).pipe(Effect.asVoid) : Effect.void)), []))
  const copy = () => {
    if (copying.current) Effect.runFork(Fiber.interrupt(copying.current))
    copying.current = Effect.runFork(Effect.tryPromise(() => navigator.clipboard.writeText(command)).pipe(
      Effect.tap(() => Effect.sync(() => { setCopied(true); setFailed(false) })),
      Effect.zipRight(Effect.sleep("3 seconds")),
      Effect.tap(() => Effect.sync(() => setCopied(false))),
      Effect.catchAll(() => Effect.sync(() => { setCopied(false); setFailed(true) })),
    ))
  }
  return <div className="min-w-0 flex-1">
    <button type="button" aria-label={label} title={command} onClick={copy} className="flex h-9 w-full min-w-0 cursor-pointer items-center gap-2 rounded-md border border-slate-200 bg-slate-50 px-3 text-left transition-colors hover:border-blue-400 hover:bg-blue-50 focus-visible:outline-2 focus-visible:outline-blue-500 dark:border-slate-700 dark:bg-slate-900 dark:hover:border-blue-500 dark:hover:bg-slate-800">
      <code className="min-w-0 flex-1 truncate text-xs text-slate-800 dark:text-slate-200">{command}</code>
      {copied ? <CheckIcon aria-hidden="true" className="size-4 shrink-0 text-green-500" /> : <CopyIcon aria-hidden="true" className="size-4 shrink-0" />}
    </button>
    {copied && <span role="status" className="sr-only">Command copied</span>}
    {failed && <p role="alert" className="mt-2 text-xs">Could not copy. Try again.</p>}
  </div>
}
