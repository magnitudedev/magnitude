import { CopyText } from "./chat/shared"

/** An error's full report from the engine, collapsed under Details: monospace, scrolling, with a copy button, for bug reports. */
export function FailureDetails({ text, notify = () => {} }: { text: string; notify?: (text: string) => void }) {
  return <details className="mt-1.5 text-xs" onClick={event => event.stopPropagation()}>
    <summary className="w-fit cursor-pointer rounded text-slate-500 hover:text-slate-800 focus-visible:outline-2 focus-visible:outline-blue-500 dark:text-slate-400 dark:hover:text-slate-200">Details</summary>
    <div className="relative mt-2 rounded-md border border-slate-200 bg-slate-50 dark:border-slate-750 dark:bg-slate-900">
      <pre className="max-h-48 overflow-auto whitespace-pre-wrap break-words p-3 pr-10 font-mono text-[11px] leading-5 text-slate-600 dark:text-slate-400">{text}</pre>
      <span className="absolute right-1.5 top-1.5"><CopyText text={text} label="Copy details" notify={notify} /></span>
    </div>
  </details>
}
