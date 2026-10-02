import { useState, type ReactNode } from "react"
import { ArrowUpRightIcon, CaretDownIcon, PlugsIcon } from "@phosphor-icons/react"
import { CopyCommand } from "./copy-command"
import { pageLayout } from "./page-layout"
import { exampleRequest } from "./example-request"

const docs = "https://docs.magnitude.dev"
const inlineLink = "inline-flex items-center gap-0.5 font-medium text-slate-700 hover:underline dark:text-slate-300"

function DocsLink({ path, children }: { path: string; children: ReactNode }) {
  return <a href={`${docs}${path}`} target="_blank" rel="noreferrer" className={inlineLink}>{children}<ArrowUpRightIcon aria-hidden="true" className="size-3.5" /></a>
}

export function OtherApps({ origin, model, platform, onOpenSettings }: {
  origin: string
  model: string | undefined
  platform: string
  onOpenSettings: () => void
}) {
  const [showExample, setShowExample] = useState(false)
  const baseUrl = `${origin}/inference/v1`
  return <section aria-label="Other apps and agents" className="mt-8">
    <h2 className="mb-4 text-sm font-medium text-slate-500">Other apps and agents</h2>
    <article className={pageLayout.harnessCard}>
      <div className="flex min-w-0 items-center gap-3">
        <span className="flex size-10 shrink-0 items-center justify-center rounded-lg bg-blue-50 text-blue-600 dark:bg-slate-800 dark:text-blue-400"><PlugsIcon aria-hidden="true" className="size-5" /></span>
        <div className="min-w-0">
          <h3 className="text-lg font-semibold">OpenAI-compatible API</h3>
          <p className="mt-1 text-sm text-slate-500">Served on your computer. Any app or agent can use it. <DocsLink path="/integrations/other-agents">Learn more</DocsLink></p>
        </div>
      </div>
      <div className="mt-4 border-t border-slate-200 pt-4 dark:border-slate-750">
        <div className="mb-2 flex items-baseline justify-between gap-4 text-xs text-slate-500"><span className="font-medium text-slate-700 dark:text-slate-300">Base URL</span><span>Any API key works</span></div>
        <CopyCommand command={baseUrl} label="Copy base URL" />
        <div className="mt-2 flex flex-wrap items-center justify-between gap-x-4 gap-y-1 text-xs text-slate-500">
          <p>Anthropic-compatible apps use <code>/inference/anthropic</code> instead.</p>
          <button type="button" aria-expanded={showExample} aria-controls="other-apps-example" onClick={() => setShowExample(value => !value)} className="inline-flex cursor-pointer items-center gap-1 font-medium text-slate-600 transition-colors hover:text-slate-900 dark:text-slate-400 dark:hover:text-slate-100">Example request<CaretDownIcon aria-hidden="true" className={`size-3.5 transition-transform ${showExample ? "rotate-180" : ""}`} /></button>
        </div>
        {showExample && <div id="other-apps-example" className="mt-3"><CopyCommand multiline command={exampleRequest(baseUrl, model ?? "MODEL_ID", platform)} label="Copy example request" /></div>}
      </div>
      <p className="mt-4 border-t border-slate-200 pt-4 text-sm text-slate-500 dark:border-slate-750">To use it from another device, turn on network access in <button type="button" onClick={onOpenSettings} className={`${inlineLink} cursor-pointer`}>Settings</button>.</p>
    </article>
  </section>
}
