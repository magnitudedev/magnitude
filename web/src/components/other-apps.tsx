import { useState, type ReactNode } from "react"
import { ArrowUpRightIcon, CaretDownIcon, PlugsIcon } from "@phosphor-icons/react"
import { CopyCommand } from "./copy-command"
import { pageLayout } from "./page-layout"
import { exampleRequest } from "./example-request"

import { readLanguage, translate } from "../i18n"

const tr = (key: Parameters<typeof translate>[1]) => translate(readLanguage(), key)

const docs = "https://docs.magnitude.dev"
const inlineLink = "inline-flex items-center gap-0.5 font-medium text-slate-700 hover:underline dark:text-slate-300"

function DocsLink({ path, children }: { path: string; children: ReactNode }) {
  return <a href={`${docs}${path}`} target="_blank" rel="noreferrer" className={inlineLink}>{children}<ArrowUpRightIcon aria-hidden="true" className="size-3.5" /></a>
}

/** The inference API for other apps; from another device it describes the key that device must send. */
export function OtherApps({ origin, model, platform, remote = false, networkAccess = false, onOpenSettings }: {
  origin: string
  model: string | undefined
  platform: string
  remote?: boolean
  networkAccess?: boolean
  onOpenSettings: () => void
}) {
  const [showExample, setShowExample] = useState(false)
  const baseUrl = `${origin}/inference/v1`
  return <section aria-label={tr("otherAppsAgents")} className="mt-8">
    <h2 className="mb-4 text-sm font-medium text-slate-500">{tr("otherAppsAgents")}</h2>
    <article className={pageLayout.harnessCard}>
      <div className="flex min-w-0 items-center gap-3">
        <span className="flex size-10 shrink-0 items-center justify-center rounded-lg bg-blue-50 text-blue-600 dark:bg-slate-800 dark:text-blue-400"><PlugsIcon aria-hidden="true" className="size-5" /></span>
        <div className="min-w-0">
          <h3 className="text-lg font-semibold">{tr("openAiApi")}</h3>
          <p className="mt-1 text-sm text-slate-500">{remote ? tr("servedByComputer") : tr("servedOnComputer")} <DocsLink path="/integrations/other-agents">{tr("learnMore")}</DocsLink></p>
        </div>
      </div>
      <div className="mt-4 border-t border-slate-200 pt-4 dark:border-slate-750">
        <div className="mb-2 flex items-baseline justify-between gap-4 text-xs text-slate-500"><span className="font-medium text-slate-700 dark:text-slate-300">{tr("baseUrlLabel")}</span><span>{remote ? tr("sendNetworkKey") : tr("anyApiKeyWorks")}</span></div>
        <CopyCommand command={baseUrl} label={tr("copyBaseUrl")} />
        <div className="mt-2 flex flex-wrap items-center justify-between gap-x-4 gap-y-1 text-xs text-slate-500">
          <p>{tr("anthropicAppsHint")}</p>
          <button type="button" aria-expanded={showExample} aria-controls="other-apps-example" onClick={() => setShowExample(value => !value)} className="inline-flex cursor-pointer items-center gap-1 font-medium text-slate-600 transition-colors hover:text-slate-900 dark:text-slate-400 dark:hover:text-slate-100">{tr("exampleRequest")}<CaretDownIcon aria-hidden="true" className={`size-3.5 transition-transform ${showExample ? "rotate-180" : ""}`} /></button>
        </div>
        {showExample && <div id="other-apps-example" className="mt-3"><CopyCommand multiline command={exampleRequest(baseUrl, model ?? "MODEL_ID", platform, remote)} label={tr("copyExampleRequest")} /></div>}
      </div>
      <p className="mt-4 border-t border-slate-200 pt-4 text-sm text-slate-500 dark:border-slate-750">{remote ? "tr("remoteCopyKeyIntro") : networkAccess ? "tr("otherDeviceKeyIntro") : "tr("turnOnNetworkIntro")}<button type="button" onClick={onOpenSettings} className={`${inlineLink} cursor-pointer`}>{tr("settings")}</button>.</p>
    </article>
  </section>
}
