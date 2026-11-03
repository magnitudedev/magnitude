import { useMemo, useState } from "react"
import { Atom, Result, useAtomSet } from "@effect-atom/atom-react"
import { Option } from "effect"
import { ArrowRightIcon, StackIcon } from "@phosphor-icons/react"
import { installedLocalModels, localModelProviderModelId, localModelServes, useLocalInferenceHardware, useLocalModels } from "@magnitudedev/client-common"
import type { LocalModel } from "@magnitudedev/sdk"
import { useSession } from "../session"
import { Button } from "./ui/button"
import { SegmentedControl } from "./ui/segmented-control"
import { CopyCommand } from "./copy-command"
import { ErrorNotice } from "./error-notice"
import { ModelCard } from "./model-card"
import { BaseUrl, DocsLink } from "./other-apps"
import { ModelCardsSkeleton } from "./page-skeletons"
import { pageLayout } from "./page-layout"
import { decisionsExample, decisionsPython, exampleRequest, systemOneExample, systemOnePython } from "./example-request"

type Format = "systemone" | "decisions"
type Language = "shell" | "python"
const formats = [{ value: "systemone", label: "System One" }, { value: "decisions", label: "OpenAI Decisions" }] as const

/** The example for a format, language and shell: the HTTP request, or the format's Python SDK. */
export const decisionExample = (origin: string, model: string, format: Format, language: Language, platform: string, remote: boolean): string => {
  const apiRoot = `${origin}/inference`
  const baseUrl = `${apiRoot}/v1`
  if (language === "python") return format === "systemone" ? systemOnePython(apiRoot, model, remote) : decisionsPython(baseUrl, model, remote)
  return exampleRequest(baseUrl, format === "systemone" ? systemOneExample(model) : decisionsExample(model), platform, remote)
}

/**
 * What decisions are, how to call them, and the decision models on this computer. Browsing opens
 * Discover's catalog on decision models.
 */
export function DecisionsPage({ origin, platform, remote, onOpenSettings }: {
  origin: Option.Option<string>
  platform: string
  remote: boolean
  onOpenSettings: () => void
}) {
  const session = useSession()
  const browse = useAtomSet(useMemo(() => Atom.fn((_: void) => session.browseDecisionModels), [session]))
  const local = useLocalModels()
  const hardware = useLocalInferenceHardware()
  const [format, setFormat] = useState<Format>("systemone")
  const [language, setLanguage] = useState<Language>("shell")
  const installed: readonly LocalModel[] = Result.isSuccess(local) ? installedLocalModels(local.value).filter(model => Option.contains(localModelServes(model), "decisions")) : []
  const exampleModel = installed.flatMap(model => Option.toArray(localModelProviderModelId(model)))[0] ?? "MODEL_ID"
  const browseButton = (primary: boolean) => <Button variant={primary ? "default" : "outline"} onClick={() => browse()}><StackIcon />Browse decision models{primary && <ArrowRightIcon />}</Button>
  return <>
    <div className="mt-3 max-w-3xl space-y-2 text-sm text-slate-600 dark:text-slate-400">
      <p>A decision model answers typed questions about a state you send: text, JSON or images. Each answer is a probability over a fixed set of answers, read from a single pass over the input with no generated text, so it comes back in milliseconds and its confidence is calibrated.</p>
      <p>Use one where an app needs a quick, reliable judgment rather than prose: routing requests, moderating content, extracting fields, judging outputs and checking images. <DocsLink path="/api/decisions">Learn more</DocsLink></p>
    </div>

    <section aria-label="Your decision models" className="mt-8">
      <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
        <h2 className="text-sm font-medium text-slate-500">Your decision models</h2>
        {installed.length > 0 && browseButton(false)}
      </div>
      {Result.isFailure(local) ? <ErrorNotice title="Couldn’t load your models" description="Model information is unavailable. Check the service in Settings." />
        : !Result.isSuccess(local) ? <ModelCardsSkeleton library />
        : installed.length === 0 ? <div className={`${pageLayout.harnessCard} flex flex-wrap items-center justify-between gap-4`}>
          <p className="text-sm text-slate-500">No decision models on this computer yet. Download one from the catalog to answer decisions here.</p>
          {browseButton(true)}
        </div>
        : <div className="grid items-start gap-3">{installed.map(model => <div key={model.modelId}>
          <ModelCard model={model} models={local.value.models} hardware={Result.value(hardware)} page="models" />
          {Option.match(localModelProviderModelId(model), {
            onNone: () => null,
            onSome: id => <div className="mt-2 flex items-center gap-3 px-1 text-xs text-slate-500"><span className="shrink-0 font-medium text-slate-700 dark:text-slate-300">Model ID</span><CopyCommand command={id} label={`Copy model ID ${id}`} /></div>,
          })}
        </div>)}</div>}
    </section>

    {Option.isSome(origin) && <section aria-label="Call the decisions API" className="mt-8">
      <h2 className="mb-4 text-sm font-medium text-slate-500">Call the API</h2>
      <article className={pageLayout.harnessCard}>
        <p className="mb-4 text-sm text-slate-500">Two formats share one base URL: TypeSafe System One at <code>/systemone</code>, with questions keyed by ID, and OpenAI Decisions at <code>/decisions</code>, with questions in order. Both give the same probabilities for the same request.</p>
        <BaseUrl baseUrl={`${origin.value}/inference/v1`} remote={remote} />
        <div className="mt-4 flex flex-wrap items-center justify-between gap-3">
          <SegmentedControl label="Request format" options={formats} value={format} onChange={setFormat} />
          <SegmentedControl label="Example language" options={[{ value: "shell", label: platform === "win32" ? "PowerShell" : "curl" }, { value: "python", label: "Python" }]} value={language} onChange={setLanguage} />
        </div>
        <div className="mt-3"><CopyCommand key={`${format}-${language}`} multiline command={decisionExample(origin.value, exampleModel, format, language, platform, remote)} label="Copy example request" /></div>
        <p className="mt-4 border-t border-slate-200 pt-4 text-sm text-slate-500 dark:border-slate-750">{remote ? "Copy the key from Network access in " : "To call it from another device, use the address and API key under Network access in "}<button type="button" onClick={onOpenSettings} className="inline-flex cursor-pointer items-center font-medium text-slate-700 hover:underline dark:text-slate-300">Settings</button>.</p>
      </article>
    </section>}
  </>
}
