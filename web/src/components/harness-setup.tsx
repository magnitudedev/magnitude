import { useId, useState, type ReactNode } from "react"
import { Result, useAtomValue } from "@effect-atom/atom-react"
import { Brand, Effect } from "effect"
import { CaretDownIcon, CheckIcon, CopyIcon } from "@phosphor-icons/react"
import { HARNESS_PRIORITY, type HarnessId, type HarnessSetupPlatform, type ProviderModelId } from "@magnitudedev/sdk"
import { useAgentClient } from "@magnitudedev/client-common"
import { writeClipboardText } from "../lib/clipboard"
import { CopyCommand } from "./copy-command"
import { ErrorNotice } from "./error-notice"
import { HarnessLogo } from "./harness-logo"
import { pageLayout } from "./page-layout"
import { Button } from "./ui/button"

import { readLanguage, translate } from "../i18n"

const tr = (key: Parameters<typeof translate>[1]) => translate(readLanguage(), key)

const harnessNames: Record<Brand.Brand.Unbranded<HarnessId>, string> = {
  pi: "Pi",
  opencode: "OpenCode",
  hermes: "Hermes",
  openclaw: "OpenClaw",
  codex: "Codex",
  "claude-code": "Claude Code",
  "oh-my-pi": "Oh My Pi",
  cline: "Cline",
}

/**
 * A browser can't know where its viewer's agents run, so it never configures them in place. Each
 * harness offers a prompt to paste into an agent on the viewer's computer, which applies the setup
 * and ends by telling them how to start the harness.
 */
export function HarnessSetupList({ defaultModel, origin, remote, platform }: {
  defaultModel: ProviderModelId | undefined
  origin: string
  remote: boolean
  platform: HarnessSetupPlatform
}) {
  return <section aria-label={tr("agents")} className="mt-7">
    <p className="mb-4 text-sm text-slate-500">{tr("setupPromptInstruction")}</p>
    <div className={pageLayout.harnessGrid}>
      {HARNESS_PRIORITY.map(harness => {
        const name = harnessNames[Brand.unbranded(harness)]
        return defaultModel === undefined
          ? <article key={harness} aria-label={name} className={pageLayout.harnessCard}>
            <SetupHeader harness={harness} name={name}><Button disabled>{tr("copySetupPrompt")}</Button></SetupHeader>
          </article>
          : <SetupCard key={harness} harness={harness} name={name} model={defaultModel} origin={origin} remote={remote} platform={platform} />
      })}
    </div>
  </section>
}

function SetupHeader({ harness, name, children }: { harness: HarnessId; name: string; children: ReactNode }) {
  return <div className="flex flex-wrap items-center justify-between gap-4 md:flex-nowrap">
    <div className="flex min-w-0 items-center gap-3">
      <HarnessLogo id={harness} name={name} />
      <h3 className="text-lg font-semibold">{name}</h3>
    </div>
    <div className="ml-auto flex shrink-0 items-center gap-4">{children}</div>
  </div>
}

function SetupCard({ harness, name, model, origin, remote, platform }: {
  harness: HarnessId
  name: string
  model: ProviderModelId
  origin: string
  remote: boolean
  platform: HarnessSetupPlatform
}) {
  const client = useAgentClient()
  const setup = useAtomValue(client.Connections.DescribeHarnessSetup({ harness, model, platform, origin, remote })).result
  const [open, setOpen] = useState(false)
  const [copied, setCopied] = useState(false)
  const [failed, setFailed] = useState(false)
  const promptId = useId()
  const copy = (prompt: string) => Effect.runFork(writeClipboardText(prompt).pipe(
    Effect.tap(() => Effect.sync(() => { setCopied(true); setFailed(false) })),
    Effect.zipRight(Effect.sleep("3 seconds")),
    Effect.tap(() => Effect.sync(() => setCopied(false))),
    Effect.catchAll(() => Effect.sync(() => { setCopied(false); setFailed(true) })),
  ))
  const ready = Result.isSuccess(setup)
  return <article aria-label={name} className={pageLayout.harnessCard}>
    <SetupHeader harness={harness} name={name}>
      {!Result.isFailure(setup) && <>
        <button type="button" aria-expanded={open} aria-controls={promptId} disabled={!ready} onClick={() => setOpen(value => !value)} className="inline-flex cursor-pointer items-center gap-1 rounded text-sm font-medium text-slate-600 transition-colors hover:text-slate-900 focus-visible:text-slate-900 focus-visible:outline-2 focus-visible:outline-blue-500 disabled:cursor-default disabled:opacity-50 dark:text-slate-400 dark:hover:text-slate-100 dark:focus-visible:text-slate-100">{tr("showPrompt")}<CaretDownIcon aria-hidden="true" className={`size-3.5 transition-transform ${open ? "rotate-180" : ""}`} /></button>
        <Button aria-label={`Copy ${name} setup prompt`} disabled={!ready} onClick={() => ready && copy(setup.value.prompt)}>
          {copied ? <CheckIcon aria-hidden="true" /> : <CopyIcon aria-hidden="true" />}{copied ? tr("copied") : tr("copySetupPrompt")}
        </Button>
      </>}
    </SetupHeader>
    {Result.isFailure(setup) && <ErrorNotice title={tr("couldNotPreparePrompt")} description={tr("checkDownloadedModel")} className="mt-3" />}
    {copied && <span role="status" className="sr-only">{tr("setupPromptCopied").replace("{name}", name)}</span>}
    {failed && <ErrorNotice title={tr("couldNotCopyClipboard")} className="mt-3" />}
    {open && ready && <div id={promptId} className="mt-4"><CopyCommand multiline command={setup.value.prompt} label={tr("copyPrompt").replace("{name}", name)} /></div>}
  </article>
}
