import { pageLayout } from "./page-layout"
import type { DesktopHarnessConnection, HarnessId } from "@magnitudedev/client-common"
import { Brand, Effect, Fiber } from "effect"
import { Atom, useAtomMount } from "@effect-atom/atom-react"
import { useMemo, useRef, useState } from "react"
import type { ProviderModelId } from "@magnitudedev/sdk"
import { harnessCommand } from "./harness-command"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "../../web/src/components/ui/select"
import { ArrowClockwiseIcon, ArrowUpRightIcon, CopyIcon, CheckIcon } from "@phosphor-icons/react"
import { Button } from "../../web/src/components/ui/button"
import { ActionTooltip, TooltipProvider } from "../../web/src/components/ui/tooltip"
import { HarnessLogo } from "./harness-logo"

const installationDocs: Record<Brand.Brand.Unbranded<HarnessId>, string> = {
  pi: "https://github.com/badlogic/pi-mono/tree/main/packages/coding-agent#quick-start",
  opencode: "https://opencode.ai/docs/#install",
  hermes: "https://hermes-agent.nousresearch.com/docs/getting-started/installation/",
  openclaw: "https://docs.openclaw.ai/install",
  codex: "https://developers.openai.com/codex/cli",
  "claude-code": "https://code.claude.com/docs/en/overview",
  "oh-my-pi": "https://github.com/can1357/oh-my-pi#installation",
  cline: "https://docs.cline.bot/getting-started/installing-cline",
}

type Props = {
  connections: readonly DesktopHarnessConnection[]
  busy: boolean
  canConnect: boolean
  onConnect: (id: HarnessId) => void
  models: readonly CommandModel[]
  defaultModel: ProviderModelId | undefined
  platform: string
  onDisconnect: (id: HarnessId) => void
}

export function HarnessConnections({ connections, busy, canConnect, onConnect, onDisconnect, models, defaultModel, platform }: Props) {
  return <TooltipProvider><div className="mt-7 space-y-8">{[true, false].map(installed => {
    const rows = connections.filter(row => row.installed === installed)
      .sort((a, b) => Number(b.inspection._tag === "Connected") - Number(a.inspection._tag === "Connected"))
    if (rows.length === 0) return null
    const title = installed ? "Installed on your machine" : "Not installed"
    return <section key={title} aria-label={title}>
      <h2 className="mb-4 text-sm font-medium text-slate-500">{title}</h2>
      <div className={pageLayout.harnessGrid}>{rows.map(row => {
        const needsAttention = row.managed && row.inspection._tag === "Disconnected"
        return <article key={row.id} aria-label={row.name} className={pageLayout.harnessCard}>
        <div className="flex items-center justify-between gap-4">
          <div className="flex min-w-0 items-center gap-3">
            <HarnessLogo id={row.id} name={row.name} />
            <div><h3 className="text-lg font-semibold">{row.name}</h3>
              <p className={`mt-1 flex items-center gap-2 text-sm ${installed && row.inspection._tag === "Connected" ? "text-green-600 dark:text-green-400" : needsAttention ? "text-orange-600 dark:text-orange-400" : "text-slate-500"}`}>
                {installed && <span aria-hidden="true" className={`size-2 shrink-0 rounded-full ${row.inspection._tag === "Connected" ? "bg-green-600 dark:bg-green-400" : needsAttention ? "bg-orange-500" : "bg-slate-400"}`} />}
                {!installed ? "Not installed" : row.inspection._tag === "Connected" ? "Connected" : row.inspection._tag === "Unavailable" ? "Could not verify connection" : needsAttention ? "Connection needs repair" : "Not connected"}
              </p>
              {installed && row.inspection._tag === "Unavailable" && <p className="mt-1 text-xs text-slate-500">{row.inspection.reason}</p>}
            </div>
          </div>
          <div className="ml-auto flex shrink-0 flex-wrap items-center justify-end gap-3">
            {installed && row.id === "pi" && row.plugin._tag === "Some" && <span className="text-sm text-slate-500">Includes <a href="https://pi.dev/packages/@magnitudedev/pi-extension" target="_blank" rel="noreferrer" className="inline-flex items-center gap-1 hover:underline">Pi extension<ArrowUpRightIcon aria-hidden="true" className="size-3.5" /></a></span>}
            {installed ? <>{row.inspection._tag === "Connected"
              ? <ActionTooltip label="Refresh connection" trigger={<Button variant="ghost" size="icon-sm" aria-label="Refresh connection" disabled={busy || !canConnect} onClick={() => onConnect(row.id)}><ArrowClockwiseIcon aria-hidden="true" className="size-4" /></Button>} />
              : <Button disabled={busy || !canConnect} onClick={() => onConnect(row.id)}>{needsAttention ? "Repair connection" : "Connect"}</Button>}{!needsAttention && (row.managed || row.inspection._tag === "Connected") && <Button variant="outline" disabled={busy} onClick={() => onDisconnect(row.id)}>Disconnect</Button>}</>
              : <a href={installationDocs[Brand.unbranded(row.id)]} target="_blank" rel="noreferrer" className="inline-flex items-center gap-1 text-sm text-slate-500 hover:underline">Install {row.name}<ArrowUpRightIcon aria-hidden="true" className="size-4" /></a>}
          </div>
        </div>
        {installed && row.inspection._tag === "Connected" && <div className="mt-4 border-t border-slate-200 pt-4 text-sm text-slate-500 dark:border-slate-750"><HarnessCommand harness={row.id} name={row.name} models={models} defaultModel={defaultModel} platform={platform} /><div className="relative mt-3 text-xs"><details className="group"><summary className="w-fit cursor-pointer list-none hover:text-slate-700 dark:hover:text-slate-300 [&::-webkit-details-marker]:hidden"><span aria-hidden="true" className="mr-1 inline-block transition-transform group-open:rotate-90">▸</span>Configuration files</summary><ul className="mt-2 space-y-1">{row.configurationFiles.map(file => <li key={file} className="break-all font-mono text-xs">{file}</li>)}</ul></details>{models.length > 0 && <span className="absolute right-0 top-0 max-w-[calc(100%-9rem)] truncate text-right text-slate-500">Run this in {platform === "win32" ? "PowerShell" : "your terminal"} from your project folder.</span>}</div></div>}

      </article>})}</div>
    </section>
  })}</div></TooltipProvider>
}


export interface CommandModel { readonly id: ProviderModelId; readonly label: string }

function CopyCommand({ command, label }: { command: string; label: string }) {
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

export function HarnessCommand({ harness, name, models, defaultModel, platform }: { harness: HarnessId; name: string; models: readonly CommandModel[]; defaultModel: ProviderModelId | undefined; platform: string }) {
  const [selection, setSelection] = useState<string | null>(null)
  const selected = models.find(model => model.id === selection) ?? models.find(model => model.id === defaultModel) ?? models[0]
  if (!selected) return <p>Download a compatible model to get a command for this agent.</p>
  const command = harnessCommand(harness, selected.id, platform)
  return <div className="space-y-2">
    <div className="flex min-w-0 items-center gap-3"><div className="flex min-w-0 max-w-[45%] shrink-0 items-center gap-2"><Select items={models.map(model => ({ value: model.id, label: model.label }))} value={selected.id} onValueChange={setSelection}><SelectTrigger variant="inline" aria-label={`${name} model`} className="min-w-0 max-w-full py-1 text-slate-800 dark:text-slate-200"><SelectValue className="min-w-0 truncate" /></SelectTrigger><SelectContent className="w-max min-w-64 max-w-[calc(100vw-2rem)]">{models.map(model => <SelectItem key={model.id} value={model.id}>{model.label}</SelectItem>)}</SelectContent></Select></div>
    <CopyCommand key={command} command={command} label={`Copy ${name} command`} />
    </div>
    {harness === "openclaw" && <><p className="text-xs">With your gateway running, open the TUI, then enter this inside it to select the model for the current session:</p><CopyCommand key={selected.id} command={`/model magnitude/${selected.id}`} label="Copy OpenClaw model command" /></>}
  </div>
}
