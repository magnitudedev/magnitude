import type { DesktopHarnessConnection, HarnessId } from "@magnitudedev/client-common"
import { Brand } from "effect"
import { ArrowUpRight } from "lucide-react"
import { Button } from "../../web/src/components/ui/button"
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
  onDisconnect: (id: HarnessId) => void
}

export function HarnessConnections({ connections, busy, canConnect, onConnect, onDisconnect }: Props) {
  return <div className="mt-7 space-y-8">{[true, false].map(installed => {
    const rows = connections.filter(row => row.installed === installed)
      .sort((a, b) => Number(b.inspection._tag === "Connected") - Number(a.inspection._tag === "Connected"))
    if (rows.length === 0) return null
    const title = installed ? "Installed on your machine" : "Not installed"
    return <section key={title} aria-label={title}>
      <h2 className="mb-4 text-sm font-medium text-slate-500">{title}</h2>
      <div className="grid items-start gap-5 xl:grid-cols-2">{rows.map(row => <article key={row.id} aria-label={row.name} className="rounded-xl border border-slate-200 bg-white p-5 dark:border-slate-750 dark:bg-slate-850">
        <div className="flex items-center justify-between gap-4">
          <div className="flex min-w-0 items-center gap-3">
            <HarnessLogo id={row.id} name={row.name} />
            <div><h3 className="text-lg font-semibold">{row.name}</h3>
              <p className={`mt-1 flex items-center gap-2 text-sm ${installed && row.inspection._tag === "Connected" ? "text-green-600 dark:text-green-400" : "text-slate-500"}`}>
                {installed && <span aria-hidden="true" className={`size-2 shrink-0 rounded-full ${row.inspection._tag === "Connected" ? "bg-green-600 dark:bg-green-400" : "bg-slate-400"}`} />}
                {!installed ? "Not installed" : row.inspection._tag === "Connected" ? "Connected" : row.inspection._tag === "Unavailable" ? "Could not verify connection" : "Not connected"}
              </p>
              {installed && row.inspection._tag === "Unavailable" && <p className="mt-1 text-xs text-slate-500">{row.inspection.reason}</p>}
            </div>
          </div>
          <div className="ml-auto flex shrink-0 flex-wrap items-center justify-end gap-3">
            {installed && row.id === "pi" && row.plugin._tag === "Some" && <span className="text-sm text-slate-500">Includes <a href="https://pi.dev/packages/@magnitudedev/pi-extension" target="_blank" rel="noreferrer" className="inline-flex items-center gap-1 hover:underline">Pi extension<ArrowUpRight aria-hidden="true" className="size-3.5" /></a></span>}
            {installed ? <><Button disabled={busy || !canConnect} onClick={() => onConnect(row.id)}>{row.inspection._tag === "Connected" ? "Reconnect" : "Connect"}</Button>{row.managed && row.inspection._tag === "Connected" && <Button variant="outline" disabled={busy} onClick={() => onDisconnect(row.id)}>Disconnect</Button>}</>
              : <a href={installationDocs[Brand.unbranded(row.id)]} target="_blank" rel="noreferrer" className="inline-flex items-center gap-1 text-sm text-slate-500 hover:underline">Install {row.name}<ArrowUpRight aria-hidden="true" className="size-4" /></a>}
          </div>
        </div>
        {installed && row.inspection._tag === "Connected" && <div className="mt-4 border-t border-slate-200 pt-4 text-sm text-slate-500 dark:border-slate-750"><p>Configuration files</p><ul className="mt-2 space-y-1">{row.configurationFiles.map(file => <li key={file} className="break-all font-mono text-xs">{file}</li>)}</ul></div>}

      </article>)}</div>
    </section>
  })}</div>
}
