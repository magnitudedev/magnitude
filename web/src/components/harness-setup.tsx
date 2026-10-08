import { useState } from "react"
import { Result, useAtomValue } from "@effect-atom/atom-react"
import { Brand, Effect } from "effect"
import { CheckIcon, CopyIcon } from "@phosphor-icons/react"
import { HARNESS_PRIORITY, type HarnessId, type HarnessSetupPlatform, type ProviderModelId } from "@magnitudedev/sdk"
import { useAgentClient } from "@magnitudedev/client-common"
import { writeClipboardText } from "../lib/clipboard"
import { ErrorNotice } from "./error-notice"
import { HarnessLogo } from "./harness-logo"
import { pageLayout } from "./page-layout"
import { Button } from "./ui/button"

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
  return <section aria-label="Agents" className="mt-7">
    <p className="mb-4 text-sm text-slate-500">Copy an agent’s setup prompt and paste it into that agent, or any coding agent, on the computer you’re using now.</p>
    <div className={pageLayout.harnessGrid}>
      {HARNESS_PRIORITY.map(harness => {
        const name = harnessNames[Brand.unbranded(harness)]
        return <article key={harness} aria-label={name} className={pageLayout.harnessCard}>
          <div className="flex flex-wrap items-center justify-between gap-4 md:flex-nowrap">
            <div className="flex min-w-0 items-center gap-3">
              <HarnessLogo id={harness} name={name} />
              <h3 className="text-lg font-semibold">{name}</h3>
            </div>
            <div className="ml-auto shrink-0">
              {defaultModel === undefined
                ? <Button disabled>Copy setup prompt</Button>
                : <CopySetupPrompt harness={harness} name={name} model={defaultModel} origin={origin} remote={remote} platform={platform} />}
            </div>
          </div>
        </article>
      })}
    </div>
  </section>
}

function CopySetupPrompt({ harness, name, model, origin, remote, platform }: {
  harness: HarnessId
  name: string
  model: ProviderModelId
  origin: string
  remote: boolean
  platform: HarnessSetupPlatform
}) {
  const client = useAgentClient()
  const setup = useAtomValue(client.Connections.DescribeHarnessSetup({ harness, model, platform, origin, remote })).result
  const [copied, setCopied] = useState(false)
  const [failed, setFailed] = useState(false)
  const copy = (prompt: string) => Effect.runFork(writeClipboardText(prompt).pipe(
    Effect.tap(() => Effect.sync(() => { setCopied(true); setFailed(false) })),
    Effect.zipRight(Effect.sleep("3 seconds")),
    Effect.tap(() => Effect.sync(() => setCopied(false))),
    Effect.catchAll(() => Effect.sync(() => { setCopied(false); setFailed(true) })),
  ))
  if (Result.isFailure(setup)) return <ErrorNotice title="Couldn’t prepare the setup prompt" description="Check that a model is still downloaded, then try again." />
  return <>
    <Button aria-label={`Copy ${name} setup prompt`} disabled={!Result.isSuccess(setup)} onClick={() => Result.isSuccess(setup) && copy(setup.value.prompt)}>
      {copied ? <CheckIcon aria-hidden="true" /> : <CopyIcon aria-hidden="true" />}{copied ? "Copied" : "Copy setup prompt"}
    </Button>
    {copied && <span role="status" className="sr-only">{name} setup prompt copied</span>}
    {failed && <ErrorNotice title="Couldn’t copy to the clipboard" className="mt-2" />}
  </>
}
