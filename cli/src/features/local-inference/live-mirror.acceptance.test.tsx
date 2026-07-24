import { act } from "react"
import { testRender } from "@opentui/react/test-utils"
import { RegistryProvider, Result } from "@effect-atom/atom-react"
import { Cause, Option } from "effect"
import { expect, test, vi } from "vitest"
import {
  AgentClientProvider,
  createAgentClient,
  useLocalInferenceQuery,
} from "@magnitudedev/client-common"
import { protocolLayer } from "@magnitudedev/sdk"
import { LocalInferenceStatusBar } from "./status-bar"

vi.mock("../../hooks/use-theme", () => ({
  useTheme: () => ({
    primary: "blue", secondary: "gray", info: "cyan", link: "blue",
    foreground: "white", muted: "gray", border: "gray", warning: "magenta",
  }),
}))

const acnUrl = Option.fromNullable(process.env.LIVE_ACN_URL)

test.skipIf(Option.isNone(acnUrl))("live independent mirrors compose into the local inference view", async () => {
  const rendered: string[] = []
  const Probe = () => {
    const result = useLocalInferenceQuery()
    if (!Result.isSuccess(result)) {
      rendered.push(Result.isFailure(result) ? Cause.pretty(result.cause) : result._tag)
      return <text>mirror:{result._tag}</text>
    }
    rendered.push("success")
    return (
      <box style={{ flexDirection: "column" }}>
        <text>mirror:success</text>
        <LocalInferenceStatusBar state={result.value} width={100} selectedModelName="Qwen Test" selectedProviderId={null} onOpenModels={() => {}} onOpenHardware={() => {}} />
      </box>
    )
  }

  const url = Option.getOrThrow(acnUrl)
  const agentClient = createAgentClient(protocolLayer(url))
  const view = await testRender(
    <RegistryProvider defaultIdleTTL={5_000}>
      <AgentClientProvider tag={agentClient}>
        <Probe />
      </AgentClientProvider>
    </RegistryProvider>,
    { width: 110, height: 8 },
  )
  try {
    const deadline = Date.now() + 15_000
    while (!rendered.includes("success") && Date.now() < deadline) {
      await act(view.renderOnce)
      await new Promise<void>((resolve) => setTimeout(resolve, 25))
    }
    expect(rendered).toContain("success")
    expect(view.captureCharFrame()).toContain("mirror:success")
  } finally {
    await act(async () => view.renderer.destroy())
  }
})
