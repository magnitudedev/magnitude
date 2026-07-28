import { act } from "react"
import { testRender } from "@opentui/react/test-utils"
import { RegistryProvider, Result } from "@effect-atom/atom-react"
import { Option } from "effect"
import { expect, test, vi } from "vitest"
import {
  AgentClientProvider,
  createAgentClient,
  useLocalModels,
  useModelSlots,
} from "@magnitudedev/client-common"
import { PRIMARY_SLOT_ID, ProviderIdSchema, protocolLayer } from "@magnitudedev/sdk"
import { deriveLocalInferenceFooterView } from "./footer-status"

const LOCAL_PROVIDER_ID = ProviderIdSchema.make("local")

vi.mock("../../hooks/use-theme", () => ({
  useTheme: () => ({
    primary: "blue", secondary: "gray", info: "cyan", link: "blue",
    foreground: "white", muted: "gray", border: "gray", warning: "magenta",
  }),
}))

const acnUrl = Option.fromNullable(process.env.LIVE_ACN_URL)

test.skipIf(Option.isNone(acnUrl))("live independent mirrors remain independently observable", async () => {
  const rendered: string[] = []
  const Probe = () => {
    const models = useLocalModels()
    const slots = useModelSlots()
    if (!Result.isSuccess(models) || !Result.isSuccess(slots)) {
      const status = `${models._tag}/${slots._tag}`
      rendered.push(status)
      return <text>mirrors:{status}</text>
    }
    rendered.push("success")
    const footer = deriveLocalInferenceFooterView(
      models.value,
      slots.value,
      null,
      LOCAL_PROVIDER_ID,
      PRIMARY_SLOT_ID,
    )
    return (
      <box style={{ flexDirection: "column" }}>
        <text>mirror:success</text>
        <text>{footer.memoryLabel ?? "memory:unavailable"}</text>
      </box>
    )
  }

  const url = Option.getOrUndefined(acnUrl)
  if (url === undefined) return
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
