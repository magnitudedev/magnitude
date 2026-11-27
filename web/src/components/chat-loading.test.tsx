import { renderToStaticMarkup } from "react-dom/server"
import { Atom, RegistryProvider, Result } from "@effect-atom/atom-react"
import { Option, Schema } from "effect"
import { describe, expect, it } from "vitest"
import {
  AgentClientProvider,
  selectedChat,
  type AgentClient,
  type ApplicationSession,
} from "@magnitudedev/client-common"
import { makeSetupModel } from "@magnitudedev/client-common/fixtures/model"
import { InferenceModelsResponseSchema, type LocalModelsState } from "@magnitudedev/sdk"
import { SessionProvider } from "../session"
import type { ServiceObservation } from "../service-view"
import { ChatPage } from "./chat"
import { chatModelsPending } from "./chat/served-models"

const inferenceModel = Schema.decodeUnknownSync(InferenceModelsResponseSchema)({
  object: "list",
  data: [
    {
      id: "qwen-test",
      name: "Qwen Test",
      description: "",
      created: 0,
      object: "model",
      owned_by: "local",
      context_length: 32768,
      architecture: { input_modalities: ["text"], output_modalities: ["text"] },
      top_provider: { context_length: 32768, max_completion_tokens: 32768 },
      supported_parameters: ["tools"],
      reasoning: null,
    },
  ],
}).data[0]!

const localState = (discoveryComplete: boolean, installed = true): LocalModelsState => ({
  preparation: {
    discovery: { complete: discoveryComplete, modelsFound: installed ? 1 : 0 },
    assessment: { complete: true, settledModels: 1, totalModels: 1 },
    measurement: { _tag: "Measured" },
  },
  unsupported: [],
  hidden: [],
  models: [makeSetupModel(installed)],
})

describe("chatModelsPending", () => {
  it("stays pending until servable models, local model discovery, and saved defaults settle", () => {
    expect(chatModelsPending({ servable: Result.initial(), local: Result.success(localState(true)), defaults: Result.success({}), models: [] })).toBe(true)
    expect(chatModelsPending({ servable: Result.success({ data: [inferenceModel] }), local: Result.initial(), defaults: Result.success({}), models: [inferenceModel] })).toBe(true)
    expect(chatModelsPending({ servable: Result.success({ data: [inferenceModel] }), local: Result.success(localState(true)), defaults: Result.initial(), models: [inferenceModel] })).toBe(true)
    expect(chatModelsPending({ servable: Result.success({ data: [] }), local: Result.success(localState(false)), defaults: Result.success({}), models: [] })).toBe(true)
  })

  it("waits when servable models are refetching after discovery found ready local models, without blanking on ordinary refresh", () => {
    expect(chatModelsPending({
      servable: Result.success({ data: [] }, { waiting: true }),
      local: Result.success(localState(true, true)),
      defaults: Result.success({}),
      models: [],
    })).toBe(true)
    expect(chatModelsPending({
      servable: Result.success({ data: [inferenceModel] }, { waiting: true }),
      local: Result.success(localState(true, true)),
      defaults: Result.success({}),
      models: [inferenceModel],
    })).toBe(false)
    expect(chatModelsPending({
      servable: Result.success({ data: [] }, { waiting: true }),
      local: Result.success(localState(true, false)),
      defaults: Result.success({}),
      models: [],
    })).toBe(false)
  })

  it("never stays pending when a query fails or everything has settled", () => {
    expect(chatModelsPending({ servable: Result.fail("offline"), local: Result.initial(), defaults: Result.initial(), models: [] })).toBe(false)
    expect(chatModelsPending({ servable: Result.initial(), local: Result.initial(), defaults: Result.fail("offline"), models: [] })).toBe(false)
    expect(chatModelsPending({ servable: Result.success({ data: [inferenceModel] }), local: Result.fail("offline"), defaults: Result.success({}), models: [inferenceModel] })).toBe(false)
    expect(chatModelsPending({ servable: Result.success({ data: [inferenceModel] }), local: Result.success(localState(true)), defaults: Result.success({}), models: [inferenceModel] })).toBe(false)
  })
})

describe("ChatPage startup loading state", () => {
  const renderChat = ({
    observation,
    servable = Result.initial(),
    defaults = Result.initial(),
    local = Result.initial(),
  }: {
    observation: ServiceObservation
    servable?: Result.Result<{ readonly data: readonly (typeof inferenceModel)[] }, unknown>
    defaults?: Result.Result<Record<string, never>, unknown>
    local?: Result.Result<LocalModelsState, unknown>
  }) => {
    const localModelsService = { state: Atom.make(local) }
    const client = {
      runtime: { atom: () => Atom.make(Result.success(localModelsService)) },
      Models: {
        ListServableModels: () => Atom.make({ result: servable }),
      },
      Chat: {
        ListChatModelDefaults: () => Atom.make({ result: defaults }),
        StartChat: Atom.make(Result.initial()),
        SendChatMessage: Atom.make(Result.initial()),
        RetryChatResponse: Atom.make(Result.initial()),
        StopChatResponse: Atom.make(Result.initial()),
        RenameChat: Atom.make(Result.initial()),
      },
    } as unknown as AgentClient
    const session = {
      navigate: () => {
        throw new Error("not expected during static render")
      },
    } as unknown as ApplicationSession
    return renderToStaticMarkup(
      <RegistryProvider initialValues={[[selectedChat, null]]}>
        <AgentClientProvider tag={client}>
          <SessionProvider session={session}>
            <ChatPage observation={observation} />
          </SessionProvider>
        </AgentClientProvider>
      </RegistryProvider>
    )
  }

  it("shows a non-interactive starting skeleton before the service is ready", () => {
    for (const service of [Result.initial(), Result.success({ _tag: "Starting" as const })]) {
      const html = renderChat({ observation: { service, tray: Option.none(), canRetry: true } })
      expect(html).toContain("Starting Magnitude…")
      expect(html).toContain('aria-busy="true"')
      expect(html).toContain('data-slot="skeleton"')
      expect(html).not.toMatch(/<(button|input|textarea|form)\b/)
    }
  })

  it("shows an error instead of a skeleton when service observation fails", () => {
    const html = renderChat({ observation: { service: Result.fail("boom"), tray: Option.none(), canRetry: true } })
    expect(html).toContain("Magnitude needs your attention")
    expect(html).not.toContain('data-slot="skeleton"')
  })

  it("shows a non-interactive loading skeleton after the service is ready until models and defaults settle", () => {
    const html = renderChat({
      observation: { service: Result.success({ _tag: "Ready" }), tray: Option.none(), canRetry: true },
      servable: Result.initial(),
      defaults: Result.initial(),
      local: Result.initial(),
    })
    expect(html).toContain("Loading chat…")
    expect(html).toContain('aria-busy="true"')
    expect(html).toContain('data-slot="skeleton"')
    expect(html).not.toMatch(/<(button|input|textarea|form)\b/)
  })

  it("renders the interactive composer with the selected model once startup data has settled", () => {
    const html = renderChat({
      observation: { service: Result.success({ _tag: "Ready" }), tray: Option.none(), canRetry: true },
      servable: Result.success({ data: [inferenceModel] }),
      defaults: Result.success({}),
      local: Result.success(localState(true)),
    })
    expect(html).toContain("<textarea")
    expect(html).toContain("New chat")
    expect(html).toContain("Qwen Test")
    expect(html).not.toContain('data-slot="skeleton"')
  })
})
