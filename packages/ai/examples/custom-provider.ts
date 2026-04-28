import {
  bindModel,
  classifyGenericError,
  execute,
  PromptBuilder,
  type ProviderDefinition,
  type ProviderModel,
  type ResolvedAuth,
} from "../src/index.js"

const customModel: ProviderModel = {
  id: "my-model",
  providerId: "my-provider",
  providerName: "My Custom Provider",
  canonicalModelId: null,
  name: "My Model",
  contextWindow: 128_000,
  maxContextTokens: null,
  maxOutputTokens: 4_096,
  supportsToolCalls: true,
  supportsReasoning: false,
  supportsVision: false,
  costs: null,
}

const myProvider: ProviderDefinition = {
  id: "my-provider",
  name: "My Custom Provider",
  family: "cloud",
  defaultBaseUrl: "https://api.myprovider.com/v1",
  authMethods: [{ type: "api-key", envKeys: ["MY_PROVIDER_KEY"] }],
  codecId: "native-chat-completions",
  classifyError: classifyGenericError,
  models: [customModel],
}

const auth: ResolvedAuth = {
  _tag: "ApiKeyAuth",
  apiKey: process.env.MY_PROVIDER_KEY ?? "replace-me",
}

const prompt = PromptBuilder.empty()
  .system("You are a helpful assistant.")
  .user("Say hello from a custom provider.")
  .build()

const bound = bindModel(myProvider, customModel, auth)

const stream = execute(bound, prompt, [], {})

// This file is intentionally illustrative. To actually run it, provide an
// HTTP client layer and consume the returned Effect/Stream in your runtime.
void stream
