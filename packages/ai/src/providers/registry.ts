import type { ProviderDefinition } from "../lib/execution/provider-definition"
import { anthropicProvider } from "./anthropic"
import { cerebrasProvider } from "./cerebras"
import { deepseekProvider } from "./deepseek"
import { fireworksAiProvider } from "./fireworks-ai"
import { kimiForCodingProvider } from "./kimi-for-coding"
import { llamaCppProvider } from "./llama.cpp"
import { lmstudioProvider } from "./lmstudio"
import { magnitudeProvider } from "./magnitude"
import { minimaxProvider } from "./minimax"
import { moonshotAiProvider } from "./moonshotai"
import { openaiProvider } from "./openai"
import { openAiCompatibleLocalProvider } from "./openai-compatible-local"
import { ollamaProvider } from "./ollama"
import { openrouterProvider } from "./openrouter"
import { vercelProvider } from "./vercel"
import { zaiCodingPlanProvider } from "./zai-coding-plan"
import { zaiProvider } from "./zai"

const providers: readonly ProviderDefinition[] = [
  magnitudeProvider,
  anthropicProvider,
  openaiProvider,
  openrouterProvider,
  vercelProvider,
  deepseekProvider,
  minimaxProvider,
  zaiProvider,
  zaiCodingPlanProvider,
  moonshotAiProvider,
  kimiForCodingProvider,
  cerebrasProvider,
  fireworksAiProvider,
  lmstudioProvider,
  ollamaProvider,
  llamaCppProvider,
  openAiCompatibleLocalProvider,
]

export function getProvider(id: string): ProviderDefinition | undefined {
  return providers.find((provider) => provider.id === id)
}

export function getAllProviders(): readonly ProviderDefinition[] {
  return providers
}

export { providers }
