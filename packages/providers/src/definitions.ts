export type ProviderAuthKind = "api" | "endpoint" | "none"

export interface SupportedProviderDefinition {
  readonly id: string
  readonly displayName: string
  readonly authKind: ProviderAuthKind
  readonly environmentKeys: readonly string[]
}

export const SUPPORTED_PROVIDER_DEFINITIONS: readonly SupportedProviderDefinition[] = [
  { id: "magnitude", displayName: "Magnitude", authKind: "api", environmentKeys: ["MAGNITUDE_API_KEY"] },
  { id: "llamacpp", displayName: "Llama.cpp", authKind: "endpoint", environmentKeys: [] },
  { id: "openrouter", displayName: "OpenRouter", authKind: "api", environmentKeys: ["OPENROUTER_API_KEY"] },
  { id: "vercel", displayName: "Vercel AI Gateway", authKind: "api", environmentKeys: ["AI_GATEWAY_API_KEY"] },
  { id: "deepseek", displayName: "DeepSeek API", authKind: "api", environmentKeys: ["DEEPSEEK_API_KEY"] },
  { id: "zai", displayName: "Z.AI API", authKind: "api", environmentKeys: ["ZAI_API_KEY"] },
  { id: "zai-coding-plan", displayName: "GLM Coding Plan", authKind: "api", environmentKeys: ["ZAI_API_KEY"] },
  { id: "kimi-api", displayName: "Kimi API", authKind: "api", environmentKeys: ["MOONSHOT_API_KEY"] },
  { id: "kimi-for-coding", displayName: "Kimi Code", authKind: "api", environmentKeys: ["KIMI_API_KEY"] },
]

export function getSupportedProviderDefinition(
  providerId: string,
): SupportedProviderDefinition | null {
  return SUPPORTED_PROVIDER_DEFINITIONS.find((definition) => definition.id === providerId) ?? null
}
