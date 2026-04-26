export interface ModelsDevModel {
  id: string
  name: string
  family?: string
  tool_call: boolean
  reasoning: boolean
  attachment?: boolean
  temperature?: boolean
  release_date?: string
  status?: string
  cost?: { input: number; output: number; cache_read?: number; cache_write?: number }
  limit?: { context?: number; output?: number; input?: number }
  modalities?: { input?: string[]; output?: string[] }
}

export interface ModelsDevProvider {
  id: string
  name: string
  env: string[]
  npm?: string
  api?: string
  models: Record<string, ModelsDevModel>
}

export type ModelsDevResponse = Record<string, ModelsDevProvider>

export interface OpenRouterModel {
  id: string
  name: string
  description?: string
  context_length: number
  pricing: {
    prompt: string
    completion: string
    image?: string
    request?: string
    web_search?: string
    internal_reasoning?: string
    input_cache_read?: string
    input_cache_write?: string
  }
  top_provider?: {
    context_length?: number | null
    max_completion_tokens?: number | null
  } | null
  architecture?: {
    input_modalities?: string[]
    output_modalities?: string[]
  } | null
  supported_parameters?: string[]
  created: number
  per_request_limits?: Record<string, unknown> | null
  canonical_slug?: string
  hugging_face_id?: string
  hugging_face_name?: string
  provider?: Record<string, unknown>
  tier?: string
  aliases?: string[]
  disabled?: boolean
  endpoint?: Record<string, unknown>
  moderation?: boolean
  deprecated?: boolean
  expiration_date?: string | null
}

export interface OpenRouterResponse {
  data: OpenRouterModel[]
}

