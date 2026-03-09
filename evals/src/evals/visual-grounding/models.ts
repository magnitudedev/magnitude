/**
 * Hardcoded model list for visual grounding eval.
 *
 * Each model family tested through one provider:
 * - Anthropic models → anthropic
 * - OpenAI models → openai
 * - Google models → google
 * - Everything else → openrouter
 *
 * Edit this list to add/remove models.
 */

import type { ModelSpec } from '../../types'

export const VISUAL_GROUNDING_MODELS: ModelSpec[] = [
  // ─── Anthropic (direct) ───────────────────────────────────────
  { provider: 'anthropic', model: 'claude-opus-4-6', label: 'anthropic:claude-opus-4-6' },
  { provider: 'anthropic', model: 'claude-sonnet-4-6', label: 'anthropic:claude-sonnet-4-6' },
  { provider: 'anthropic', model: 'claude-opus-4-5', label: 'anthropic:claude-opus-4-5' },
  { provider: 'anthropic', model: 'claude-sonnet-4-5', label: 'anthropic:claude-sonnet-4-5' },
  { provider: 'anthropic', model: 'claude-haiku-4-5', label: 'anthropic:claude-haiku-4-5' },

  // ─── OpenAI (direct) ─────────────────────────────────────────
  { provider: 'openai', model: 'gpt-5.3-codex', label: 'openai:gpt-5.3-codex' },
  { provider: 'openai', model: 'gpt-5.3-codex-spark', label: 'openai:gpt-5.3-codex-spark' },
  { provider: 'openai', model: 'gpt-5.2-codex', label: 'openai:gpt-5.2-codex' },
  { provider: 'openai', model: 'gpt-5.2', label: 'openai:gpt-5.2' },
  { provider: 'openai', model: 'gpt-5.1', label: 'openai:gpt-5.1' },

  // ─── Google (direct) ─────────────────────────────────────────
  { provider: 'google', model: 'gemini-3.1-pro-preview', label: 'google:gemini-3.1-pro-preview' },
  { provider: 'google', model: 'gemini-3-pro-preview', label: 'google:gemini-3-pro-preview' },
  { provider: 'google', model: 'gemini-3-flash-preview', label: 'google:gemini-3-flash-preview' },
  { provider: 'google', model: 'gemini-2.5-flash-lite', label: 'google:gemini-2.5-flash-lite' },

  // ─── OpenRouter: Qwen ────────────────────────────────────────
  { provider: 'openrouter', model: 'qwen/qwen3.5-397b-a17b', label: 'openrouter:qwen/qwen3.5-397b-a17b' },
  { provider: 'openrouter', model: 'qwen/qwen3.5-122b-a10b', label: 'openrouter:qwen/qwen3.5-122b-a10b' },
  { provider: 'openrouter', model: 'qwen/qwen3.5-35b-a3b', label: 'openrouter:qwen/qwen3.5-35b-a3b' },
  { provider: 'openrouter', model: 'qwen/qwen3.5-27b', label: 'openrouter:qwen/qwen3.5-27b' },
  { provider: 'openrouter', model: 'qwen/qwen3-max-thinking', label: 'openrouter:qwen/qwen3-max-thinking' },
  { provider: 'openrouter', model: 'qwen/qwen3.5-plus-02-15', label: 'openrouter:qwen/qwen3.5-plus-02-15' },
  { provider: 'openrouter', model: 'qwen/qwen3.5-flash-02-23', label: 'openrouter:qwen/qwen3.5-flash-02-23' },
  { provider: 'openrouter', model: 'qwen/qwen3-coder-next', label: 'openrouter:qwen/qwen3-coder-next' },

  // ─── OpenRouter: Kimi (Moonshot) ─────────────────────────────
  { provider: 'openrouter', model: 'moonshotai/kimi-k2.5', label: 'openrouter:moonshotai/kimi-k2.5' },
  { provider: 'openrouter', model: 'moonshotai/kimi-k2-thinking', label: 'openrouter:moonshotai/kimi-k2-thinking' },

  // ─── OpenRouter: DeepSeek ────────────────────────────────────
  { provider: 'openrouter', model: 'deepseek/deepseek-v3.2', label: 'openrouter:deepseek/deepseek-v3.2' },
  { provider: 'openrouter', model: 'deepseek/deepseek-v3.2-speciale', label: 'openrouter:deepseek/deepseek-v3.2-speciale' },

  // ─── OpenRouter: MiniMax ─────────────────────────────────────
  { provider: 'openrouter', model: 'minimax/minimax-m2.5', label: 'openrouter:minimax/minimax-m2.5' },
  { provider: 'openrouter', model: 'minimax/minimax-m2.1', label: 'openrouter:minimax/minimax-m2.1' },

  // ─── OpenRouter: GLM (ZhipuAI) ──────────────────────────────
  { provider: 'openrouter', model: 'z-ai/glm-5', label: 'openrouter:z-ai/glm-5' },
  { provider: 'openrouter', model: 'z-ai/glm-4.7', label: 'openrouter:z-ai/glm-4.7' },
  { provider: 'openrouter', model: 'z-ai/glm-4.6v', label: 'openrouter:z-ai/glm-4.6v' },
  { provider: 'openrouter', model: 'z-ai/glm-4.7-flash', label: 'openrouter:z-ai/glm-4.7-flash' },

  // ─── OpenRouter: Grok (xAI) ─────────────────────────────────
  { provider: 'openrouter', model: 'x-ai/grok-4', label: 'openrouter:x-ai/grok-4' },
  { provider: 'openrouter', model: 'x-ai/grok-4.1-fast', label: 'openrouter:x-ai/grok-4.1-fast' },

  // ─── OpenRouter: GPT Open-Source ─────────────────────────────
  { provider: 'openrouter', model: 'openai/gpt-oss-120b', label: 'openrouter:openai/gpt-oss-120b' },
  { provider: 'openrouter', model: 'openai/gpt-oss-20b', label: 'openrouter:openai/gpt-oss-20b' },

  // ─── OpenRouter: Mistral ─────────────────────────────────────
  { provider: 'openrouter', model: 'mistralai/mistral-large-2512', label: 'openrouter:mistralai/mistral-large-2512' },
  { provider: 'openrouter', model: 'mistralai/devstral-2512', label: 'openrouter:mistralai/devstral-2512' },

  // ─── OpenRouter: Arcee ─────────────────────────────────────────
  { provider: 'openrouter', model: 'arcee-ai/trinity-large-preview:free', label: 'openrouter:arcee-ai/trinity-large-preview:free' },
]
