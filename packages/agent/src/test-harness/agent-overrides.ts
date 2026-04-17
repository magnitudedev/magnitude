import type { AgentVariant } from '../agents/variants'
import { clearAgentOverrides, registerAgentDefinition, getAgentDefinition } from '../agents/registry'
import { runWithGlobalAgentTestGuard } from './global-test-guard'

type AgentDefinitionForVariant = ReturnType<typeof getAgentDefinition>

export type AgentOverrideMap = Partial<Record<AgentVariant, AgentDefinitionForVariant>>

export async function withAgentOverrides<T>(
  overrides: AgentOverrideMap,
  fn: () => Promise<T> | T
): Promise<T> {
  return runWithGlobalAgentTestGuard('agent-overrides', async () => {
    try {
      for (const [variant, definition] of Object.entries(overrides)) {
        if (!definition) continue
        registerAgentDefinition(variant, definition)
      }

      return await fn()
    } finally {
      clearAgentOverrides()
    }
  })
}