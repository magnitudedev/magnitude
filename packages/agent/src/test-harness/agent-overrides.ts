import type { RoleId } from '../agents/role-validation'
import { clearAgentOverrides, registerAgentDefinition, getAgentDefinition } from '../agents/registry'
import { runWithGlobalAgentTestGuard } from './global-test-guard'

type AgentDefinitionForRole = ReturnType<typeof getAgentDefinition>

export type AgentOverrideMap = Partial<Record<RoleId, AgentDefinitionForRole>>

export async function withAgentOverrides<T>(
  overrides: AgentOverrideMap,
  fn: () => Promise<T> | T
): Promise<T> {
  return runWithGlobalAgentTestGuard('agent-overrides', async () => {
    try {
      for (const [roleId, definition] of Object.entries(overrides)) {
        if (!definition) continue
        registerAgentDefinition(roleId, definition)
      }

      return await fn()
    } finally {
      clearAgentOverrides()
    }
  })
}