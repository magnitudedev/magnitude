import { Effect, Layer } from 'effect'
import { AgentModelResolver } from '../model/model-resolver'
import { makeAgentBoundModel } from '../model/agent-model'
import { createTestBoundModel, type TestModelConfig } from './test-model'
import type { RoleId } from '../agents/role-validation'
import { ROLE_TO_SLOT } from '@magnitudedev/roles'
import type { ModelProfile } from '@magnitudedev/ai'

const DEFAULT_TEST_PROFILE: ModelProfile = {
  contextWindow: 200_000,
  maxOutputTokens: 32_768,
  capabilities: { vision: true, toolCalls: true, structuredOutput: true, grammar: true, toolChoiceModes: ['auto', 'none', 'required', 'named'] },
}

export function makeTestModelResolver(config: TestModelConfig = {}): Layer.Layer<AgentModelResolver> {
  const bound = createTestBoundModel(config)
  const makeTestAgentModel = (modelId: string, slotId: 'primary' | 'secondary', roleId: RoleId | null = null) =>
    makeAgentBoundModel({
      rawModel: bound,
      modelSource: { slotId },
      modelId,
      profile: DEFAULT_TEST_PROFILE,
      debug: false,
      agentId: 'test',
      roleId,
    })

  return Layer.succeed(AgentModelResolver, {
    resolvePrimary: (roleId: RoleId) =>
      Effect.succeed(makeTestAgentModel('test-model', ROLE_TO_SLOT[roleId], roleId)),
    resolveSecondary: () =>
      Effect.succeed(makeTestAgentModel('test-secondary', 'secondary')),
  })
}
