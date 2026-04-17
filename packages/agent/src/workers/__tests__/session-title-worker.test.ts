import { describe, expect, it } from 'bun:test'
import { Agent } from '@magnitudedev/event-core'
import { Effect } from 'effect'
import type { AppEvent } from '../../events'
import { SessionContextProjection } from '../../projections/session-context'
import { AgentStatusProjection } from '../../projections/agent-status'
import { TaskGraphProjection } from '../../projections/task-graph'
import { SessionTitleWorker } from '../session-title-worker'
import {
  InMemoryChatPersistenceTag,
  makeInMemoryChatPersistenceLayer,
} from '../../test-harness/in-memory-persistence'

const TestAgent = Agent.define<AppEvent>()({
  name: 'SessionTitleWorkerTestAgent',
  projections: [
    SessionContextProjection,
    AgentStatusProjection,
    TaskGraphProjection,
  ],
  workers: [SessionTitleWorker],
})

describe('SessionTitleWorker', () => {
  it('persists the root task title into session metadata on task creation', async () => {
    const client = await TestAgent.createClient(
      makeInMemoryChatPersistenceLayer({
        metadata: { chatName: 'New Chat' },
      })
    )

    try {
      await client.send({
        type: 'session_initialized',
        forkId: null,
        context: {
          cwd: process.cwd(),
          workspacePath: '/tmp/test-workspace',
          platform: 'macos',
          shell: '/bin/zsh',
          timezone: 'UTC',
          username: 'tester',
          fullName: null,
          git: null,
          folderStructure: '.',
          agentsFile: null,
          skills: null,
        },
      })

      await client.send({
        type: 'task_created',
        forkId: null,
        taskId: 'root-task',
        parentId: null,
        title: 'Initial task title',

        timestamp: 1,
      })

      const metadata = await client.runEffect(
        Effect.flatMap(InMemoryChatPersistenceTag, (persistence) => persistence.inspectMetadata())
      )
      expect(metadata.chatName).toBe('Initial task title')
    } finally {
      await client.dispose()
    }
  })
})
