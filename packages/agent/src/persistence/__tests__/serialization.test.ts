// @ts-nocheck
import { describe, test, expect } from 'bun:test'
import type { AppEvent } from '../../events'

describe('Event Serialization', () => {
  describe('round-trip serialization', () => {
    test('serializes and deserializes session_initialized', () => {
      const event: AppEvent = {
        type: 'session_initialized',
        forkId: null,
        context: {
          cwd: '/test',
          platform: 'macos',
          shell: 'zsh',
          timezone: 'America/Los_Angeles',
          username: 'testuser',
          fullName: 'Test User',
          git: {
            branch: 'main',
            status: 'clean',
            recentCommits: 'abc123 Initial commit'
          },
          folderStructure: 'src/\n  index.ts',
          agentsFile: {
            filename: 'AGENTS.md',
            content: '# Instructions'
          },
          skills: null
        }
      }

      const serialized = JSON.stringify(event)
      const deserialized = JSON.parse(serialized) as AppEvent
      
      expect(deserialized).toEqual(event)
      expect(deserialized.type).toBe('session_initialized')
      expect(deserialized.context.platform).toBe('macos')
    })

    test('serializes and deserializes user_message', () => {
      const event: AppEvent = {
        type: 'user_message',
        forkId: null,
        content: 'Hello, world!',
        mode: 'text',
        synthetic: false, taskMode: false
      }

      const serialized = JSON.stringify(event)
      const deserialized = JSON.parse(serialized) as AppEvent
      
      expect(deserialized).toEqual(event)
    })

    test('serializes and deserializes turn_outcome', () => {
      const event: AppEvent = {
        type: 'turn_outcome',
        forkId: null,
        turnId: 'turn-123',
        chainId: 'chain-456',
        strategyId: 'xml-act',

        result: {
          _tag: 'Completed',
          completion: {
            yieldTarget: 'user',
            feedback: [],
          },
        },
        inputTokens: null,
        outputTokens: null,
        cacheReadTokens: null,
        cacheWriteTokens: null,
        providerId: null,
        modelId: null,
      }

      const serialized = JSON.stringify(event)
      const deserialized = JSON.parse(serialized) as AppEvent
      
      expect(deserialized).toEqual(event)
      expect(deserialized.outcome._tag).toBe('Completed')
    })

    test('serializes and deserializes fork events', () => {
      const events: AppEvent[] = [
        {
          type: 'fork_started', mode: 'clone' as const,
          forkId: 'fork-1',
          parentForkId: null,
          name: 'test-fork',
          context: 'Task: Do something',
        },
        {
          type: 'fork_completed',
          forkId: 'fork-1',
          parentForkId: null,
          result: { status: 'success', data: { key: 'value' } },
        }
      ]

      for (const event of events) {
        const serialized = JSON.stringify(event)
        const deserialized = JSON.parse(serialized) as AppEvent
        expect(deserialized).toEqual(event)
      }
    })

    test('serializes and deserializes tool events', () => {
      const toolEvent: AppEvent = {
        type: 'tool_event',
        forkId: null,
        turnId: 'turn-1',
        toolCallId: 'call-1',
        toolKey: 'shell',
        event: {
          _tag: 'ToolExecutionEnded',
          toolCallId: 'call-1',
          group: 'default',
          toolName: 'shell',
          result: {
            _tag: 'Success',
            output: { stdout: 'file1\nfile2', stderr: '', exitCode: 0 },
            ref: { id: 'r1', tree: { tag: 'element' as const, name: 'shell', attrs: {}, children: [] } }
          },
        },
      }

      expect(JSON.parse(JSON.stringify(toolEvent))).toEqual(toolEvent)
    })

    test('serializes and deserializes user_bash_command', () => {
      const event: AppEvent = {
        type: 'user_bash_command',
        forkId: null,
        timestamp: Date.now(),
        command: 'ls -la',
        cwd: '/tmp',
        exitCode: 0,
        stdout: 'a\nb',
        stderr: '',
      }

      const deserialized = JSON.parse(JSON.stringify(event)) as AppEvent
      expect(deserialized).toEqual(event)
    })

    test('serializes and deserializes streaming events', () => {
      const events: AppEvent[] = [
        {
          type: 'message_chunk',
          forkId: null,
          turnId: 'turn-1',
          id: 'm1',
          text: 'Hello '
        },
        {
          type: 'thinking_chunk',
          forkId: null,
          turnId: 'turn-1',
          text: 'I should...'
        },
        {
          type: 'message_end',
          forkId: null,
          turnId: 'turn-1',
          id: 'm1'
        }
      ]

      for (const event of events) {
        const deserialized = JSON.parse(JSON.stringify(event)) as AppEvent
        expect(deserialized).toEqual(event)
      }
    })
  })

  describe('special characters', () => {
    test('handles special characters in strings', () => {
      const event: AppEvent = {
        type: 'user_message',
        forkId: null,
        content: 'Special chars: " \' \\ \n \t \r < > &',
        mode: 'text',
        synthetic: false, taskMode: false
      }

      const deserialized = JSON.parse(JSON.stringify(event)) as AppEvent
      expect(deserialized.content).toBe(event.content)
    })

    test('handles unicode and emoji', () => {
      const event: AppEvent = {
        type: 'user_message',
        forkId: null,
        content: 'Unicode: 你好 مرحبا 🚀 ✨',
        mode: 'text',
        synthetic: false, taskMode: false
      }

      const deserialized = JSON.parse(JSON.stringify(event)) as AppEvent
      expect(deserialized.content).toBe(event.content)
    })

    test('handles very long strings', () => {
      const longContent = 'A'.repeat(100000)
      const event: AppEvent = {
        type: 'message_chunk',
        forkId: null,
        turnId: 'turn-1',
        id: 'm-long',
        text: longContent
      }

      const deserialized = JSON.parse(JSON.stringify(event)) as AppEvent
      expect(deserialized.text).toBe(longContent)
    })
  })

  describe('nested structures', () => {
    test('handles deeply nested tool results', () => {
      const event: AppEvent = {
        type: 'tool_event',
        forkId: null,
        turnId: 'turn-1',
        toolCallId: 'call-1',
        toolKey: 'custom',
        event: {
          _tag: 'ToolExecutionEnded',
          toolCallId: 'call-1',
          group: 'default',
          toolName: 'custom',
          result: {
            _tag: 'Success',
            output: {
              level1: {
                level2: {
                  level3: {
                    data: [1, 2, 3],
                    nested: { key: 'value' }
                  }
                }
              }
            },
            ref: { id: 'r1', tree: { tag: 'element' as const, name: 'custom', attrs: {}, children: [] } }
          },
        },
      }

      const deserialized = JSON.parse(JSON.stringify(event)) as AppEvent
      expect(deserialized).toEqual(event)
    })

    test('handles arrays in events', () => {
      const event: AppEvent = {
        type: 'task_cancelled',
        forkId: null,
        taskId: 'task-1',
        cancelledSubtree: Array.from({ length: 100 }, (_, i) => `task-${i}`),
        killedWorkers: Array.from({ length: 100 }, (_, i) => ({
          agentId: `agent-${i}`,
          forkId: `fork-${i}`,
        })),
        timestamp: Date.now(),
      }

      const deserialized = JSON.parse(JSON.stringify(event)) as AppEvent
      expect(deserialized).toEqual(event)
    })
  })

  describe('null and undefined handling', () => {
    test('preserves null values', () => {
      const event: AppEvent = {
        type: 'session_initialized',
        forkId: null,
        context: {
          cwd: '/test',
          platform: 'macos',
          shell: 'zsh',
          timezone: 'UTC',
          username: 'test',
          fullName: null,
          git: null,
          folderStructure: '',
          agentsFile: null,
          skills: null
        }
      }

      const deserialized = JSON.parse(JSON.stringify(event)) as AppEvent
      expect(deserialized.forkId).toBeNull()
      expect(deserialized.context.fullName).toBeNull()
      expect(deserialized.context.git).toBeNull()
    })

    test('handles fork events with null parentForkId', () => {
      const event: AppEvent = {
        type: 'fork_started', mode: 'clone' as const,
        forkId: 'fork-1',
        parentForkId: null,
        name: 'root-fork',
        context: 'test context',
      }

      const deserialized = JSON.parse(JSON.stringify(event)) as AppEvent
      expect(deserialized.parentForkId).toBeNull()
    })
  })

  describe('error cases', () => {
    test('handles tool error results', () => {
      const event: AppEvent = {
        type: 'tool_event',
        forkId: null,
        turnId: 'turn-1',
        toolCallId: 'call-1',
        toolKey: 'shell',
        event: {
          _tag: 'ToolExecutionEnded',
          toolCallId: 'call-1',
          group: 'default',
          toolName: 'shell',
          result: {
            _tag: 'Error',
            error: 'Command failed: permission denied'
          },
        },
      }

      const deserialized = JSON.parse(JSON.stringify(event)) as AppEvent
      expect(deserialized).toEqual(event)
    })

    test('handles turn failure results', () => {
      const event: AppEvent = {
        type: 'turn_outcome',
        forkId: null,
        turnId: 'turn-1',
        chainId: 'chain-1',
        strategyId: 'xml-act',

        result: {
          _tag: 'SystemError',
          message: 'Syntax error',
        },
        inputTokens: null,
        outputTokens: null,
        cacheReadTokens: null,
        cacheWriteTokens: null,
        providerId: null,
        modelId: null,
      }

      const deserialized = JSON.parse(JSON.stringify(event)) as AppEvent
      expect(deserialized.outcome._tag).toBe('SystemError')
      if (deserialized.outcome._tag === 'SystemError') {
        expect(deserialized.outcome.message).toBe('Syntax error')
      }
    })
  })

  describe('timestamp handling', () => {
    test('preserves numeric timestamps in JSON round-trip', () => {
      const now = Date.now()
      const event = {
        type: 'user_message',
        forkId: null,
        content: 'Test',
        mode: 'text',
        timestamp: now,
        synthetic: false, taskMode: false
      }

      const deserialized = JSON.parse(JSON.stringify(event))
      expect(deserialized.timestamp).toBe(now)
    })

    test('handles very large timestamps', () => {
      const futureTimestamp = Date.now() + 1000000000000
      const event = {
        type: 'fork_started', mode: 'clone' as const,
        forkId: 'fork-1',
        parentForkId: null,
        name: 'test',
        context: 'test context',
        timestamp: futureTimestamp
      }

      const deserialized = JSON.parse(JSON.stringify(event))
      expect(deserialized.timestamp).toBe(futureTimestamp)
    })
  })
})
