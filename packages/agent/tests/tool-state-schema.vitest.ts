import { describe, expect, it } from 'vitest'
import { Schema } from 'effect'
import { ToolHandleSchema } from '../src/models/tool-handle-schema'

const decodeToolHandle = Schema.decodeUnknownEither(ToolHandleSchema)

const baseHandle = {
  toolCallId: 'call-1',
  providerToolCallId: 'provider-call-1',
}

describe('tool state schema', () => {
  it('validates handle state against the literal tool key', () => {
    const validFileWrite = decodeToolHandle({
      ...baseHandle,
      toolKey: 'fileWrite',
      state: {
        phase: 'streaming',
        path: 'notes.txt',
        body: 'hello',
        charCount: 5,
        lineCount: 1,
        isScratchpad: false,
      },
    })

    expect(validFileWrite._tag).toBe('Right')
  })

  it('does not allow rich tools to fall through to generic base state', () => {
    const invalidFileWrite = decodeToolHandle({
      ...baseHandle,
      toolKey: 'fileWrite',
      state: {
        phase: 'streaming',
      },
    })

    expect(invalidFileWrite._tag).toBe('Left')
  })

  it('allows base state for tool keys that are registered with base state', () => {
    const validCompact = decodeToolHandle({
      ...baseHandle,
      toolKey: 'compact',
      state: {
        phase: 'streaming',
      },
    })

    expect(validCompact._tag).toBe('Right')
  })
})
