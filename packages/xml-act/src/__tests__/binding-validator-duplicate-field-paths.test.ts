import { describe, expect, test } from 'vitest'
import { Schema } from '@effect/schema'
import { Effect } from 'effect'
import { createXmlRuntime } from '../execution/xml-runtime'
import type { RegisteredTool } from '../types'

describe('binding validator — duplicate input field paths', () => {
  test('throws at registration time when same field is mapped in attributes and childTags', () => {
    const registeredTool: RegisteredTool = {
      tagName: 'agent-create',
      groupName: 'agent',
      tool: {
        name: 'agent-create',
        description: 'test tool',
        inputSchema: Schema.Struct({
          options: Schema.Struct({
            type: Schema.String,
          }),
        }) as Schema.Schema.Any,
        outputSchema: Schema.Void as Schema.Schema.Any,
        execute: (_input: unknown, _ctx: unknown) => Effect.void as Effect.Effect<unknown, unknown, unknown>,
        label: () => 'test',
      },
      binding: {
        tag: 'agent-create',
        attributes: [{ field: 'options.type', attr: 'type' }],
        childTags: [{ field: 'options.type', tag: 'type' }],
      },
    }

    expect(() =>
      createXmlRuntime({
        tools: new Map([['agent-create', registeredTool]]),
      })
    ).toThrow("Binding error on <agent-create>: field 'options.type' is mapped multiple times")
  })
})
