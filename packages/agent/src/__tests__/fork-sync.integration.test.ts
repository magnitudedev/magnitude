// @ts-nocheck
/**
 * forkSync Integration Test
 * 
 * Tests that forkSync properly blocks and returns parsed results.
 */

import { describe, test, expect } from 'bun:test'
import { Effect } from 'effect'
import { Agent } from '@magnitudedev/event-core'
import { CodingAgent } from '../coding-agent'

describe('forkSync integration', () => {
  test('blocks and returns parsed result with schema', async () => {
    const client = await CodingAgent.createClient()
    
    const events: any[] = []
    client.onEvent((event) => events.push(event))
    
    // Send code that uses forkSync with a schema
    await client.send({
      type: 'user_message',
      forkId: null,
      content: `
var result = forkSync('test-fork', {
  prompt: 'Return a test result',
  outputSchema: {
    type: 'object',
    properties: {
      success: { type: 'boolean' },
      message: { type: 'string' }
    },
    required: ['success', 'message']
  }
});

message(\`Got result: \${JSON.stringify(result)}\`);
      `.trim(),
      mode: 'text' as const,
      synthetic: false,
      taskMode: false
    })
    
    // Wait for the fork to be created
    await new Promise(resolve => setTimeout(resolve, 100))
    
    // Find the fork_started event
    const forkStarted = events.find(e => e.type === 'fork_started' && e.blocking === true)
    expect(forkStarted).toBeDefined()
    expect(forkStarted?.outputSchema).toBeDefined()
    
    // Simulate the fork submitting a result
    await client.send({
      type: 'user_message',
      forkId: forkStarted.forkId,
      content: `
submit('{"success": true, "message": "Test passed"}');
      `.trim(),
      mode: 'text' as const,
      synthetic: false,
      taskMode: false
    })
    
    // Wait for processing
    await new Promise(resolve => setTimeout(resolve, 200))
    
    // Check that the parent received the parsed result
    const textChunks = events.filter(e => e.type === 'message_chunk' && e.forkId === null)
    const fullText = textChunks.map(e => e.text).join('')
    
    expect(fullText).toContain('Got result:')
    expect(fullText).toContain('"success":true')
    expect(fullText).toContain('"message":"Test passed"')
    
    // Verify fork completed
    const forkCompleted = events.find(e => 
      e.type === 'fork_completed' && 
      e.forkId === forkStarted.forkId
    )
    expect(forkCompleted).toBeDefined()
    expect(forkCompleted?.result).toEqual({
      success: true,
      message: 'Test passed'
    })
    
    await client.dispose()
  }, { timeout: 10000 })
  
  test('works without schema (raw string result)', async () => {
    const client = await CodingAgent.createClient()
    
    const events: any[] = []
    client.onEvent((event) => events.push(event))
    
    // Send code that uses forkSync without a schema
    await client.send({
      type: 'user_message',
      forkId: null,
      content: `
var result = forkSync('test-fork-raw', {
  prompt: 'Return a simple message'
});

message(\`Got: \${result}\`);
      `.trim(),
      mode: 'text' as const,
      synthetic: false,
      taskMode: false
    })
    
    // Wait for the fork to be created
    await new Promise(resolve => setTimeout(resolve, 100))
    
    // Find the fork_started event
    const forkStarted = events.find(e => e.type === 'fork_started' && e.blocking === true)
    expect(forkStarted).toBeDefined()
    
    // Simulate the fork submitting a result
    await client.send({
      type: 'user_message',
      forkId: forkStarted.forkId,
      content: `
submit('This is a raw string result');
      `.trim(),
      mode: 'text' as const,
      synthetic: false,
      taskMode: false
    })
    
    // Wait for processing
    await new Promise(resolve => setTimeout(resolve, 200))
    
    // Check that the parent received the raw string
    const textChunks = events.filter(e => e.type === 'message_chunk' && e.forkId === null)
    const fullText = textChunks.map(e => e.text).join('')
    
    expect(fullText).toContain('Got: This is a raw string result')
    
    await client.dispose()
  }, { timeout: 10000 })
})
