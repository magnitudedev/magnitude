import { describe, test, expect } from 'bun:test'
import {
  runTestSandbox,
  findCalls,
  findCall,
  type TestSandboxResult
} from '../../src/test-sandbox'
import {
  checkNoUnwantedEscaping,
  checkNoLiteralNewlineEscapes
} from '../../src/evals/prose/evaluator'

// =============================================================================
// checkNoUnwantedEscaping
// =============================================================================

describe('checkNoUnwantedEscaping', () => {
  test('passes for clean string', () => {
    const result = checkNoUnwantedEscaping('hello `world` ${name}', 'test')
    expect(result.passed).toBe(true)
  })

  test('fails for escaped backtick', () => {
    const result = checkNoUnwantedEscaping('hello \\`world\\`', 'test')
    expect(result.passed).toBe(false)
    expect(result.message).toContain('escaped backtick')
  })

  test('fails for escaped dollar sign', () => {
    const result = checkNoUnwantedEscaping('hello \\${name}', 'test')
    expect(result.passed).toBe(false)
    expect(result.message).toContain('escaped dollar')
  })

  test('passes for non-string values', () => {
    const result = checkNoUnwantedEscaping(42, 'test')
    expect(result.passed).toBe(true)
  })
})

// =============================================================================
// checkNoLiteralNewlineEscapes
// =============================================================================

describe('checkNoLiteralNewlineEscapes', () => {
  test('passes for string with real newlines only', () => {
    const result = checkNoLiteralNewlineEscapes('line1\nline2\nline3', 'test')
    expect(result.passed).toBe(true)
  })

  test('passes for string with no newlines at all', () => {
    const result = checkNoLiteralNewlineEscapes('no newlines here', 'test')
    expect(result.passed).toBe(true)
  })

  test('fails for string mixing real newlines and literal \\n', () => {
    const result = checkNoLiteralNewlineEscapes('line1\nline2\\nline3', 'test')
    expect(result.passed).toBe(false)
    expect(result.message).toContain('literal \\n')
  })

  test('passes for \\n inside quoted strings in code', () => {
    const result = checkNoLiteralNewlineEscapes('line1\nconst x = "\\n"\nline3', 'test')
    expect(result.passed).toBe(true)
  })

  test('passes for non-string values', () => {
    const result = checkNoLiteralNewlineEscapes(123, 'test')
    expect(result.passed).toBe(true)
  })
})

// =============================================================================
// Test Sandbox — integration tests
// =============================================================================

describe('runTestSandbox', () => {
  test('captures message() call with resolved content', async () => {
    const response = 'message(\u00ABHello, world!\u00BB);'
    const result = await runTestSandbox(response)
    expect(result.error).toBeUndefined()
    const msgs = findCalls(result, 'default.message')
    expect(msgs.length).toBeGreaterThanOrEqual(1)
    expect(msgs[0].input.content).toBe('Hello, world!')
  })

  test('captures think() call with resolved content', async () => {
    const response = 'think(\u00ABI need to figure this out.\u00BB);'
    const result = await runTestSandbox(response)
    expect(result.error).toBeUndefined()
    const thinks = findCalls(result, 'default.think')
    expect(thinks.length).toBeGreaterThanOrEqual(1)
    expect(thinks[0].input.thought).toBe('I need to figure this out.')
  })

  test('captures fs.write() with code containing template literals', async () => {
    const response = 'fs.write("src/greet.ts", \u00ABexport function greet(name: string) {\n  return `Hello, ${name}!`;\n}\u00BB);'
    const result = await runTestSandbox(response)
    expect(result.error).toBeUndefined()
    const writes = findCalls(result, 'fs.write')
    expect(writes.length).toBeGreaterThanOrEqual(1)
    expect(writes[0].input.path).toBe('src/greet.ts')
    const content = writes[0].input.content as string
    expect(content).toContain('`Hello, ${name}!`')
    expect(content).not.toContain('\\`')
    expect(content).not.toContain('\\$')
  })

  test('captures shell() call', async () => {
    const response = 'var result = shell(\u00ABfind src -name "*.ts" | wc -l\u00BB);'
    const result = await runTestSandbox(response)
    expect(result.error).toBeUndefined()
    const shells = findCalls(result, 'default.shell')
    expect(shells.length).toBeGreaterThanOrEqual(1)
    expect(shells[0].input.command).toBe('find src -name "*.ts" | wc -l')
  })

  test('captures multiple tool calls in sequence', async () => {
    const response = [
      'think(\u00ABLet me create this file.\u00BB);',
      'fs.write("test.ts", \u00ABconst x = 1;\u00BB);',
      'message(\u00ABDone!\u00BB);',
      'done();'
    ].join('\n')
    const result = await runTestSandbox(response)
    expect(result.error).toBeUndefined()
    expect(findCalls(result, 'default.think')).toHaveLength(1)
    expect(findCalls(result, 'fs.write')).toHaveLength(1)
    expect(findCalls(result, 'default.message')).toHaveLength(1)
    expect(findCalls(result, 'default.done')).toHaveLength(1)
  })

  test('handles response with backticks in prose content correctly', async () => {
    const response = 'message(\u00ABUse `const` for constants and `let` for variables.\u00BB);'
    const result = await runTestSandbox(response)
    expect(result.error).toBeUndefined()
    const msgs = findCalls(result, 'default.message')
    expect(msgs.length).toBeGreaterThanOrEqual(1)
    const content = msgs[0].input.content as string
    expect(content).toContain('`const`')
    expect(content).toContain('`let`')
    expect(content).not.toContain('\\`')
  })

  test('handles fs.write with markdown code fences', async () => {
    const mdContent = '# README\n\n```typescript\nconst greeting = `hello ${name}`;\n```\n\n```bash\nnpm install\n```'
    const response = `fs.write("README.md", \u00AB${mdContent}\u00BB);`
    const result = await runTestSandbox(response)
    expect(result.error).toBeUndefined()
    const writes = findCalls(result, 'fs.write')
    expect(writes.length).toBeGreaterThanOrEqual(1)
    const content = writes[0].input.content as string
    expect(content).toContain('```typescript')
    expect(content).toContain('```bash')
    expect(content).toContain('`hello ${name}`')
  })

  test('captures fs.edit() with content fields', async () => {
    const response = `fs.edit("src/app.ts", [
  { from: '14:b2', content: \u00AB  const timeout = 10000;\u00BB }
]);`
    const result = await runTestSandbox(response)
    expect(result.error).toBeUndefined()
    const edits = findCalls(result, 'fs.edit')
    expect(edits.length).toBeGreaterThanOrEqual(1)
    expect(edits[0].input.path).toBe('src/app.ts')
    const editOps = edits[0].input.edits as Array<{ from: string; content?: string }>
    expect(editOps[0].from).toBe('14:b2')
    expect(editOps[0].content).toBe('  const timeout = 10000;')
  })

  test('captures updateTask() with details', async () => {
    const response = `updateTask("my-task", {
  details: \u00AB## Architecture

The system uses a \`Router\` class:

\`\`\`typescript
class Router {
  handle(req: Request): Response {
    return new Response(\`Hello \${req.url}\`);
  }
}
\`\`\`
\u00BB
});`
    const result = await runTestSandbox(response)
    expect(result.error).toBeUndefined()
    const tasks = findCalls(result, 'task.updateTask')
    expect(tasks.length).toBeGreaterThanOrEqual(1)
    const updates = tasks[0].input.updates as Record<string, unknown>
    const details = updates.details as string
    expect(details).toContain('```typescript')
    expect(details).toContain('`Router`')
    expect(details).not.toContain('\\`')
  })
})

// =============================================================================
// Negative cases — detecting bad responses
// =============================================================================

describe('detecting bad responses', () => {
  test('detects escaped backticks in resolved fs.write content', async () => {
    // Simulate a model that incorrectly escapes backticks inside prose delimiters.
    // Inside « », backslash is literal, so \\` becomes literal \` in the resolved string.
    const response = 'fs.write("test.ts", \u00ABconst x = \\`hello\\`;\u00BB);'
    const result = await runTestSandbox(response)
    const writes = findCalls(result, 'fs.write')
    if (writes.length > 0) {
      const check = checkNoUnwantedEscaping(writes[0].input.content, 'test')
      expect(check.passed).toBe(false)
    }
  })
})
