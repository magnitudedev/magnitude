/**
 * Prose delimiter eval scenarios
 *
 * Each scenario tests whether the model correctly uses « » prose delimiters
 * and avoids incorrect escaping inside them. Checks are scenario-specific,
 * targeting the exact tool calls and content each scenario should produce.
 */

import type { Scenario, Check } from '../../types'
import type { TestSandboxResult, CapturedCall } from '../../test-sandbox'
import { findCalls, findCall } from '../../test-sandbox'
import { checkNoUnwantedEscaping, checkNoLiteralNewlineEscapes } from './evaluator'
import { editRegexAndTemplatesScenario } from './scenarios/edit-regex-and-templates'

// =============================================================================
// Shared session context
// =============================================================================

const SESSION_CONTEXT = `<session_context>
Username: testuser
Full name: Test User
Working directory: /home/testuser/myapp
Platform: linux
Shell: bash
Timezone: America/Los_Angeles

Git branch: main
Git status:
(clean)

Recent commits:
abc1234 initial commit

Folder structure:
src/
  index.ts
  utils/
    helpers.ts
  components/
    Button.tsx
package.json
tsconfig.json
README.md
</session_context>`

// =============================================================================
// Shared base checks
// =============================================================================

function executesCleanly(): Check {
  return {
    id: 'executes-cleanly',
    description: 'Response executes in sandbox without errors',
    evaluate(raw, result) {
      if (result.error) {
        return { passed: false, message: 'Sandbox error: ' + result.error, snippet: raw.substring(0, 200) }
      }
      if (result.calls.length === 0) {
        return { passed: false, message: 'No tool calls captured' }
      }
      return { passed: true }
    }
  }
}

function usesProseDelimiters(): Check {
  return {
    id: 'uses-prose-delimiters',
    description: 'Raw response contains \u00AB \u00BB prose delimiters',
    evaluate(raw) {
      if (!raw.includes('\u00AB') || !raw.includes('\u00BB')) {
        return { passed: false, message: 'Response does not use prose delimiters' }
      }
      return { passed: true }
    }
  }
}

// =============================================================================
// Helper: check a specific tool call's string content for escaping issues
// =============================================================================

function extractStrings(obj: unknown, prefix = ''): Array<{ path: string; value: string }> {
  const strings: Array<{ path: string; value: string }> = []
  if (typeof obj === 'string') {
    strings.push({ path: prefix, value: obj })
  } else if (Array.isArray(obj)) {
    obj.forEach((item, i) => strings.push(...extractStrings(item, `${prefix}[${i}]`)))
  } else if (obj && typeof obj === 'object') {
    for (const [key, val] of Object.entries(obj)) {
      strings.push(...extractStrings(val, prefix ? `${prefix}.${key}` : key))
    }
  }
  return strings
}

function checkCallEscaping(call: CapturedCall): { passed: boolean; message?: string; snippet?: string } {
  for (const { path, value } of extractStrings(call.input)) {
    const r = checkNoUnwantedEscaping(value, `${call.slug} ${path}`)
    if (!r.passed) return r
    const r2 = checkNoLiteralNewlineEscapes(value, `${call.slug} ${path}`)
    if (!r2.passed) return r2
  }
  return { passed: true }
}

// =============================================================================
// Scenarios
// =============================================================================

export const writeCodeFileScenario: Scenario = {
  id: 'write-code-file',
  description: 'Write a TypeScript file containing template literals and backticks',
  messages: [
    { role: 'user', content: [SESSION_CONTEXT] },
    {
      role: 'user',
      content: [`<user mode="text" at="2026-Feb-17 12:00:00">
Create a new file at src/utils/greeting.ts with these exact functions:

1. \`greet(name: string): string\` — returns a template literal greeting like \`Hello, \${name}! Welcome to the app.\`
2. \`formatUser(user: {name: string, role: string}): string\` — returns a formatted string using template literals like \`User: \${user.name} (Role: \${user.role})\`

This is a brand new file — just create it directly.
</user>`]
    }
  ],
  checks: [
    executesCleanly(),
    usesProseDelimiters(),
    {
      id: 'calls-write',
      description: 'Calls fs.write to create the file',
      evaluate(raw, result) {
        const call = findCall(result, 'fs.write')
        if (!call) return { passed: false, message: 'No fs.write call found' }
        return { passed: true }
      }
    },
    {
      id: 'content-has-template-literals',
      description: 'Written file contains template literal interpolation (${...})',
      evaluate(raw, result) {
        const call = findCall(result, 'fs.write')
        if (!call) return { passed: false, message: 'No fs.write call' }
        const content = call.input?.content as string ?? ''
        if (!content.includes('${')) {
          return { passed: false, message: 'Written content missing ${} interpolation', snippet: content.substring(0, 200) }
        }
        return { passed: true }
      }
    },
    {
      id: 'no-escaping-in-written-code',
      description: 'Written code has no escaped backticks or dollar signs',
      evaluate(raw, result) {
        const call = findCall(result, 'fs.write')
        if (!call) return { passed: false, message: 'No fs.write call' }
        return checkCallEscaping(call)
      }
    }
  ]
}

export const writeMarkdownScenario: Scenario = {
  id: 'write-markdown',
  description: 'Write a markdown file with triple-backtick code fences containing template literals',
  messages: [
    { role: 'user', content: [SESSION_CONTEXT] },
    {
      role: 'user',
      content: [`<user mode="text" at="2026-Feb-17 12:00:00">
Rewrite the README.md with a "Getting Started" section that includes code blocks.
</user>`]
    },
    {
      role: 'assistant',
      content: ['var readme = fs.read("README.md", { lines: true });\ninspect(readme);']
    },
    {
      role: 'user',
      content: [`<results>
<tool name="fs.read">
<output>1:a1|# MyApp
2:b2|
3:c3|A simple application.</output>
</tool>
<reminder>Your turn is still active. Use action tools (shell, fs, inspect, etc.) to continue working. Call done() only when you are finished and ready to yield to the user.</reminder>
</results>`]
    },
    {
      role: 'user',
      content: [`<user mode="text" at="2026-Feb-17 12:01:00">
Great, now rewrite the whole file using fs.write(). Include:
1. The existing title and description
2. A "Getting Started" section with:
   - A bash code block showing \`npm install\`
   - A TypeScript code block showing a usage example with template literals like \`const greeting = \\\`hello \\\${name}\\\`\`
   - A JSON code block showing a sample config

Write the complete file with fs.write().
</user>`]
    }
  ],
  checks: [
    executesCleanly(),
    usesProseDelimiters(),
    {
      id: 'calls-write',
      description: 'Calls fs.write to create the markdown file',
      evaluate(raw, result) {
        const call = findCall(result, 'fs.write')
        if (!call) return { passed: false, message: 'No fs.write call found' }
        return { passed: true }
      }
    },
    {
      id: 'content-has-code-fences',
      description: 'Written markdown contains triple-backtick code fences',
      evaluate(raw, result) {
        const call = findCall(result, 'fs.write')
        if (!call) return { passed: false, message: 'No fs.write call' }
        const content = call.input?.content as string ?? ''
        if (!content.includes('```')) {
          return { passed: false, message: 'Written content missing code fences (```)', snippet: content.substring(0, 300) }
        }
        return { passed: true }
      }
    },
    {
      id: 'no-escaping-in-written-markdown',
      description: 'Written markdown has no escaped backticks or dollar signs',
      evaluate(raw, result) {
        const call = findCall(result, 'fs.write')
        if (!call) return { passed: false, message: 'No fs.write call' }
        return checkCallEscaping(call)
      }
    }
  ]
}

export const editCodeFileScenario: Scenario = {
  id: 'edit-code-file',
  description: 'Edit a file using fs.edit() replacing string concatenation with template literals',
  messages: [
    { role: 'user', content: [SESSION_CONTEXT] },
    {
      role: 'user',
      content: [`<user mode="text" at="2026-Feb-17 12:00:00">
Refactor the string concatenation in helpers.ts to use template literals.
</user>`]
    },
    {
      role: 'assistant',
      content: ['var content = fs.read("src/utils/helpers.ts", { lines: true });\ninspect(content);']
    },
    {
      role: 'user',
      content: [`<results>
<tool name="fs.read">
<output>1:a1|export function formatName(first: string, last: string): string {
2:b2|  return first + " " + last;
3:c3|}
4:d4|
5:e5|export function logMessage(msg: string): void {
6:f6|  console.log("[LOG] " + msg);
7:g7|}
8:h8|
9:i9|export function buildUrl(base: string, path: string, query: string): string {
10:j0|  return base + "/" + path + "?" + query;
11:k1|}</output>
</tool>
<reminder>Your turn is still active. Use action tools (shell, fs, inspect, etc.) to continue working. Call done() only when you are finished and ready to yield to the user.</reminder>
</results>`]
    }
  ],
  checks: [
    executesCleanly(),
    usesProseDelimiters(),
    {
      id: 'calls-fs-edit',
      description: 'Calls fs.edit to modify the file',
      evaluate(raw, result) {
        const call = findCall(result, 'fs.edit')
        if (!call) return { passed: false, message: 'No fs.edit call found' }
        return { passed: true }
      }
    },
    {
      id: 'edit-content-has-template-literals',
      description: 'Edit content contains template literal interpolation (${...})',
      evaluate(raw, result) {
        const call = findCall(result, 'fs.edit')
        if (!call) return { passed: false, message: 'No fs.edit call' }
        const strings = extractStrings(call.input)
        const hasInterpolation = strings.some(s => s.value.includes('${'))
        if (!hasInterpolation) {
          return { passed: false, message: 'Edit content missing ${} interpolation' }
        }
        return { passed: true }
      }
    },
    {
      id: 'no-escaping-in-edit-content',
      description: 'Edit content has no escaped backticks or dollar signs',
      evaluate(raw, result) {
        const call = findCall(result, 'fs.edit')
        if (!call) return { passed: false, message: 'No fs.edit call' }
        return checkCallEscaping(call)
      }
    }
  ]
}

export const updateTaskMarkdownScenario: Scenario = {
  id: 'update-task-markdown',
  description: 'Update task details with markdown containing TypeScript code blocks and template literals',
  messages: [
    { role: 'user', content: [SESSION_CONTEXT] },
    {
      role: 'user',
      content: [`<agent_mode mode="plan">
You are in **plan mode**. The user has opened the task panel to plan work.
</agent_mode>`]
    },
    {
      role: 'assistant',
      content: ['startTask("feature", "add-logger", "Add structured logging utility");']
    },
    {
      role: 'user',
      content: [`<results>
<tool name="task.startTask">
<output>Task "add-logger" started in PLAN mode.</output>
</tool>
<reminder>Your turn is still active. Use action tools (shell, fs, inspect, etc.) to continue working. Call done() only when you are finished and ready to yield to the user.</reminder>
</results>`]
    },
    {
      role: 'user',
      content: [`<user mode="text" at="2026-Feb-17 12:00:00">
Write up the task details. Include a TypeScript code block showing the Logger interface with methods like \`info(message: string)\` and \`error(message: string)\`, and a usage example showing template literal log messages like \`logger.info(\\\`User \\\${userId} logged in\\\`)\`.
</user>`]
    }
  ],
  checks: [
    executesCleanly(),
    usesProseDelimiters(),
    {
      id: 'calls-update-task',
      description: 'Calls updateTask to set task details',
      evaluate(raw, result) {
        const call = findCall(result, 'task.updateTask')
        if (!call) return { passed: false, message: 'No updateTask call found' }
        return { passed: true }
      }
    },
    {
      id: 'details-has-code-block',
      description: 'Task details contain a code block with template literals',
      evaluate(raw, result) {
        const call = findCall(result, 'task.updateTask')
        if (!call) return { passed: false, message: 'No updateTask call' }
        const strings = extractStrings(call.input)
        const hasCodeFence = strings.some(s => s.value.includes('```'))
        if (!hasCodeFence) {
          return { passed: false, message: 'Task details missing code fences' }
        }
        return { passed: true }
      }
    },
    {
      id: 'no-escaping-in-task-details',
      description: 'Task details have no escaped backticks or dollar signs',
      evaluate(raw, result) {
        const call = findCall(result, 'task.updateTask')
        if (!call) return { passed: false, message: 'No updateTask call' }
        return checkCallEscaping(call)
      }
    }
  ]
}

export const shellSpecialCharsScenario: Scenario = {
  id: 'shell-special-chars',
  description: 'Run shell commands containing $, quotes, pipes, and glob patterns',
  messages: [
    { role: 'user', content: [SESSION_CONTEXT] },
    {
      role: 'user',
      content: [`<user mode="text" at="2026-Feb-17 12:00:00">
Run these shell commands for me:
1. Find all TypeScript files and count them with \`find src -name "*.ts" | wc -l\`
2. Search for TODO comments with \`grep -r "TODO" src/\`
3. Print the NODE_ENV variable with \`echo $NODE_ENV\`
</user>`]
    }
  ],
  checks: [
    executesCleanly(),
    usesProseDelimiters(),
    {
      id: 'calls-shell',
      description: 'Calls shell at least once',
      evaluate(raw, result) {
        const calls = findCalls(result, 'default.shell')
        if (calls.length === 0) return { passed: false, message: 'No shell calls found' }
        return { passed: true }
      }
    },
    {
      id: 'shell-has-pipe',
      description: 'At least one shell command contains a pipe (|)',
      evaluate(raw, result) {
        const calls = findCalls(result, 'default.shell')
        const strings = calls.flatMap(c => extractStrings(c.input))
        const hasPipe = strings.some(s => s.value.includes('|'))
        if (!hasPipe) {
          return { passed: false, message: 'No shell command contains a pipe' }
        }
        return { passed: true }
      }
    },
    {
      id: 'shell-has-dollar',
      description: 'At least one shell command contains $ (env var reference)',
      evaluate(raw, result) {
        const calls = findCalls(result, 'default.shell')
        const strings = calls.flatMap(c => extractStrings(c.input))
        const hasDollar = strings.some(s => s.value.includes('$'))
        if (!hasDollar) {
          return { passed: false, message: 'No shell command contains $' }
        }
        return { passed: true }
      }
    }
  ]
}

export const combinedWriteScenario: Scenario = {
  id: 'combined-write',
  description: 'Write a file with template literals and explain it — tests multi-statement prose usage',
  messages: [
    { role: 'user', content: [SESSION_CONTEXT] },
    {
      role: 'user',
      content: [`<user mode="text" at="2026-Feb-17 12:00:00">
Create a new file at src/utils/formatter.ts with:
1. A \`formatGreeting(name: string, time: string): string\` function that uses template literals to return something like \`Good \${time}, \${name}!\`
2. A \`formatError(code: number, message: string): string\` function that returns \`Error \${code}: \${message}\`
3. JSDoc comments with backtick-fenced code examples

This is a new file — create it directly.
</user>`]
    }
  ],
  checks: [
    executesCleanly(),
    usesProseDelimiters(),
    {
      id: 'calls-write',
      description: 'Calls fs.write to create the file',
      evaluate(raw, result) {
        const call = findCall(result, 'fs.write')
        if (!call) return { passed: false, message: 'No fs.write call found' }
        return { passed: true }
      }
    },
    {
      id: 'content-has-template-literals',
      description: 'Written file contains template literal interpolation (${...})',
      evaluate(raw, result) {
        const call = findCall(result, 'fs.write')
        if (!call) return { passed: false, message: 'No fs.write call' }
        const content = call.input?.content as string ?? ''
        if (!content.includes('${')) {
          return { passed: false, message: 'Written content missing ${} interpolation', snippet: content.substring(0, 200) }
        }
        return { passed: true }
      }
    },
    {
      id: 'no-escaping-in-written-code',
      description: 'Written code has no escaped backticks or dollar signs',
      evaluate(raw, result) {
        const call = findCall(result, 'fs.write')
        if (!call) return { passed: false, message: 'No fs.write call' }
        return checkCallEscaping(call)
      }
    }
  ]
}

export const messageWithBackticksScenario: Scenario = {
  id: 'message-with-backticks',
  description: 'Respond to user with a message containing inline code backticks',
  messages: [
    { role: 'user', content: [SESSION_CONTEXT] },
    {
      role: 'user',
      content: [`<user mode="text" at="2026-Feb-17 12:00:00">
Explain the Button component to me. Include code snippets in your explanation.
</user>`]
    },
    {
      role: 'assistant',
      content: ['var buttonContent = fs.read("src/components/Button.tsx");\ninspect(buttonContent);']
    },
    {
      role: 'user',
      content: [`<results>
<tool name="fs.read">
<output>import React from 'react';

interface ButtonProps {
  label: string;
  onClick: () => void;
  variant?: 'primary' | 'secondary';
  disabled?: boolean;
}

export function Button({ label, onClick, variant = 'primary', disabled = false }: ButtonProps) {
  const className = \`btn btn-\${variant}\${disabled ? ' btn-disabled' : ''}\`;
  return (
    <button className={className} onClick={onClick} disabled={disabled}>
      {label}
    </button>
  );
}</output>
</tool>
<reminder>Your turn is still active. Use action tools (shell, fs, inspect, etc.) to continue working. Call done() only when you are finished and ready to yield to the user.</reminder>
</results>`]
    }
  ],
  checks: [
    executesCleanly(),
    usesProseDelimiters(),
    {
      id: 'calls-message',
      description: 'Calls message() to explain the component',
      evaluate(raw, result) {
        const call = findCall(result, 'default.message')
        if (!call) return { passed: false, message: 'No message() call found' }
        return { passed: true }
      }
    }
  ]
}

export const nestedTemplateLiteralsScenario: Scenario = {
  id: 'nested-template-literals',
  description: 'Write code with nested template literals and tagged templates',
  messages: [
    { role: 'user', content: [SESSION_CONTEXT] },
    {
      role: 'user',
      content: [`<user mode="text" at="2026-Feb-17 12:00:00">
Create a new file at src/utils/sql.ts with a tagged template function called \`sql\` that takes a template literal and returns a query object. Include an example function \`getUserQuery\` that uses nested template literals like:

\`\`\`
sql\`SELECT * FROM users WHERE name = \${name} AND role = \${role}\`
\`\`\`

Also add a \`buildInsert\` function that constructs an INSERT statement using template literals with \`Object.keys(data).join(', ')\` and \`Object.values(data).map(v => \\\`'\\\${v}'\\\`).join(', ')\`.

This is a new file — create it directly.
</user>`]
    }
  ],
  checks: [
    executesCleanly(),
    usesProseDelimiters(),
    {
      id: 'calls-write',
      description: 'Calls fs.write to create the file',
      evaluate(raw, result) {
        const call = findCall(result, 'fs.write')
        if (!call) return { passed: false, message: 'No fs.write call found' }
        return { passed: true }
      }
    },
    {
      id: 'content-has-tagged-template',
      description: 'Written code contains tagged template usage (sql`...`)',
      evaluate(raw, result) {
        const call = findCall(result, 'fs.write')
        if (!call) return { passed: false, message: 'No fs.write call' }
        const content = call.input?.content as string ?? ''
        if (!content.includes('sql`')) {
          return { passed: false, message: 'Written code missing tagged template (sql`...`)', snippet: content.substring(0, 300) }
        }
        return { passed: true }
      }
    },
    {
      id: 'no-escaping-in-written-code',
      description: 'Written code has no escaped backticks or dollar signs',
      evaluate(raw, result) {
        const call = findCall(result, 'fs.write')
        if (!call) return { passed: false, message: 'No fs.write call' }
        return checkCallEscaping(call)
      }
    }
  ]
}

/**
 * Multi-line message — tests that message() content uses real newlines, not literal \n
 * This is a known issue with OpenAI models that output literal \n in prose.
 */
export const multilineMessageScenario: Scenario = {
  id: 'multiline-message',
  description: 'Respond with a detailed multi-line explanation — checks for real newlines vs literal \\n',
  messages: [
    { role: 'user', content: [SESSION_CONTEXT] },
    {
      role: 'user',
      content: [`<user mode="text" at="2026-Feb-17 12:00:00">
Give me a detailed explanation of how JavaScript promises work. Cover:
1. What promises are and why they exist
2. The three states (pending, fulfilled, rejected)
3. How .then() and .catch() chaining works
4. A code example showing async/await with error handling

Be thorough — I want a complete explanation with multiple paragraphs.
</user>`]
    }
  ],
  checks: [
    executesCleanly(),
    usesProseDelimiters(),
    {
      id: 'calls-message',
      description: 'Calls message() to respond',
      evaluate(raw, result) {
        const call = findCall(result, 'default.message')
        if (!call) return { passed: false, message: 'No message() call found' }
        return { passed: true }
      }
    },
    {
      id: 'message-has-real-newlines',
      description: 'Message content uses real newlines (not literal \\n)',
      evaluate(raw, result) {
        const call = findCall(result, 'default.message')
        if (!call) return { passed: false, message: 'No message() call' }
        const content = call.input?.content as string ?? ''
        // Must have real newlines (multi-paragraph response)
        if (!content.includes('\n')) {
          return { passed: false, message: 'Message has no real newlines — expected multi-line response', snippet: content.substring(0, 200) }
        }
        // Check for literal \n mixed with real newlines
        const r = checkNoLiteralNewlineEscapes(content, 'default.message content')
        if (!r.passed) return r
        return { passed: true }
      }
    },
    {
      id: 'message-is-multiline',
      description: 'Message contains at least 5 lines (thorough explanation)',
      evaluate(raw, result) {
        const call = findCall(result, 'default.message')
        if (!call) return { passed: false, message: 'No message() call' }
        const content = call.input?.content as string ?? ''
        const lineCount = content.split('\n').length
        if (lineCount < 5) {
          return { passed: false, message: `Message only has ${lineCount} lines, expected at least 5`, snippet: content.substring(0, 200) }
        }
        return { passed: true }
      }
    }
  ]
}

/**
 * Multi-line message after multi-file reads — reproduces the OpenAI literal \\n issue.
 * Hypothesis: after reading multiple files at once (which appear as escaped JSON strings
 * in the raw API response), OpenAI is more likely to emit literal \\n in subsequent prose.
 */
export const multilineMessageAfterFileReadsScenario: Scenario = {
  id: 'multiline-message-after-file-reads',
  description: 'Read multiple files then produce a detailed multi-line summary — reproduces OpenAI literal \\\\n issue after escaped JSON content',
  messages: [
    { role: 'user', content: [`<session_context>
Username: testuser
Full name: Test User
Working directory: /home/testuser/myapp
Platform: linux
Shell: bash
Timezone: America/Los_Angeles

Git branch: main
Git status:
(clean)

Recent commits:
abc1234 initial commit

Folder structure:
src/
  index.ts
  utils/
    helpers.ts
  components/
    Button.tsx
</session_context>`] },
    {
      role: 'user',
      content: [`<user mode="text" at="2026-Feb-17 12:00:00">
Give me a thorough summary of this codebase. Read the source files and explain what each one does, how they relate to each other, and any patterns or design decisions you notice. I want multiple paragraphs — be detailed.
</user>`]
    },
    {
      role: 'assistant',
      content: ['var indexContent = fs.read("src/index.ts");\nvar helpersContent = fs.read("src/utils/helpers.ts");\nvar buttonContent = fs.read("src/components/Button.tsx");\ninspect({ indexContent, helpersContent, buttonContent });']
    },
    {
      role: 'user',
      content: [`<results>
<tool name="fs.read">
<output>import { formatName, logMessage, buildUrl } from './utils/helpers'
import { Button } from './components/Button'

const greeting = \`Hello, \${process.env.USER ?? 'world'}!\`
logMessage(\`App started: \${greeting}\`)

const url = buildUrl('https://api.example.com', 'users', \`limit=10&offset=\${0}\`)
console.log(\`Resolved URL: \${url}\`)

export default function main() {
  const name = formatName('Jane', 'Doe')
  return { name, url, greeting }
}</output>
</tool>
<tool name="fs.read">
<output>export function formatName(first: string, last: string): string {
  return \`\${first} \${last}\`
}

export function logMessage(msg: string): void {
  const timestamp = new Date().toISOString()
  console.log(\`[\${timestamp}] \${msg}\`)
}

export function buildUrl(base: string, path: string, query: string): string {
  return \`\${base}/\${path}?\${query}\`
}

export function formatError(code: number, message: string): string {
  return \`Error \${code}: \${message}\`
}</output>
</tool>
<tool name="fs.read">
<output>import React from 'react'

interface ButtonProps {
  label: string
  onClick: () => void
  variant?: 'primary' | 'secondary'
  disabled?: boolean
}

export function Button({ label, onClick, variant = 'primary', disabled = false }: ButtonProps) {
  const className = \`btn btn-\${variant}\${disabled ? ' btn-disabled' : ''}\`
  return (
    <button className={className} onClick={onClick} disabled={disabled}>
      {label}
    </button>
  )
}</output>
</tool>
<tool name="default.inspect">
<output>{
  "indexContent": "import { formatName, logMessage, buildUrl } from './utils/helpers'\nimport { Button } from './components/Button'\n\nconst greeting = \`Hello, \${process.env.USER ?? 'world'}!\`\nlogMessage(\`App started: \${greeting}\`)\n\nconst url = buildUrl('https://api.example.com', 'users', \`limit=10&offset=\${0}\`)\nconsole.log(\`Resolved URL: \${url}\`)\n\nexport default function main() {\n  const name = formatName('Jane', 'Doe')\n  return { name, url, greeting }\n}",
  "helpersContent": "export function formatName(first: string, last: string): string {\n  return \`\${first} \${last}\`\n}\n\nexport function logMessage(msg: string): void {\n  const timestamp = new Date().toISOString()\n  console.log(\`[\${timestamp}] \${msg}\`)\n}\n\nexport function buildUrl(base: string, path: string, query: string): string {\n  return \`\${base}/\${path}?\${query}\`\n}\n\nexport function formatError(code: number, message: string): string {\n  return \`Error \${code}: \${message}\`\n}",
  "buttonContent": "import React from 'react'\n\ninterface ButtonProps {\n  label: string\n  onClick: () => void\n  variant?: 'primary' | 'secondary'\n  disabled?: boolean\n}\n\nexport function Button({ label, onClick, variant = 'primary', disabled = false }: ButtonProps) {\n  const className = \`btn btn-\${variant}\${disabled ? ' btn-disabled' : ''}\`\n  return (\n    <button className={className} onClick={onClick} disabled={disabled}>\n      {label}\n    </button>\n  )\n}"
}</output>
</tool>
<reminder>Your turn is still active. Use action tools (shell, fs, inspect, etc.) to continue working. Call done() only when you are finished and ready to yield to the user.</reminder>
</results>`]
    }
  ],
  checks: [
    executesCleanly(),
    usesProseDelimiters(),
    {
      id: 'calls-message',
      description: 'Calls message() to summarize the codebase',
      evaluate(raw, result) {
        const call = findCall(result, 'default.message')
        if (!call) return { passed: false, message: 'No message() call found' }
        return { passed: true }
      }
    },
    {
      id: 'message-has-real-newlines',
      description: 'Message content uses real newlines (not literal \\\\n) — key check for post-file-read prose',
      evaluate(raw, result) {
        const call = findCall(result, 'default.message')
        if (!call) return { passed: false, message: 'No message() call' }
        const content = call.input?.content as string ?? ''
        if (!content.includes('\n')) {
          return { passed: false, message: 'Message has no real newlines — expected multi-paragraph summary', snippet: content.substring(0, 200) }
        }
        const r = checkNoLiteralNewlineEscapes(content, 'default.message content')
        if (!r.passed) return r
        return { passed: true }
      }
    },
    {
      id: 'message-is-multiline',
      description: 'Message contains at least 5 lines (detailed summary)',
      evaluate(raw, result) {
        const call = findCall(result, 'default.message')
        if (!call) return { passed: false, message: 'No message() call' }
        const content = call.input?.content as string ?? ''
        const lineCount = content.split('\n').length
        if (lineCount < 5) {
          return { passed: false, message: `Message only has ${lineCount} lines, expected at least 5`, snippet: content.substring(0, 200) }
        }
        return { passed: true }
      }
    }
  ]
}


// =============================================================================
// All scenarios
// =============================================================================

export const ALL_SCENARIOS: Scenario[] = [
  writeCodeFileScenario,
  writeMarkdownScenario,
  editCodeFileScenario,
  updateTaskMarkdownScenario,
  shellSpecialCharsScenario,
  combinedWriteScenario,
  messageWithBackticksScenario,
  nestedTemplateLiteralsScenario,
  multilineMessageScenario,
  multilineMessageAfterFileReadsScenario,
  editRegexAndTemplatesScenario,
]
