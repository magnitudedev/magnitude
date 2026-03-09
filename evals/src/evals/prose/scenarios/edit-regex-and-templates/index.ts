import { readFileSync } from 'fs'
import { join } from 'path'
import type { Scenario } from '../../../../types'
import type { CapturedCall } from '../../../../test-sandbox'
import { findCall, findCalls } from '../../../../test-sandbox'
import { checkNoUnwantedEscaping, checkNoLiteralNewlineEscapes } from '../../evaluator'

const DIR = __dirname

function loadMessage(filename: string): string {
  return readFileSync(join(DIR, filename), 'utf-8')
}

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

export const editRegexAndTemplatesScenario: Scenario = {
  id: 'edit-regex-and-templates',
  description: 'Edit a file using fs.edit() to add regex validation with template literal error messages',
  messages: [
    { role: 'user', content: SESSION_CONTEXT },
    { role: 'user', content: loadMessage('user1.txt') },
    { role: 'assistant', content: loadMessage('assistant1.txt') },
    { role: 'user', content: loadMessage('results1.txt') }
  ],
  checks: [
    {
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
    },
    {
      id: 'uses-prose-delimiters',
      description: 'Raw response contains prose delimiters',
      evaluate(raw) {
        if (!raw.includes('\u00AB') || !raw.includes('\u00BB')) {
          return { passed: false, message: 'Response does not use prose delimiters' }
        }
        return { passed: true }
      }
    },
    {
      id: 'writes-file-content',
      description: 'Calls fs.edit or fs.write to modify the file',
      evaluate(raw, result) {
        const editCall = findCall(result, 'fs.edit')
        const writeCall = findCall(result, 'fs.write')
        if (!editCall && !writeCall) return { passed: false, message: 'No fs.edit or fs.write call found' }
        return { passed: true }
      }
    },
    {
      id: 'content-has-regex',
      description: 'Written content contains a regex pattern',
      evaluate(raw, result) {
        const call = findCall(result, 'fs.edit') || findCall(result, 'fs.write')
        if (!call) return { passed: false, message: 'No fs.edit or fs.write call' }
        const strings = extractStrings(call.input)
        const hasRegex = strings.some(s => s.value.includes('/^') || s.value.includes('$/'))
        if (!hasRegex) {
          return { passed: false, message: 'Content missing regex patterns' }
        }
        return { passed: true }
      }
    },
    {
      id: 'content-has-template-literals',
      description: 'Written content contains template literal interpolation',
      evaluate(raw, result) {
        const call = findCall(result, 'fs.edit') || findCall(result, 'fs.write')
        if (!call) return { passed: false, message: 'No fs.edit or fs.write call' }
        const strings = extractStrings(call.input)
        const hasInterpolation = strings.some(s => s.value.includes('${'))
        if (!hasInterpolation) {
          return { passed: false, message: 'Content missing ${} interpolation' }
        }
        return { passed: true }
      }
    },
    {
      id: 'no-escaping-in-content',
      description: 'Written content has no escaped backticks or dollar signs',
      evaluate(raw, result) {
        const call = findCall(result, 'fs.edit') || findCall(result, 'fs.write')
        if (!call) return { passed: false, message: 'No fs.edit or fs.write call' }
        return checkCallEscaping(call)
      }
    }
  ]
}
