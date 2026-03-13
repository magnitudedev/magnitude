import { Effect, Layer } from 'effect'
import { ModelResolver, makeModelResolver, makeNoopTracer } from '@magnitudedev/providers'
import { type RunnableEval, type Scenario, type ScenarioResult, type Check, type CheckResult, type EvalVariant, type ModelSpec, type ChatMessage } from '../../types'
import { getEvalProviderClient } from '../../provider-runtime'
import type { TestSandboxResult } from '../../test-sandbox'
import { callModel } from '../../runner'
import { FORMATS } from './formats'
import { SCENARIO_DEFS, type ScenarioDef } from './scenarios'
import { FAKE_TOOLS, type FakeTool } from './tools'

const escapeRegExp = (value: string): string => value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')

const TOOL_MAP = new Map(FAKE_TOOLS.map((tool) => [tool.name, tool]))

const getRequiredParams = (toolName: string): string[] =>
  (TOOL_MAP.get(toolName)?.params ?? []).filter((p) => p.required !== false).map((p) => p.name)

const hasToolCall = (raw: string, formatId: string, toolName: string): boolean => {
  const xmlToolName = toolName.replace(/\./g, '-')

  if (formatId.includes('js-act')) {
    return new RegExp(`\\b${escapeRegExp(toolName)}\\s*\\(`).test(raw)
  }

  if (formatId.includes('xml-act')) {
    return new RegExp(`<\\s*${escapeRegExp(xmlToolName)}\\b`, 'i').test(raw)
  }

  if (formatId.includes('antml')) {
    return new RegExp(`<\\s*invoke\\b[^>]*\\bname\\s*=\\s*["']${escapeRegExp(toolName)}["']`, 'i').test(raw)
  }

  if (formatId === 'openai-native') {
    return new RegExp(`\\[function_call\\]\\s*${escapeRegExp(toolName)}\\s*\\(`, 'i').test(raw)
  }

  return false
}

type DetectedCall = { toolName: string; args?: string; attrs?: string; body?: string; parsedArgs?: unknown }

type OpenAIResponsesContentItem = {
  type?: string
  text?: string
}

type OpenAIResponsesOutputItem = {
  type?: string
  name?: string
  arguments?: unknown
  content?: OpenAIResponsesContentItem[]
}

type OpenAICompletedResponse = {
  output?: OpenAIResponsesOutputItem[]
}

const detectToolCalls = (raw: string, formatId: string): DetectedCall[] => {
  const calls: DetectedCall[] = []

  if (formatId.includes('js-act')) {
    const toolAlternation = Array.from(TOOL_MAP.keys()).map(escapeRegExp).join('|')
    const regex = new RegExp(`\\b(${toolAlternation}|shell)\\s*\\(([^)]*)\\)`, 'g')
    let match: RegExpExecArray | null
    while ((match = regex.exec(raw)) !== null) {
      const toolName = match[1]
      if (TOOL_MAP.has(toolName)) {
        calls.push({ toolName, args: match[2] ?? '' })
      }
    }
    return calls
  }

  if (formatId.includes('xml-act')) {
    const xmlToolAlternation = Array.from(TOOL_MAP.keys())
      .map((name) => escapeRegExp(name.replace(/\./g, '-')))
      .join('|')
    const regex = new RegExp(`<\\s*(${xmlToolAlternation})\\b([^>]*)>([\\s\\S]*?)<\\/\\s*\\1\\s*>|<\\s*(${xmlToolAlternation})\\b([^>]*)\\s*\\/>`, 'gi')
    let match: RegExpExecArray | null
    while ((match = regex.exec(raw)) !== null) {
      const xmlToolName = match[1] ?? match[4]
      const attrs = (match[2] ?? match[5] ?? '').trim()
      const body = match[3] ?? ''
      const toolName = xmlToolName?.replace(/-/g, '.')
      if (toolName && TOOL_MAP.has(toolName)) calls.push({ toolName, attrs, body })
    }
    return calls
  }

  if (formatId.includes('antml')) {
    const paired = /<\s*invoke\b([^>]*)>([\s\S]*?)<\/\s*invoke\s*>/gi
    let match: RegExpExecArray | null
    while ((match = paired.exec(raw)) !== null) {
      const attrs = match[1] ?? ''
      const name = attrs.match(/\bname\s*=\s*["']([^"']+)["']/i)?.[1]
      if (name && TOOL_MAP.has(name)) calls.push({ toolName: name, attrs, body: match[2] ?? '' })
    }

    const selfClosing = /<\s*invoke\b([^>]*)\/>/gi
    while ((match = selfClosing.exec(raw)) !== null) {
      const attrs = match[1] ?? ''
      const name = attrs.match(/\bname\s*=\s*["']([^"']+)["']/i)?.[1]
      if (name && TOOL_MAP.has(name)) calls.push({ toolName: name, attrs, body: '' })
    }

    return calls
  }

  if (formatId === 'openai-native') {
    const regex = /\[function_call\]\s*([A-Za-z_][\w.]*)\s*\((.*)\)/g
    let match: RegExpExecArray | null
    while ((match = regex.exec(raw)) !== null) {
      const toolName = match[1]
      if (!TOOL_MAP.has(toolName)) continue
      const args = (match[2] ?? '').trim()
      let parsedArgs: unknown = undefined
      try {
        parsedArgs = JSON.parse(args)
        if (typeof parsedArgs === 'string') {
          parsedArgs = JSON.parse(parsedArgs)
        }
      } catch {
        parsedArgs = undefined
      }
      calls.push({ toolName, args, parsedArgs })
    }
  }

  return calls
}

const buildChecks = (formatId: string, def: ScenarioDef): Check[] => [
  {
    id: 'appropriate-tool-called',
    description: 'At least one acceptable tool appears as an actual tool call',
    evaluate: (rawResponse: string): CheckResult => {
      const foundCount = def.acceptableTools.filter((tool) => {
        const xmlTool = tool.replace(/\./g, '-')
        return hasToolCall(rawResponse, formatId, tool) || hasToolCall(rawResponse, formatId, xmlTool)
      }).length
      const score = def.acceptableTools.length > 0 ? foundCount / def.acceptableTools.length : 0
      const passed = score > 0
      return {
        passed,
        score,
        message: passed
          ? `Called ${foundCount}/${def.acceptableTools.length} acceptable tools`
          : `Called ${foundCount}/${def.acceptableTools.length} acceptable tools`,
      }
    },
  },
  {
    id: 'schema-adherence',
    description: 'Response adheres to expected structural format for this variant',
    evaluate: (rawResponse: string): CheckResult => {
      const trimmed = rawResponse.trim()
      let passed = false

      if (formatId.includes('js-act')) {
        const firstLine = trimmed.split('\n').map((line) => line.trim()).find(Boolean) ?? ''
        passed =
          /^(var|let|const)\b/.test(firstLine) ||
          /^[A-Za-z_][\w]*\s*\(/.test(firstLine)
      } else if (formatId === 'xml-act-inspect') {
        passed = /<\s*[A-Za-z_][\w:-]*\b[^>]*>/.test(trimmed) && /<\s*inspect\b[\s\S]*<\/\s*inspect\s*>/i.test(trimmed)
      } else if (formatId === 'xml-act-actions' || formatId === 'xml-act-actions-think') {
        passed = /<\s*actions\b[\s\S]*<\/\s*actions\s*>/i.test(trimmed)
      } else if (formatId.includes('xml-act')) {
        passed = /<\s*[A-Za-z_][\w:-]*\b[^>]*>/.test(trimmed)
      } else if (formatId.includes('antml')) {
        passed = /<\s*(function_calls|invoke)\b/i.test(trimmed)
      } else if (formatId === 'openai-native') {
        passed = /\[function_call\]/.test(trimmed)
      }

      return {
        passed,
        score: passed ? 1 : 0,
        message: passed ? 'Schema matched expected format' : `Schema mismatch for ${formatId}`,
      }
    },
  },
  {
    id: 'no-trailing-content',
    description: 'No substantive content appears after the final action boundary',
    evaluate: (rawResponse: string): CheckResult => {
      let boundary = rawResponse.length

      if (formatId === 'openai-native') {
        return { passed: true, score: 1, message: 'Structured output auto-pass' }
      }

      if (formatId.includes('js-act')) {
        const lines = rawResponse.split('\n')
        let lastActionLine = -1
        for (let i = 0; i < lines.length; i += 1) {
          const line = lines[i].trim()
          if (/;\s*$/.test(line) || /\b[A-Za-z_][\w]*\s*\([^)]*\)/.test(line)) {
            lastActionLine = i
          }
        }
        if (lastActionLine >= 0) {
          boundary = lines.slice(0, lastActionLine + 1).join('\n').length
        }
      } else if (formatId === 'xml-act-inspect') {
        const idx = rawResponse.toLowerCase().lastIndexOf('</inspect>')
        boundary = idx >= 0 ? idx + '</inspect>'.length : 0
      } else if (formatId === 'xml-act-actions' || formatId === 'xml-act-actions-think') {
        const idx = rawResponse.toLowerCase().lastIndexOf('</actions>')
        boundary = idx >= 0 ? idx + '</actions>'.length : 0
      } else if (formatId.includes('xml-act')) {
        const tags = [...rawResponse.matchAll(/<[^>]+>/g)]
        if (tags.length > 0) {
          const last = tags[tags.length - 1]
          boundary = (last.index ?? 0) + last[0].length
        } else {
          boundary = 0
        }
      } else if (formatId.includes('antml')) {
        const idx = rawResponse.toLowerCase().lastIndexOf('</function_calls>')
        boundary = idx >= 0 ? idx + '</function_calls>'.length : 0
      }

      const trailing = rawResponse.slice(boundary)
      const passed = trailing.trim().length === 0
      return {
        passed,
        score: passed ? 1 : 0,
        message: passed ? 'No trailing content found' : 'Found trailing non-whitespace content after final action boundary',
      }
    },
  },
  {
    id: 'tool-params-valid',
    description: 'Detected tool calls include all required parameters',
    evaluate: (rawResponse: string): CheckResult => {
      const calls = detectToolCalls(rawResponse, formatId)
      if (calls.length === 0) {
        return {
          passed: false,
          score: 0,
          message: 'No tool calls detected',
        }
      }

      let validCount = 0

      for (const call of calls) {
        const canonicalToolName = call.toolName.replace(/-/g, '.')
        const requiredParams = getRequiredParams(canonicalToolName)
        let valid = true

        if (formatId.includes('js-act')) {
          const argCount = (call.args ?? '')
            .split(',')
            .map((part) => part.trim())
            .filter(Boolean).length
          valid = argCount >= requiredParams.length
        } else if (formatId.includes('xml-act')) {
          const attrs = call.attrs ?? ''
          const body = call.body ?? ''
          const xmlBodyParam = TOOL_MAP.get(canonicalToolName)?.xmlBinding?.body
          valid = requiredParams.every((param) => {
            if (xmlBodyParam === param) {
              return body.trim().length > 0
            }
            const attr = new RegExp(`\\b${escapeRegExp(param)}\\s*=`, 'i').test(attrs)
            const child = new RegExp(`<\\s*${escapeRegExp(param)}\\b[^>]*>`, 'i').test(body)
            return attr || child
          })
        } else if (formatId.includes('antml')) {
          const body = call.body ?? ''
          valid = requiredParams.every((param) =>
            new RegExp(`<\\s*[A-Za-z_][\\w:-]*\\b[^>]*\\bname\\s*=\\s*["']${escapeRegExp(param)}["']`, 'i').test(body)
          )
        } else if (formatId === 'openai-native') {
          const parsed = call.parsedArgs
          valid =
            typeof parsed === 'object' &&
            parsed !== null &&
            requiredParams.every((param) => Object.prototype.hasOwnProperty.call(parsed, param))
        }

        if (valid) validCount += 1
      }

      const score = validCount / calls.length
      const passed = score === 1
      return {
        passed,
        score,
        message: passed
          ? 'All detected tool calls have required parameters'
          : `${validCount}/${calls.length} tool calls included required parameters`,
      }
    },
  },
  {
    id: 'single-turn-discipline',
    description: 'Response avoids simulated multi-turn artifacts and fake tool results',
    evaluate: (rawResponse: string): CheckResult => {
      const hasFakeResult =
        /<\s*status\s*>\s*(sent|success)\s*<\/\s*status\s*>/i.test(rawResponse) ||
        /<\s*(email_result|tool_result|result)\b[\s\S]*?(sent|success)/i.test(rawResponse)

      const hasRoleMarkers = /^\s*(User|Assistant|System|Human):\s*$/gim.test(rawResponse)

      const responseForTurnCheck = rawResponse.replace(/^\s*<\s*think\b[\s\S]*?<\s*\/\s*think\s*>\s*/i, '')
      const lines = responseForTurnCheck.split('\n')
      let proseToActionTransitions = 0
      let previousType: 'empty' | 'prose' | 'action' = 'empty'
      let xmlDepth = 0
      for (const line of lines) {
        const trimmed = line.trim()
        if (!trimmed) continue

        // Track XML nesting depth
        // Count opening tags (not self-closing)
        const openTags = trimmed.match(/<[A-Za-z_][\w:-]*(?:\s[^>]*)?>(?!.*\/>)/g) || []
        const closeTags = trimmed.match(/<\/[A-Za-z_][\w:-]*\s*>/g) || []
        const selfClosing = trimmed.match(/<[A-Za-z_][\w:-]*(?:\s[^>]*)?\s*\/>/g) || []

        // A line is 'action' if we're inside XML nesting OR the line itself is/contains XML or code
        const isXmlLine = /<\s*[A-Za-z_][\w:-]*\b/.test(trimmed) || /<\//.test(trimmed)
        const isCodeLine = /\b[A-Za-z_][\w]*\s*\([^)]*\)\s*;?$/.test(trimmed) || /^(var|let|const)\b/.test(trimmed)
        const isFunctionCall = /\[function_call\]/.test(trimmed)

        const wasInXml = xmlDepth > 0

        // Update depth: add opens, subtract closes
        xmlDepth += openTags.length
        xmlDepth -= closeTags.length
        if (xmlDepth < 0) xmlDepth = 0

        const isInXml = xmlDepth > 0 || wasInXml
        const isAction = isInXml || isXmlLine || isCodeLine || isFunctionCall

        const currentType: 'prose' | 'action' = isAction ? 'action' : 'prose'
        if (previousType === 'prose' && currentType === 'action') proseToActionTransitions += 1
        previousType = currentType
      }

      const hasMultipleCycles = proseToActionTransitions > 1
      const passed = !(hasFakeResult || hasRoleMarkers || hasMultipleCycles)

      return {
        passed,
        score: passed ? 1 : 0,
        message: passed
          ? 'No multi-turn artifacts detected'
          : 'Detected fake results, role markers, or multiple prose/action cycles',
      }
    },
  },
]

const scenarios: Scenario[] = FORMATS.flatMap((format) =>
  SCENARIO_DEFS.map((def) => {
    const messages: ChatMessage[] = [{ role: 'user', content: [def.userMessage] }]
    return {
      id: `${format.id}/${def.id}`,
      description: `[${format.label}] ${def.description}`,
      variantId: format.id,
      messages,
      checks: buildChecks(format.id, def),
    }
  })
)

const variants: EvalVariant[] = FORMATS.map((format) => ({
  id: format.id,
  label: format.label,
  count: SCENARIO_DEFS.length,
}))

const callModelOpenAINative = async (
  systemPrompt: string,
  messages: ChatMessage[],
  modelSpec: ModelSpec,
  tools: FakeTool[]
): Promise<string> => {
  const providerClient = await getEvalProviderClient()
  const auth = await providerClient.auth.getAuth(modelSpec.provider)
  await providerClient.state.setSelection('primary', modelSpec.provider, modelSpec.model, auth ?? null, { persist: false })

  const toolDefs = tools.map((tool) => ({
    type: 'function' as const,
    name: tool.name,
    description: tool.description,
    parameters: {
      type: 'object',
      properties: Object.fromEntries(
        tool.params.map((p) => [p.name, { type: p.type, description: p.description }])
      ),
      required: tool.params.filter((p) => p.required !== false).map((p) => p.name),
    },
    strict: null,
  }))

  const bound = await Effect.runPromise(
    Effect.gen(function* () {
      const runtime = yield* ModelResolver
      return yield* runtime.resolve('primary')
    }).pipe(Effect.provide(makeModelResolver().pipe(Layer.provide(providerClient.layer), Layer.provide(makeNoopTracer())))),
  )
  if (bound.connection._tag !== 'Responses') {
    throw new Error('OpenAI native format requires a Responses connection')
  }
  const endpoint = bound.connection.endpoint
  const headers = bound.connection.headers

  const body = {
    model: modelSpec.model,
    instructions: systemPrompt,
    input: messages.map((m) => ({
      role: m.role === 'assistant' ? 'assistant' : 'user',
      content: Array.isArray(m.content) ? m.content.join('') : String(m.content),
    })),
    tools: toolDefs,
    stream: true,
    store: false,
  }

  const response = await fetch(endpoint, {
    method: 'POST',
    headers,
    body: JSON.stringify(body),
  })

  if (!response.ok) {
    const errorText = await response.text()
    throw new Error(`Responses API error ${response.status}: ${errorText}`)
  }

  const reader = response.body?.getReader()
  if (!reader) {
    throw new Error('Responses API returned no stream body')
  }

  const decoder = new TextDecoder()
  let buffer = ''
  let textOutput = ''
  let completedResponse: OpenAICompletedResponse | null = null

  while (true) {
    const { done, value } = await reader.read()
    if (done) break

    buffer += decoder.decode(value, { stream: true })
    const lines = buffer.split('\n')
    buffer = lines.pop() ?? ''

    for (const line of lines) {
      if (!line.startsWith('data: ')) continue
      const data = line.slice(6)
      if (data === '[DONE]') continue

      try {
        const event = JSON.parse(data)
        if (event.type === 'response.output_text.delta') {
          textOutput += event.delta ?? ''
        } else if (event.type === 'response.completed') {
          completedResponse = event.response ?? null
        }
      } catch {
        // ignore non-json/partial events
      }
    }
  }

  const rendered: string[] = []

  if (textOutput.trim()) {
    rendered.push(textOutput.trim())
  }

  const outputItems = completedResponse?.output ?? []
  for (const item of outputItems) {
    if (item?.type === 'function_call') {
      rendered.push(`[function_call] ${item.name}(${JSON.stringify(item.arguments)})`)
    }
  }

  if (rendered.length > 0) {
    return rendered.join('\n')
  }

  // Fallback: attempt to render message/function output from completed payload
  const fallback = outputItems
    .map((item: OpenAIResponsesOutputItem) => {
      if (item?.type === 'message') {
        const text = item.content
          ?.filter((c: OpenAIResponsesContentItem) => c.type === 'output_text')
          .map((c: OpenAIResponsesContentItem) => c.text ?? '')
          .join('')
        return text?.trim() ? text : ''
      }

      if (item?.type === 'function_call') {
        return `[function_call] ${item.name}(${JSON.stringify(item.arguments)})`
      }

      return ''
    })
    .filter(Boolean)

  return fallback.join('\n')
}

const executeScenario = async (scenario: Scenario, modelSpec: ModelSpec): Promise<ScenarioResult> => {
  const [formatId, scenarioDefId] = scenario.id.split('/')
  const format = FORMATS.find((f) => f.id === formatId)
  const def = SCENARIO_DEFS.find((d) => d.id === scenarioDefId)

  if (!format || !def) {
    return {
      scenarioId: scenario.id,
      checks: Object.fromEntries(
        scenario.checks.map((check) => [
          check.id,
          { passed: false, message: `Invalid scenario id: ${scenario.id}` },
        ])
      ),
      passed: false,
      score: 0,
      rawResponse: '',
    }
  }

  let rawResponse = ''

  try {
    if (formatId === 'openai-native') {
      rawResponse = await callModelOpenAINative(
        format.buildSystemPrompt(FAKE_TOOLS),
        scenario.messages,
        modelSpec,
        FAKE_TOOLS
      )
    } else {
      const raw = await callModel(format.buildSystemPrompt(FAKE_TOOLS), scenario.messages, modelSpec)
      rawResponse = String(raw ?? '')
    }
  } catch (error) {
    const checks = Object.fromEntries(
      scenario.checks.map((check) => [
        check.id,
        {
          passed: false,
          message: `Model call failed: ${error instanceof Error ? error.message : String(error)}`,
        },
      ])
    )

    return {
      scenarioId: scenario.id,
      checks,
      passed: false,
      score: 0,
      rawResponse,
    }
  }

  const emptySandboxResult: TestSandboxResult = {
    calls: [],
    events: [],
  }
  const checks = Object.fromEntries(
    scenario.checks.map((check) => [check.id, check.evaluate(rawResponse, emptySandboxResult)])
  )
  const checkResults = Object.values(checks)
  const score = checkResults.length > 0
    ? checkResults.reduce((sum, check) => sum + (check.score ?? (check.passed ? 1 : 0)), 0) / checkResults.length
    : 0

  return {
    scenarioId: scenario.id,
    checks,
    passed: checkResults.every((result) => result.passed),
    score,
    rawResponse,
  }
}

export const formatCompareEval: RunnableEval = {
  id: 'format-compare',
  name: 'Response Format Comparison',
  description: `Compares ${FORMATS.length} response formats across ${scenarios.length} scenarios for tool-calling structure and schema adherence.`,
  scenarios,
  variants,
  defaultConcurrency: 8,
  runScenario: executeScenario,
}