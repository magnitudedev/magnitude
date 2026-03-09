/**
 * XPath 3.1 eval — tests LLM ability to write XPath 3.1 queries
 * for both XML navigation and JSON filtering.
 *
 * Two guidance variants:
 * - guided: system prompt includes XPath 3.1 syntax reference + examples
 * - unguided: just "write an XPath 3.1 expression", no teaching
 */

import type { RunnableEval, Scenario, ScenarioResult, CheckResult, ModelSpec } from '../../types'
import { callModel } from '../../runner'
import { ALL_SCENARIOS, type XPathScenario } from './scenarios'
import { evaluateQuery } from './evaluator'

// =============================================================================
// System prompts
// =============================================================================

const GUIDED_SYSTEM_PROMPT = `You are an XPath 3.1 query writer. You will be given structured data (XML, JSON, or both) and a question about what to extract from it.

Respond with ONLY a valid XPath 3.1 expression — no explanation, no markdown, no code fences. Just the raw query.

XPath 3.1 key features you may need:
- Standard XML navigation: /root/child, //descendant, @attribute, [predicate]
- Functions: count(), string(), number(), text()
- Map lookup operator: ?key (e.g. $var?fieldName)
- Array wildcard: ?* to iterate all items in an array
- Filter predicates on maps: ?*[?field = 'value']
- parse-json() to convert a JSON string (e.g. from XML text content) into a map/array
- Variables are referenced with $ prefix: $data, $result

Examples of XPath 3.1 JSON queries:
- $data?name — get the "name" field from a map variable
- $data?users?*?name — get all "name" fields from an array of user objects
- $data?users?*[?role = 'admin']?name — filter array, get names of admins
- parse-json(/root/body)?key — parse JSON from XML element, lookup key`

const UNGUIDED_SYSTEM_PROMPT = `You are an XPath 3.1 query writer. You will be given structured data (XML, JSON, or both) and a question about what to extract from it.

Respond with ONLY a valid XPath 3.1 expression — no explanation, no markdown, no code fences. Just the raw query.`

type GuidanceLevel = 'guided' | 'unguided'

const SYSTEM_PROMPTS: Record<GuidanceLevel, string> = {
  guided: GUIDED_SYSTEM_PROMPT,
  unguided: UNGUIDED_SYSTEM_PROMPT,
}

// =============================================================================
// Build prompt for a scenario
// =============================================================================

function buildUserPrompt(scenario: XPathScenario): string {
  const parts: string[] = []

  if (scenario.xml) {
    parts.push('XML document:')
    parts.push('```xml')
    parts.push(scenario.xml)
    parts.push('```')
  }

  if (scenario.variables) {
    parts.push('Available variables:')
    for (const [name, value] of Object.entries(scenario.variables)) {
      parts.push(`$${name} = ${JSON.stringify(value, null, 2)}`)
    }
  }

  parts.push('')
  parts.push(`Question: ${scenario.question}`)
  parts.push('')
  parts.push('Respond with ONLY the XPath 3.1 expression.')

  return parts.join('\n')
}

// =============================================================================
// Build framework scenarios — each XPathScenario x each guidance level
// =============================================================================

function toFrameworkScenario(s: XPathScenario, guidance: GuidanceLevel): Scenario {
  return {
    id: `${guidance}/${s.id}`,
    description: `[${guidance}] ${s.description}`,
    messages: [
      { role: 'user', content: [buildUserPrompt(s)] },
    ],
    checks: [
      {
        id: 'query-valid',
        description: 'LLM produced a non-empty response',
        evaluate(raw) {
          const trimmed = raw.trim()
          if (!trimmed) return { passed: false, message: 'Empty response' }
          return { passed: true }
        },
      },
      {
        id: 'query-correct',
        description: 'XPath query produces the expected result',
        evaluate() {
          return { passed: false, message: 'Should not be called directly' }
        },
      },
    ],
  }
}

const GUIDANCE_LEVELS: GuidanceLevel[] = ['guided', 'unguided']

const scenarios = GUIDANCE_LEVELS.flatMap(guidance =>
  ALL_SCENARIOS.map(s => toFrameworkScenario(s, guidance))
)

// =============================================================================
// Eval definition
// =============================================================================

export const xpathEval: RunnableEval = {
  id: 'xpath',
  name: 'XPath 3.1 Query Writing',
  description: 'Tests LLM ability to write correct XPath 3.1 queries for XML and JSON data',
  scenarios,
  variants: GUIDANCE_LEVELS.map(g => ({
    id: g,
    label: g === 'guided' ? 'With syntax guidance' : 'Without syntax guidance',
    count: ALL_SCENARIOS.length,
  })),

  async runScenario(scenario: Scenario, modelSpec: ModelSpec): Promise<ScenarioResult> {
    // Parse guidance level from scenario ID: "guided/xml/basic-attr" → "guided"
    const slashIdx = scenario.id.indexOf('/')
    const guidance = scenario.id.slice(0, slashIdx) as GuidanceLevel
    const xpathId = scenario.id.slice(slashIdx + 1)

    const xpathScenario = ALL_SCENARIOS.find(s => s.id === xpathId)
    if (!xpathScenario) {
      return {
        scenarioId: scenario.id,
        checks: { 'query-valid': { passed: false, message: 'Scenario not found' } },
        passed: false,
        score: 0,
        rawResponse: '',
      }
    }

    const systemPrompt = SYSTEM_PROMPTS[guidance]

    // Call the LLM
    const raw = await callModel(systemPrompt, scenario.messages, modelSpec)
    const query = extractQuery(raw)

    const checks: Record<string, CheckResult> = {}

    // Check 1: valid response
    if (!query) {
      checks['query-valid'] = { passed: false, message: 'Empty or unparseable response', snippet: raw.substring(0, 200) }
      checks['query-correct'] = { passed: false, message: 'No query to evaluate' }
      return {
        scenarioId: scenario.id,
        checks,
        passed: false,
        score: 0,
        rawResponse: raw,
      }
    }

    checks['query-valid'] = { passed: true }

    // Check 2: evaluate the query
    const result = evaluateQuery(query, xpathScenario)

    if (result.error) {
      checks['query-correct'] = {
        passed: false,
        message: `Query execution error: ${result.error}`,
        snippet: query,
      }
    } else if (!result.match) {
      checks['query-correct'] = {
        passed: false,
        message: `Expected ${JSON.stringify(xpathScenario.expected)}, got ${JSON.stringify(result.actual)}`,
        snippet: query,
      }
    } else {
      checks['query-correct'] = { passed: true }
    }

    const allPassed = Object.values(checks).every(c => c.passed)
    const score = Object.values(checks).filter(c => c.passed).length / Object.values(checks).length

    return {
      scenarioId: scenario.id,
      checks,
      passed: allPassed,
      score,
      rawResponse: raw,
    }
  },
}

/**
 * Extract the XPath query from the LLM response.
 * Strips markdown code fences, whitespace, and other noise.
 */
function extractQuery(raw: string): string | null {
  let text = raw.trim()

  // Strip markdown code fences
  const codeBlockMatch = text.match(/```(?:xpath|xquery|xml)?\s*\n?([\s\S]*?)\n?\s*```/)
  if (codeBlockMatch) {
    text = codeBlockMatch[1].trim()
  }

  // Strip inline backticks
  if (text.startsWith('`') && text.endsWith('`')) {
    text = text.slice(1, -1).trim()
  }

  if (!text) return null
  return text
}
