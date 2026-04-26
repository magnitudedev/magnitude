import type {
  GrammarToolDef,
  GrammarConfig,
  GrammarBuildOptions,
  ProtocolConfig,
} from './grammar-types'
import {
  sanitizeRuleName,
  sanitizeParamName,
  escapeGbnfChar,
  escapeGbnfCharClass,
} from './grammar-utils'
import { LEAD_YIELD_TAGS } from '../constants'
import {
  addWhitespaceRules,
  addAttributeRules,
  addYieldRules,
  addContinuationRules,
  addSharedBucRules,
  addTopLevelBodyRules,
  addToolRules,
  addRootRule,
} from './rule-contributors'

// Re-export public API
export type {
  GrammarToolDef,
  GrammarBuildOptions,
  GrammarParameterDef,
  ProtocolConfig,
  GrammarConfig,
} from './grammar-types'
export {
  sanitizeRuleName,
  sanitizeParamName,
  escapeGbnfChar,
  escapeGbnfCharClass,
} from './grammar-utils'

type RuleMap = Map<string, string>

function serializeGrammar(rules: RuleMap): string {
  const lines: string[] = []
  for (const [name, production] of rules) {
    lines.push(`${name} ::= ${production}`)
  }
  return lines.join('\n')
}

const defaultProtocol: ProtocolConfig = {
  minLenses: 0,
  allowMessages: true,
  allowTools: true,
  requiredMessageTo: null,
  maxLenses: undefined,
  yieldTags: LEAD_YIELD_TAGS,
  lensNames: ['alignment', 'tasks', 'diligence', 'skills', 'turn', 'pivot'],
}

export class GrammarBuilder {
  private constructor(private readonly config: GrammarConfig) {}

  static create(tools: ReadonlyArray<GrammarToolDef>): GrammarBuilder {
    return new GrammarBuilder({
      tools,
      protocol: defaultProtocol,
    })
  }

  withMinLenses(minLenses: 0 | 1): GrammarBuilder {
    return new GrammarBuilder({
      ...this.config,
      protocol: { ...this.config.protocol, minLenses },
    })
  }

  requireMessageTo(recipient: string): GrammarBuilder {
    return new GrammarBuilder({
      ...this.config,
      protocol: { ...this.config.protocol, requiredMessageTo: recipient },
    })
  }

  withMaxLenses(maxLenses: number): GrammarBuilder {
    return new GrammarBuilder({
      ...this.config,
      protocol: { ...this.config.protocol, maxLenses },
    })
  }

  withYieldTags(yieldTags: ReadonlyArray<string>): GrammarBuilder {
    return new GrammarBuilder({
      ...this.config,
      protocol: { ...this.config.protocol, yieldTags },
    })
  }

  withLensNames(lensNames: ReadonlyArray<string>): GrammarBuilder {
    return new GrammarBuilder({
      ...this.config,
      protocol: { ...this.config.protocol, lensNames },
    })
  }

  withOptions(options: GrammarBuildOptions): GrammarBuilder {
    let next = this as GrammarBuilder
    if (options.minLenses !== undefined) next = next.withMinLenses(options.minLenses)
    if (options.requiredMessageTo !== undefined) next = next.requireMessageTo(options.requiredMessageTo)
    if (options.maxLenses !== undefined) next = next.withMaxLenses(options.maxLenses)
    if (options.yieldTags !== undefined) next = next.withYieldTags(options.yieldTags)
    if (options.lensNames !== undefined) next = next.withLensNames(options.lensNames)
    return next
  }

  build(): string {
    const { requiredMessageTo, maxLenses } = this.config.protocol

    if (maxLenses !== undefined && requiredMessageTo === null) {
      throw new Error('maxLenses requires requiredMessageTo to be set')
    }

    const rules: RuleMap = new Map()

    addWhitespaceRules(rules)
    addAttributeRules(rules)
    addYieldRules(rules, this.config.protocol.yieldTags)
    addContinuationRules(rules, this.config)
    addSharedBucRules(rules)
    addTopLevelBodyRules(rules)
    addToolRules(rules, this.config)
    addRootRule(rules, this.config)

    return serializeGrammar(rules)
  }
}
