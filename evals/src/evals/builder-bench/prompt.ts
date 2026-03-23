/**
 * Prompt Construction — build strategy-specific system prompts for builder-bench.
 */

import type { RoleDefinition, ToolSet } from '@magnitudedev/roles'
import { generateJsActToolDocs, generateXmlActToolDocs } from '@magnitudedev/agent'
import type { StrategyId } from './types'

const ROLE_DESCRIPTION = `You are a software engineer tasked with fixing bugs in a project.
You have access to tools for reading files, writing files, searching, and/or running shell commands.
Your goal is to diagnose and fix the bug so that all tests pass.

Rules:
- Do NOT modify any test files. Only fix the source code.
- Read the relevant source files and test files to understand the issue.
- After making changes, run the test command to verify your fix works.
- When all tests pass, call done() to signal completion.`

/**
 * Build a strategy-appropriate system prompt from an agent definition.
 */
export function buildSystemPrompt(
  strategy: StrategyId,
  agentDef: RoleDefinition<ToolSet, string, unknown>,
): string {
  switch (strategy) {
    case 'js-act': {
      const toolDocs = generateJsActToolDocs(agentDef, [])
      // TODO: JS_ACT_PROTOCOL / XML_ACT_PROTOCOL were never exported, need proper protocol generation
      return `${''}\n\n${ROLE_DESCRIPTION}\n\n## Tools\n\n${toolDocs}`
    }
    case 'xml-act': {
      const toolDocs = generateXmlActToolDocs(agentDef, [])
      // TODO: JS_ACT_PROTOCOL / XML_ACT_PROTOCOL were never exported, need proper protocol generation
      return `${''}\n\n${ROLE_DESCRIPTION}\n\n## Tools\n\n${toolDocs}`
    }
    case 'native-openai': {
      return ROLE_DESCRIPTION
    }
  }
}
