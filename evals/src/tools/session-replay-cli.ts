#!/usr/bin/env bun
/**
 * CLI for extracting LLM conversation messages from a Magnitude session.
 * 
 * Usage:
 *   bun run src/tools/session-replay-cli.ts <events-jsonl-path> <output-dir> [--max-messages N]
 * 
 * Example:
 *   bun run src/tools/session-replay-cli.ts ~/.magnitude/sessions/2026-02-17T04-54-26Z/events.jsonl ./scenarios/my-scenario/
 */

import { extractAndWrite } from './session-replay'

const args = process.argv.slice(2)

if (args.length < 2) {
  console.error('Usage: bun run session-replay-cli.ts <events-jsonl-path> <output-dir> [--max-messages N]')
  process.exit(1)
}

const eventsPath = args[0]
const outputDir = args[1]

let maxMessages: number | undefined
const maxIdx = args.indexOf('--max-messages')
if (maxIdx !== -1 && args[maxIdx + 1]) {
  maxMessages = parseInt(args[maxIdx + 1], 10)
}

await extractAndWrite(eventsPath, outputDir, { maxMessages })
