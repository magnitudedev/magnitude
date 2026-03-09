#!/usr/bin/env npx tsx
/**
 * Generate a Mermaid diagram of the projection dependency graph for an agent.
 *
 * Usage:
 *   npx tsx scripts/dep-graph.ts <path-to-agent-file> <agent-export-name> [output-file]
 *
 * Example:
 *   npx tsx scripts/dep-graph.ts ../agent/src/coding-agent.ts CodingAgent ./docs/dep-graph.md
 *
 * Output: Writes Mermaid diagram to the specified file (or stdout if no file given)
 */

import { resolve } from 'path'
import { writeFileSync } from 'fs'

interface SignalSubscription {
  signal: string
  source: string
}

interface ProjectionInfo {
  name: string
  reads: readonly string[]
  signalSubscriptions: readonly SignalSubscription[]
  isForked: boolean
}

async function main() {
  const args = process.argv.slice(2)

  if (args.length < 2) {
    console.error('Usage: npx tsx scripts/dep-graph.ts <path-to-agent-file> <agent-export-name> [output-file]')
    console.error('')
    console.error('Example:')
    console.error('  npx tsx scripts/dep-graph.ts ../agent/src/coding-agent.ts CodingAgent ./docs/dep-graph.md')
    process.exit(1)
  }

  const [filePath, exportName, outputFile] = args
  const absolutePath = resolve(process.cwd(), filePath)

  // Dynamic import the module
  let module: Record<string, unknown>
  try {
    module = await import(absolutePath)
  } catch (err) {
    console.error(`Failed to import ${absolutePath}:`, (err as Error).message)
    process.exit(1)
  }

  const agent = module[exportName] as {
    projections?: Array<ProjectionInfo>
  } | undefined

  if (!agent) {
    console.error(`Export "${exportName}" not found in ${filePath}`)
    console.error('Available exports:', Object.keys(module).join(', '))
    process.exit(1)
  }

  if (!agent.projections || !Array.isArray(agent.projections)) {
    console.error(`Agent "${exportName}" has no projections array`)
    process.exit(1)
  }

  const projections: ProjectionInfo[] = agent.projections.map(p => ({
    name: p.name,
    reads: p.reads,
    signalSubscriptions: p.signalSubscriptions,
    isForked: p.isForked
  }))

  const mermaid = generateMermaid(exportName, projections)

  if (outputFile) {
    const outputPath = resolve(process.cwd(), outputFile)
    writeFileSync(outputPath, mermaid)
    console.log(`Wrote dependency graph to ${outputPath}`)
  } else {
    console.log(mermaid)
  }
}

function generateMermaid(agentName: string, projections: ProjectionInfo[]): string {
  const lines: string[] = []

  lines.push(`# ${agentName} Projection Dependency Graph`)
  lines.push('')
  lines.push('```mermaid')
  lines.push('flowchart TD')
  lines.push('')

  // Define nodes with styling based on forked status
  for (const proj of projections) {
    if (proj.isForked) {
      lines.push(`    ${proj.name}([${proj.name}])`)
    } else {
      lines.push(`    ${proj.name}[${proj.name}]`)
    }
  }

  lines.push('')

  // Add edges for read dependencies (solid lines)
  let hasEdges = false
  for (const proj of projections) {
    for (const dep of proj.reads) {
      lines.push(`    ${dep} --> ${proj.name}`)
      hasEdges = true
    }
  }

  // Add edges for signal subscriptions (dashed lines with signal name)
  for (const proj of projections) {
    for (const sub of proj.signalSubscriptions) {
      const signalLabel = sub.signal.split('/')[1] // e.g., "Fork/forkCreated" -> "forkCreated"
      lines.push(`    ${sub.source} -.->|${signalLabel}| ${proj.name}`)
      hasEdges = true
    }
  }

  if (!hasEdges) {
    lines.push('    %% No dependencies between projections')
  }

  lines.push('```')
  lines.push('')
  lines.push('## Legend')
  lines.push('')
  lines.push('- `[Name]` - Standard projection')
  lines.push('- `([Name])` - Forked projection (per-fork state)')
  lines.push('- `A --> B` - B reads from A')
  lines.push('- `A -.->|signal| B` - B subscribes to signal from A')
  lines.push('')

  return lines.join('\n')
}

main().catch(console.error)
