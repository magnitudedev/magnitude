#!/usr/bin/env bun

import { spawn } from 'bun'
import { main as runMain } from './run'

function printHelp() {
  console.log(`Magnitude TB2 Commands

Usage:
  bun tbench run
  bun tbench build

Commands:
  run    Start the interactive TB2 runner
  build  Run ./evals/tbench/build-linux.sh`)
}

const subcommand = process.argv[2]

switch (subcommand) {
  case 'run':
    await runMain()
    break

  case 'build': {
    const child = spawn(['./evals/tbench/build-linux.sh'], {
      stdout: 'inherit',
      stderr: 'inherit',
      stdin: 'inherit',
    })
    const code = await child.exited
    process.exit(code)
    break
  }

  case '--help':
  case '-h':
  case undefined:
    printHelp()
    break

  default:
    console.error(`Unknown tbench subcommand: ${subcommand}`)
    console.error()
    printHelp()
    process.exit(1)
}