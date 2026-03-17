#!/usr/bin/env bun

import { spawn } from 'bun'
import { Command } from '@commander-js/extra-typings'
import { main as runMain } from './run'

function parsePositiveInt(value: string, fallback: number) {
  const parsed = Number.parseInt(value, 10)
  return Number.isFinite(parsed) && parsed > 0 ? parsed : fallback
}

const program = new Command()
  .name('tbench')
  .description('Magnitude Terminal Bench utilities')

program
  .command('run')
  .description('Start the interactive TB2 runner')
  .option('-c, --concurrency <number>', 'Concurrency to pass to harbor via -n', '1')
  .option('-t, --trials <number>', 'Trials to pass to harbor via -k', '1')
  .action(async options => {
    await runMain({
      concurrency: parsePositiveInt(options.concurrency, 1),
      trials: parsePositiveInt(options.trials, 1),
    })
  })

program
  .command('build')
  .description('Run ./evals/tbench/build-linux.sh')
  .action(async () => {
    const child = spawn(['./evals/tbench/build-linux.sh'], {
      stdout: 'inherit',
      stderr: 'inherit',
      stdin: 'inherit',
    })
    const code = await child.exited
    process.exit(code)
  })

await program.parseAsync(process.argv)