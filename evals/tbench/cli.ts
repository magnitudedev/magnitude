#!/usr/bin/env bun

import { spawn } from 'bun'
import path from 'path'
import { Command } from '@commander-js/extra-typings'
import { jobsMain, submitMain } from './jobs'
import { main as runMain, resumeMain } from './run'

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
  .option('-r, --resume <jobDir>', 'Resume a previous job, retrying errored trials (except timeouts)')
  .action(async options => {
    const concurrency = parsePositiveInt(options.concurrency, 1)
    if (options.resume) {
      await resumeMain(options.resume, { concurrency })
    } else {
      await runMain({
        concurrency,
        trials: parsePositiveInt(options.trials, 1),
      })
    }
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

program
  .command('jobs')
  .description('List Harbor tbench jobs under ./jobs')
  .option('--model <filter>', 'Filter jobs by model substring')
  .option('--verbose', 'Show per-task breakdown under each job', false)
  .action(async options => {
    await jobsMain({
      modelFilter: options.model,
      verbose: Boolean(options.verbose),
    })
  })

program
  .command('submit')
  .description('Package completed tbench jobs for leaderboard submission')
  .option('--model <model>', 'Model to submit')
  .option('--jobs <ids>', 'Comma-separated job IDs / folder names to include')
  .option('--force', 'Allow submission packaging despite validation warnings/errors that are force-overridable', false)
  .option(
    '--output <dir>',
    'Submission root directory',
    path.join(process.cwd(), 'submissions', 'terminal-bench', '2.0'),
  )
  .action(async options => {
    await submitMain({
      model: options.model,
      jobs:
        typeof options.jobs === 'string'
          ? options.jobs
              .split(',')
              .map((s: string) => s.trim())
              .filter(Boolean)
          : undefined,
      force: Boolean(options.force),
      outputDir: options.output,
      interactive: !options.model && !options.jobs,
    })
  })

await program.parseAsync(process.argv)