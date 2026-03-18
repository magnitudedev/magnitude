#!/usr/bin/env bun

import * as clack from '@clack/prompts'
import ansis from 'ansis'
import { spawn } from 'bun'
import { existsSync } from 'fs'
import { readdir, readFile } from 'fs/promises'
import path from 'path'
import { Command } from '@commander-js/extra-typings'
import { jobsMain, submitMain } from './jobs'
import { main as runMain, resumeMain, TASKS_ROOT } from './run'

function parsePositiveInt(value: string, fallback: number) {
  const parsed = Number.parseInt(value, 10)
  return Number.isFinite(parsed) && parsed > 0 ? parsed : fallback
}

async function collectDockerImages(root: string): Promise<string[]> {
  const images = new Set<string>()

  async function walk(dir: string): Promise<void> {
    const entries = await readdir(dir, { withFileTypes: true })
    for (const entry of entries) {
      const fullPath = path.join(dir, entry.name)
      if (entry.isDirectory()) {
        await walk(fullPath)
        continue
      }
      if (!entry.isFile() || entry.name !== 'task.toml') continue

      try {
        const raw = await readFile(fullPath, 'utf8')
        const match = raw.match(/^\s*docker_image\s*=\s*"([^"]+)"/m)
        if (match?.[1]) {
          images.add(match[1])
        }
      } catch (error) {
        clack.log.warn(
          `Failed to read ${fullPath}: ${error instanceof Error ? error.message : String(error)}`,
        )
      }
    }
  }

  await walk(root)
  return [...images].sort()
}

async function pullImagesMain() {
  clack.intro(ansis.bold('Pull TB2 task images'))

  if (!existsSync(TASKS_ROOT)) {
    clack.log.warn(`Harbor task cache not found at ${TASKS_ROOT}`)
    clack.outro('Nothing to do')
    return
  }

  const images = await collectDockerImages(TASKS_ROOT)
  if (images.length === 0) {
    clack.log.warn(`No docker_image entries found under ${TASKS_ROOT}`)
    clack.outro('Nothing to do')
    return
  }

  clack.log.info(`Found ${images.length} unique image(s)`)

  let successCount = 0
  let failureCount = 0

  for (const image of images) {
    console.log()
    clack.log.step(`Pulling ${image}`)
    const child = spawn(['docker', 'pull', image], {
      stdout: 'inherit',
      stderr: 'inherit',
      stdin: 'inherit',
    })
    const code = await child.exited
    if (code === 0) {
      successCount += 1
    } else {
      failureCount += 1
      clack.log.error(`Failed to pull ${image} (exit code ${code})`)
    }
  }

  console.log()
  if (failureCount > 0) {
    clack.log.warn(`Done: ${successCount} succeeded, ${failureCount} failed`)
  } else {
    clack.log.success(`Done: all ${successCount} image(s) pulled successfully`)
  }
  clack.outro('Finished')
}

const program = new Command()
  .name('tbench')
  .description('Magnitude Terminal Bench utilities')

program
  .command('run')
  .description('Start the interactive TB2 runner')
  .option('-c, --concurrency <number>', 'Concurrency to pass to harbor via -n', '1')
  .option('-t, --trials <number>', 'Trials to pass to harbor via -k', '1')
  .option('--env <provider>', 'Execution environment/provider (default: daytona, use "local" for local Docker)', 'daytona')
  .option('-r, --resume <jobDir>', 'Resume a previous job, retrying errored trials (except timeouts)')
  .action(async options => {
    const concurrency = parsePositiveInt(options.concurrency, 1)
    if (options.resume) {
      await resumeMain(options.resume, { concurrency })
    } else {
      await runMain({
        concurrency,
        trials: parsePositiveInt(options.trials, 1),
        env: options.env,
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
  .command('pull-images')
  .description('Pre-pull all TB2 task Docker images from the Harbor task cache')
  .action(async () => {
    await pullImagesMain()
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