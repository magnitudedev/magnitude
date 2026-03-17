#!/usr/bin/env bun

import * as clack from '@clack/prompts'
import ansis from 'ansis'
import { existsSync, readdirSync } from 'fs'
import { access, readFile, readdir, stat } from 'fs/promises'
import os from 'os'
import path from 'path'

type Difficulty = 'easy' | 'medium' | 'hard' | 'unknown'

type TaskInfo = {
  name: string
  difficulty: Difficulty
  category: string
  tags: string[]
  timeoutSec?: number
  tomlPath: string
}

type TaskMode = 'specific' | 'all' | 'easy' | 'medium' | 'hard'

export type RunOptions = {
  concurrency: number
  trials: number
}

// Harbor result.json types
type HarborJobResult = {
  id: string
  started_at: string
  finished_at: string
  n_total_trials: number
  stats: {
    n_trials: number
    n_errors: number
    evals: Record<string, HarborEvalResult>
  }
}

type HarborEvalResult = {
  n_trials: number
  n_errors: number
  metrics: { mean: number }[]
  reward_stats: {
    reward: Record<string, string[]> // e.g. { "0.0": ["fix-git__abc"], "1.0": ["other__def"] }
  }
  exception_stats: Record<string, string[]> // e.g. { "AgentTimeoutError": ["fix-git__abc"] }
}

type ResultRow = {
  task: string
  reward: number
  status: string
  exception?: string
}

type TaskToml = {
  version?: string
  metadata?: {
    author_name?: string
    author_email?: string
    difficulty?: string
    category?: string
    tags?: string[]
    expert_time_estimate_min?: number
    junior_time_estimate_min?: number
  }
  verifier?: {
    timeout_sec?: number
  }
  agent?: {
    timeout_sec?: number
  }
  environment?: {
    build_timeout_sec?: number
    docker_image?: string
    cpus?: number
    memory?: string
    storage?: string
  }
}

const MODELS = ['anthropic/claude-sonnet-4-6', 'openai/gpt-5.4', 'openai/gpt-5.3-codex', 'openai/gpt-5.3-codex-spark', 'openrouter/qwen/qwen3.5-27b'] as const
const JOBS_DIR = path.join(process.cwd(), 'jobs')
const TASKS_ROOT = path.join(os.homedir(), '.cache/harbor/tasks')
const POLL_MS = 500
const TAIL_POLL_MS = 300



function normalizeDifficulty(value: unknown): Difficulty {
  if (typeof value !== 'string') return 'unknown'
  const normalized = value.toLowerCase()
  if (normalized === 'easy' || normalized === 'medium' || normalized === 'hard') {
    return normalized
  }
  return 'unknown'
}

function parseTaskToml(raw: string, taskName: string, tomlPath: string): TaskInfo {
  const parsed = Bun.TOML.parse(raw) as TaskToml
  const metadata = parsed.metadata
  const difficulty = normalizeDifficulty(metadata?.difficulty)
  const category = typeof metadata?.category === 'string' ? metadata.category : 'uncategorized'
  const tags = Array.isArray(metadata?.tags) ? metadata.tags.filter((tag): tag is string => typeof tag === 'string') : []

  const timeoutSec = typeof parsed.agent?.timeout_sec === 'number' ? parsed.agent.timeout_sec : undefined

  return {
    name: taskName,
    difficulty,
    category,
    tags,
    timeoutSec,
    tomlPath,
  }
}

async function discoverTasks(): Promise<TaskInfo[]> {
  if (!existsSync(TASKS_ROOT)) {
    throw new Error(`Task cache not found at ${TASKS_ROOT}`)
  }

  const hashDirs = await readdir(TASKS_ROOT, { withFileTypes: true })
  const tasks: TaskInfo[] = []

  for (const hashDir of hashDirs) {
    if (!hashDir.isDirectory()) continue
    const hashPath = path.join(TASKS_ROOT, hashDir.name)
    const taskDirs = await readdir(hashPath, { withFileTypes: true })
    for (const taskDir of taskDirs) {
      if (!taskDir.isDirectory()) continue
      const taskName = taskDir.name
      const tomlPath = path.join(hashPath, taskName, 'task.toml')
      if (!existsSync(tomlPath)) continue
      const raw = await readFile(tomlPath, 'utf8')
      tasks.push(parseTaskToml(raw, taskName, tomlPath))
    }
  }

  const difficultyOrder: Difficulty[] = ['easy', 'medium', 'hard', 'unknown']
  return tasks.sort((a, b) => {
    const diff = difficultyOrder.indexOf(a.difficulty) - difficultyOrder.indexOf(b.difficulty)
    return diff !== 0 ? diff : a.name.localeCompare(b.name)
  })
}

function countByDifficulty(tasks: TaskInfo[]) {
  return {
    easy: tasks.filter(task => task.difficulty === 'easy').length,
    medium: tasks.filter(task => task.difficulty === 'medium').length,
    hard: tasks.filter(task => task.difficulty === 'hard').length,
  }
}

function commandForRun(model: string, selectedTasks: string[], concurrency: number, trials: number) {
  const args = [
    'harbor',
    'run',
    '-d',
    'terminal-bench@2.0',
    '--agent-import-path',
    'evals.tbench.magnitude_agent:MagnitudeAgent',
    '-m',
    model,
  ]

  for (const task of selectedTasks) {
    args.push('-t', task)
  }

  args.push('-n', String(concurrency))
  args.push('-k', String(trials))
  args.push('-q')
  return args
}

function renderCommand(args: string[]) {
  return args.map(arg => (/\s/.test(arg) ? JSON.stringify(arg) : arg)).join(' ')
}

async function validateEnvironment() {
  const binaryPath = path.join(process.cwd(), 'evals/tbench/bin/magnitude')
  const warnings: string[] = []

  if (!existsSync(binaryPath)) {
    warnings.push('Missing evals/tbench/bin/magnitude — run ./evals/tbench/build-linux.sh')
  }

  const harborCheck = Bun.spawn(['which', 'harbor'], { stdout: 'ignore', stderr: 'ignore' })
  const harborCode = await harborCheck.exited
  if (harborCode !== 0) {
    warnings.push('`harbor` not found on PATH — install via pip or uv tool install harbor')
  }

  const dockerCheck = Bun.spawn(['docker', 'info'], { stdout: 'ignore', stderr: 'ignore' })
  const dockerCode = await dockerCheck.exited
  if (dockerCode !== 0) {
    warnings.push('Docker is not running or not reachable (`docker info` failed)')
  }

  return warnings
}

function sleep(ms: number) {
  return new Promise(resolve => setTimeout(resolve, ms))
}

async function sha256File(filePath: string): Promise<string> {
  const bytes = await Bun.file(filePath).arrayBuffer()
  return new Bun.CryptoHasher('sha256').update(bytes).digest('hex')
}

async function writeMagnitudeMeta(jobDir: string, startedAtIso: string) {
  try {
    const binaryPath = path.join(process.cwd(), 'evals/tbench/bin/magnitude')
    const binaryHash = await sha256File(binaryPath)
    await Bun.write(
      path.join(jobDir, 'magnitude-meta.json'),
      JSON.stringify(
        {
          binaryHash,
          binaryPath,
          timestamp: startedAtIso,
        },
        null,
        2,
      ),
    )
  } catch (error) {
    clack.log.warn(
      `Failed to write magnitude-meta.json: ${error instanceof Error ? error.message : String(error)}`,
    )
  }
}

async function waitForJobDir(before: Set<string>, signal: AbortSignal): Promise<string | null> {
  while (!signal.aborted) {
    if (existsSync(JOBS_DIR)) {
      const entries = readdirSync(JOBS_DIR, { withFileTypes: true })
        .filter(entry => entry.isDirectory())
        .map(entry => entry.name)
      const fresh = entries.filter(name => !before.has(name)).sort()
      if (fresh.length > 0) {
        return path.join(JOBS_DIR, fresh[fresh.length - 1]!)
      }
    }
    await sleep(POLL_MS)
  }
  return null
}

function formatElapsed(ms: number) {
  const totalSeconds = Math.max(0, Math.floor(ms / 1000))
  const minutes = Math.floor(totalSeconds / 60)
  const seconds = totalSeconds % 60
  return `${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`
}

function getTaskCacheDir(task: TaskInfo) {
  return path.dirname(task.tomlPath)
}

function getBaseTaskName(jobTaskDirName: string) {
  return jobTaskDirName.split('__')[0] ?? jobTaskDirName
}

async function readTaskInstruction(task: TaskInfo): Promise<string | null> {
  const instructionPath = path.join(getTaskCacheDir(task), 'instruction.md')
  try {
    const raw = await readFile(instructionPath, 'utf8')
    const text = raw.trim().replace(/\s+/g, ' ')
    return text || null
  } catch {
    return null
  }
}

function formatTaskHeader(task: TaskInfo) {
  const timeout = typeof task.timeoutSec === 'number' ? `${task.timeoutSec}s timeout` : 'unknown timeout'
  return `${task.name} (${task.difficulty} · ${task.category} · ${timeout})`
}

async function tailFile(filePath: string, taskName: string, startedAt: number, signal: AbortSignal) {
  let position = 0
  let pending = ''

  while (!signal.aborted) {
    try {
      const fileStat = await stat(filePath)
      if (fileStat.size > position) {
        const file = Bun.file(filePath)
        const chunk = await file.slice(position, fileStat.size).text()
        position = fileStat.size
        pending += chunk

        const lines = pending.split(/\r?\n/)
        pending = lines.pop() ?? ''
        for (const line of lines) {
          if (!line.trim()) continue
          const elapsed = formatElapsed(Date.now() - startedAt)
          console.log(`${ansis.dim(`[${elapsed}]`)} ${ansis.cyan(taskName)} ${line}`)
        }
      }
    } catch {
      // file may not exist yet
    }
    await sleep(TAIL_POLL_MS)
  }

  if (pending.trim()) {
    const elapsed = formatElapsed(Date.now() - startedAt)
    console.log(`${ansis.dim(`[${elapsed}]`)} ${ansis.cyan(taskName)} ${pending}`)
  }
}

async function watchTaskLogs(jobDir: string, tasks: TaskInfo[], signal: AbortSignal, alreadyTailed?: Set<string>) {
  const startedAt = Date.now()
  const tailed = new Set<string>(alreadyTailed)
  const tailPromises: Promise<void>[] = []
  const taskByName = new Map(tasks.map(task => [task.name, task]))

  while (!signal.aborted) {
    try {
      const taskDirs = await readdir(jobDir, { withFileTypes: true })
      for (const taskDir of taskDirs) {
        if (!taskDir.isDirectory()) continue
        const taskName = taskDir.name
        if (tailed.has(taskName) || taskName.startsWith('.')) continue

        const logPath = path.join(jobDir, taskName, 'agent', 'magnitude.txt')
        if (existsSync(logPath)) {
          tailed.add(taskName)

          const baseTaskName = getBaseTaskName(taskName)
          const task = taskByName.get(baseTaskName)

          console.log()
          if (task) {
            console.log(ansis.bold(formatTaskHeader(task)))
            const instruction = await readTaskInstruction(task)
            if (instruction) {
              console.log(`${instruction}`)
            }
          } else {
            console.log(ansis.bold(taskName))
          }

          tailPromises.push(tailFile(logPath, taskName, startedAt, signal))
        }
      }
    } catch {
      // job dir may not be fully ready yet
    }
    await sleep(POLL_MS)
  }

  await Promise.allSettled(tailPromises)
}

function getResultRows(jobResult: HarborJobResult): ResultRow[] {
  const rows: ResultRow[] = []

  for (const evalResult of Object.values(jobResult.stats.evals)) {
    // Build exception lookup: trialName -> exceptionType
    const exceptionByTrial = new Map<string, string>()
    for (const [exType, trials] of Object.entries(evalResult.exception_stats ?? {})) {
      for (const trial of trials) exceptionByTrial.set(trial, exType)
    }

    // Add rows for error-only trials (no reward entry)
    const trialsSeen = new Set<string>()
    for (const trials of Object.values(evalResult.reward_stats.reward ?? {})) {
      for (const t of trials) trialsSeen.add(t)
    }
    for (const [trialName, exType] of exceptionByTrial) {
      if (trialsSeen.has(trialName)) continue
      const taskName = trialName.split('__')[0] ?? trialName
      rows.push({
        task: taskName,
        reward: 0,
        status: `error (${exType})`,
        exception: exType,
      })
    }

    for (const [rewardStr, trials] of Object.entries(evalResult.reward_stats.reward ?? {})) {
      const reward = parseFloat(rewardStr)
      for (const trialName of trials) {
        const taskName = trialName.split('__')[0] ?? trialName
        const exception = exceptionByTrial.get(trialName)
        rows.push({
          task: taskName,
          reward,
          status: exception ? `failed (${exception})` : reward >= 1 ? 'passed' : 'failed',
          exception,
        })
      }
    }
  }

  return rows
}

async function printExceptionDetails(jobDir: string) {
  const entries = readdirSync(jobDir, { withFileTypes: true })
  for (const entry of entries) {
    if (!entry.isDirectory() || entry.name.startsWith('.')) continue
    const exPath = path.join(jobDir, entry.name, 'exception.txt')
    if (!existsSync(exPath)) continue
    try {
      const content = await readFile(exPath, 'utf8')
      const lines = content.trim().split('\n')
      // Show last line (the actual error) plus the task name
      const errorLine = lines[lines.length - 1] ?? content.trim()
      console.log()
      clack.log.error(`${ansis.bold(entry.name)}: ${errorLine}`)
    } catch {}
  }
}

function printResultsTable(rows: ResultRow[], resultPath: string) {
  console.log()
  clack.log.info('Results')

  if (rows.length === 0) {
    console.log(`No task rows found in ${resultPath}`)
    return
  }

  const taskWidth = Math.max('Task'.length, ...rows.map(row => row.task.length))
  console.log(`${'Task'.padEnd(taskWidth)}  ${'Reward'.padStart(6)}  Status`)
  console.log(`${'─'.repeat(taskWidth)}  ${'─'.repeat(6)}  ${'─'.repeat(20)}`)

  let passed = 0
  let rewardSum = 0
  let rewardCount = 0

  for (const row of rows) {
    rewardSum += row.reward
    rewardCount += 1
    if (row.reward >= 1) passed += 1

    const reward = row.reward.toFixed(1)
    const status =
      row.status === 'passed'
        ? ansis.green(`✓ ${row.status}`)
        : row.status.startsWith('failed')
          ? ansis.red(`✗ ${row.status}`)
          : row.status

    console.log(`${row.task.padEnd(taskWidth)}  ${reward.padStart(6)}  ${status}`)
  }

  const mean = rewardCount > 0 ? rewardSum / rewardCount : 0
  console.log()
  console.log(`Mean: ${mean.toFixed(3)}  |  ${passed}/${rows.length} passed`)
  console.log(`Full results: ${resultPath}`)
}

function buildSpecificTaskOptions(tasks: TaskInfo[]) {
  const groups: Difficulty[] = ['easy', 'medium', 'hard', 'unknown']
  const options: Array<{ value: string; label: string; hint?: string }> = []

  for (const difficulty of groups) {
    const groupTasks = tasks.filter(task => task.difficulty === difficulty)
    if (groupTasks.length === 0) continue

    options.push({
      value: `__header__${difficulty}`,
      label: `── ${difficulty} ──`,
      hint: `${groupTasks.length} tasks`,
    })

    for (const task of groupTasks) {
      options.push({
        value: task.name,
        label: `${task.name} (${task.category})`,
        hint: task.tags.length > 0 ? task.tags.slice(0, 3).join(', ') : undefined,
      })
    }
  }

  return options
}

export async function main(options: Partial<RunOptions> = {}) {
  const concurrency = options.concurrency ?? 1
  const trials = options.trials ?? 1

  clack.intro(ansis.bold('Magnitude TB2 Runner'))

  const warnings = await validateEnvironment()
  if (warnings.length > 0) {
    for (const warning of warnings) {
      clack.log.warn(warning)
    }

    const proceed = await clack.confirm({
      message: 'Continue anyway?',
      initialValue: false,
    })

    if (clack.isCancel(proceed) || !proceed) {
      clack.cancel('Cancelled')
      process.exit(1)
    }
  }

  const tasks = await discoverTasks()
  if (tasks.length === 0) {
    clack.log.error(`No tasks found under ${TASKS_ROOT}`)
    process.exit(1)
  }

  const model = await clack.select({
    message: 'Model',
    options: MODELS.map(value => ({ value, label: value })),
  })
  if (clack.isCancel(model)) {
    clack.cancel('Cancelled')
    return
  }

  const counts = countByDifficulty(tasks)
  const mode = await clack.select({
    message: 'Task selection',
    options: [
      { value: 'specific', label: 'Pick specific tasks' },
      { value: 'all', label: `All tasks (${tasks.length})` },
      { value: 'easy', label: `All easy (${counts.easy})` },
      { value: 'medium', label: `All medium (${counts.medium})` },
      { value: 'hard', label: `All hard (${counts.hard})` },
    ],
  })
  if (clack.isCancel(mode)) {
    clack.cancel('Cancelled')
    return
  }

  let selectedTasks: TaskInfo[] = []
  switch (mode as TaskMode) {
    case 'specific': {
      const picks = await clack.multiselect({
        message: 'Select tasks',
        options: buildSpecificTaskOptions(tasks),
        required: true,
      })

      if (clack.isCancel(picks)) {
        clack.cancel('Cancelled')
        return
      }

      const selected = new Set((picks as string[]).filter(value => !value.startsWith('__header__')))
      selectedTasks = tasks.filter(task => selected.has(task.name))
      break
    }
    case 'easy':
    case 'medium':
    case 'hard':
      selectedTasks = tasks.filter(task => task.difficulty === mode)
      break
    case 'all':
      selectedTasks = []
      break
  }

  if (mode !== 'all' && selectedTasks.length === 0) {
    clack.log.error('No tasks selected')
    process.exit(1)
  }

  const commandArgs = commandForRun(model as string, selectedTasks.map(task => task.name), concurrency, trials)

  console.log()
  console.log(ansis.bold('Ready to run'))
  console.log(`  Model:       ${model}`)
  console.log(
    `  Tasks:       ${
      mode === 'all'
        ? `all (${tasks.length})`
        : selectedTasks.length === 1
          ? selectedTasks[0]!.name
          : `${selectedTasks.length} tasks`
    }`,
  )
  if (concurrency > 1) console.log(`  Concurrency: ${concurrency}`)
  if (trials > 1) console.log(`  Trials:      ${trials}`)
  console.log()
  console.log(ansis.dim(renderCommand(commandArgs)))
  console.log()

  const confirmed = await clack.confirm({
    message: 'Start run?',
    initialValue: true,
  })
  if (clack.isCancel(confirmed) || !confirmed) {
    clack.cancel('Cancelled')
    return
  }

  const beforeJobs = existsSync(JOBS_DIR)
    ? new Set(
        readdirSync(JOBS_DIR, { withFileTypes: true })
          .filter(entry => entry.isDirectory())
          .map(entry => entry.name),
      )
    : new Set<string>()

  const abortController = new AbortController()
  let child: ReturnType<typeof Bun.spawn> | null = null

  process.on('SIGINT', () => {
    abortController.abort()
    if (child) {
      child.kill()
    }
  })

  const runStartedAt = new Date().toISOString()

  child = Bun.spawn(commandArgs, {
    stdout: 'pipe',
    stderr: 'inherit',
    stdin: 'inherit',
  })

  let jobDir: string | null = null
  const watchJobPromise = waitForJobDir(beforeJobs, abortController.signal).then(async found => {
    jobDir = found
    if (found) {
      await writeMagnitudeMeta(found, runStartedAt)
      clack.log.info(`Watching ${path.relative(process.cwd(), found)}`)
      await watchTaskLogs(found, tasks, abortController.signal)
    }
  })

  const exitCode = await child.exited
  abortController.abort()
  await watchJobPromise

  if (exitCode !== 0) {
    clack.log.error(`harbor exited with code ${exitCode}`)
    process.exit(exitCode)
  }

  if (!jobDir) {
    clack.log.warn('Run finished, but no new jobs directory was detected')
    return
  }

  const resultPath = path.join(jobDir, 'result.json')
  try {
    await access(resultPath)
    const parsed = JSON.parse(await readFile(resultPath, 'utf8')) as HarborJobResult
    const rows = getResultRows(parsed)
    printResultsTable(rows, path.relative(process.cwd(), resultPath))
    await printExceptionDetails(jobDir)
  } catch (error) {
    clack.log.warn(`Run finished, but could not parse result.json: ${error instanceof Error ? error.message : String(error)}`)
    await printExceptionDetails(jobDir)
  }

  clack.outro('Done')
}

export async function resumeMain(jobDirName: string, options: { concurrency: number }) {
  const { concurrency } = options
  const jobDir = path.join(JOBS_DIR, jobDirName)

  if (!existsSync(jobDir)) {
    clack.log.error(`Job directory not found: ${jobDir}`)
    process.exit(1)
  }

  const configPath = path.join(jobDir, 'config.json')
  if (!existsSync(configPath)) {
    clack.log.error(`No config.json found in ${jobDir}`)
    process.exit(1)
  }

  // Scan trial subdirectories for retryable errors
  const errorTypeCounts = new Map<string, number>()

  const entries = readdirSync(jobDir, { withFileTypes: true })
  for (const entry of entries) {
    if (!entry.isDirectory()) continue
    const trialResultPath = path.join(jobDir, entry.name, 'result.json')
    if (!existsSync(trialResultPath)) continue
    try {
      const parsed = JSON.parse(await readFile(trialResultPath, 'utf8')) as {
        exception_info?: { exception_type: string } | null
      }
      const exType = parsed.exception_info?.exception_type
      if (exType && exType !== 'AgentTimeoutError') {
        errorTypeCounts.set(exType, (errorTypeCounts.get(exType) ?? 0) + 1)
      }
    } catch {
      // skip unparseable result.json
    }
  }

  // Print summary
  clack.log.info('Resume summary:')
  if (errorTypeCounts.size > 0) {
    for (const [exType, count] of errorTypeCounts) {
      console.log(`  ${exType}: ${count} trial(s) (retrying)`)
    }
  } else {
    console.log('  No retryable errors found')
  }
  console.log('  Harbor will also pick up any incomplete (no result) trials automatically')

  // Always include CancelledError
  const filterErrors = new Set(errorTypeCounts.keys())
  filterErrors.add('CancelledError')

  const commandArgs = ['harbor', 'jobs', 'resume', '-p', jobDir]
  for (const exType of filterErrors) {
    commandArgs.push('-f', exType)
  }
  // Note: harbor jobs resume doesn't support -q or -n flags;
  // concurrency comes from the existing job config

  console.log()
  console.log(ansis.dim(renderCommand(commandArgs)))
  console.log()

  const tasks = await discoverTasks()

  // Pre-populate already-tailed tasks to avoid replaying old logs
  const alreadyTailed = new Set<string>()
  for (const entry of entries) {
    if (!entry.isDirectory()) continue
    const logPath = path.join(jobDir, entry.name, 'agent', 'magnitude.txt')
    if (existsSync(logPath)) {
      alreadyTailed.add(entry.name)
    }
  }

  const abortController = new AbortController()
  let child: ReturnType<typeof Bun.spawn> | null = null

  process.on('SIGINT', () => {
    abortController.abort()
    if (child) child.kill()
  })

  child = Bun.spawn(commandArgs, {
    stdout: 'pipe',
    stderr: 'inherit',
    stdin: 'inherit',
  })

  clack.log.info(`Watching ${path.relative(process.cwd(), jobDir)}`)
  const watchPromise = watchTaskLogs(jobDir, tasks, abortController.signal, alreadyTailed)

  const exitCode = await child.exited
  abortController.abort()
  await watchPromise

  if (exitCode !== 0) {
    clack.log.error(`harbor exited with code ${exitCode}`)
    process.exit(exitCode)
  }

  const resultPath = path.join(jobDir, 'result.json')
  try {
    await access(resultPath)
    const parsed = JSON.parse(await readFile(resultPath, 'utf8')) as HarborJobResult
    const rows = getResultRows(parsed)
    printResultsTable(rows, path.relative(process.cwd(), resultPath))
    await printExceptionDetails(jobDir)
  } catch (error) {
    clack.log.warn(
      `Resume finished, but could not parse result.json: ${error instanceof Error ? error.message : String(error)}`,
    )
    await printExceptionDetails(jobDir)
  }

  clack.outro('Done')
}

if (import.meta.path === Bun.main) {
  await main()
}