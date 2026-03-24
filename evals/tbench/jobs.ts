#!/usr/bin/env bun

import * as clack from '@clack/prompts'
import ansis from 'ansis'
import { existsSync } from 'fs'
import { cp, readdir, stat } from 'fs/promises'
import path from 'path'

export type TBenchMagnitudeMeta = {
  binaryHash: string
  binaryPath: string
  timestamp: string
}

export type HarborJobConfig = {
  job_name?: string
  timeout_multiplier?: number
  lead?: {
    type?: string
    n_concurrent_trials?: number
    quiet?: boolean
    retry?: {
      max_retries?: number
      include_exceptions?: string[]
      exclude_exceptions?: string[]
      wait_multiplier?: number
      min_wait_sec?: number
      max_wait_sec?: number
    }
    kwargs?: Record<string, unknown>
  }
  environment?: {
    type?: string
    import_path?: string
    force_build?: boolean
    delete?: boolean
    override_cpus?: number | null
    override_memory_mb?: number | null
    override_storage_mb?: number | null
    kwargs?: Record<string, unknown>
  }
  verifier?: {
    disable?: boolean
    override_timeout_sec?: number | null
    max_timeout_sec?: number | null
  }
  agents?: Array<{
    name?: string
    import_path?: string
    model_name?: string
    override_timeout_sec?: number | null
    max_timeout_sec?: number | null
    kwargs?: Record<string, unknown>
  }>
  datasets?: Array<{
    name?: string
    version?: string
    task_names?: string[]
    exclude_task_names?: string[]
    n_tasks?: number | null
    registry?: {
      name?: string
      url?: string
    }
  }>
  tasks?: unknown
}

export type HarborEvalResult = {
  n_trials: number
  n_errors: number
  metrics: Array<{ mean: number }>
  reward_stats: {
    reward: Record<string, string[]>
  }
  exception_stats: Record<string, string[]>
}

export type HarborJobResult = {
  id: string
  started_at: string
  finished_at: string | null
  n_total_trials: number
  stats: {
    n_trials: number
    n_errors: number
    evals: Record<string, HarborEvalResult>
  }
}

export type HarborTrialResult = {
  id?: string
  task_name?: string
  trial_name?: string
  verifier_result?: {
    rewards?: {
      reward?: number
    }
  }
  exception_info?: {
    exception_type?: string
    exception_message?: string
  } | null
}

export type JobStatus = 'complete' | 'partial' | 'in-progress'

export type TaskAggregate = {
  taskName: string
  trials: number
  passed: number
  failed: number
  errors: number
  meanReward: number | null
}

export type JobSummary = {
  jobId: string
  jobDirName: string
  jobPath: string
  configPath: string
  resultPath: string
  startedAt: string | null
  finishedAt: string | null
  dateLabel: string
  modelName: string | null
  sanitizedModelName: string | null
  taskCount: number
  totalTrialsExpected: number | null
  totalTrialsObserved: number
  passed: number
  failed: number
  errors: number
  meanReward: number | null
  status: JobStatus
  evalName: string | null
  taskBreakdown: TaskAggregate[]
  binaryMeta: TBenchMagnitudeMeta | null
}

export type SubmissionValidationIssue = {
  severity: 'error' | 'warning'
  code:
    | 'MISSING_META'
    | 'HASH_MISMATCH'
    | 'TIMEOUT_MULTIPLIER'
    | 'AGENT_TIMEOUT_OVERRIDE'
    | 'VERIFIER_TIMEOUT_OVERRIDE'
    | 'RESOURCE_OVERRIDE'
    | 'MISSING_TRIAL_RESULT'
    | 'MISSING_TRIAL_ARTIFACTS'
    | 'LOW_COVERAGE'
    | 'MODEL_MISMATCH'
    | 'INCOMPLETE_JOB'
  message: string
  jobId?: string
  taskName?: string
  trialDir?: string
}

export type CoverageRow = {
  taskName: string
  trials: number
  passed: number
  failed: number
  errors: number
}

export type SubmissionValidationResult = {
  issues: SubmissionValidationIssue[]
  coverage: CoverageRow[]
  binaryHashes: string[]
  modelName: string | null
  selectedJobs: JobSummary[]
}

export type SubmitOptions = {
  model?: string
  jobs?: string[]
  force?: boolean
  outputDir?: string
  interactive?: boolean
}

export const JOBS_DIR = path.join(process.cwd(), 'jobs')
export const DEFAULT_SUBMISSION_ROOT = path.join(process.cwd(), 'submissions', 'terminal-bench', '2.0')

export function sanitizeModelName(model: string): string {
  const tail = model.split('/').pop() ?? model
  return tail.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-+|-+$/g, '')
}

export async function safeReadJsonFile<T>(filePath: string): Promise<T | null> {
  try {
    return (await Bun.file(filePath).json()) as T
  } catch {
    return null
  }
}

export async function readMagnitudeMeta(jobPath: string): Promise<TBenchMagnitudeMeta | null> {
  return safeReadJsonFile<TBenchMagnitudeMeta>(path.join(jobPath, 'magnitude-meta.json'))
}

async function exists(filePath: string) {
  return existsSync(filePath)
}

async function isDirectory(dirPath: string) {
  try {
    return (await stat(dirPath)).isDirectory()
  } catch {
    return false
  }
}

export async function listJobDirectories(jobsDir = JOBS_DIR): Promise<string[]> {
  if (!(await isDirectory(jobsDir))) return []
  const entries = await readdir(jobsDir, { withFileTypes: true })
  return entries
    .filter(entry => entry.isDirectory())
    .map(entry => entry.name)
    .sort((a, b) => b.localeCompare(a))
}

export function extractTaskNameFromTrialDir(dirName: string): string {
  return dirName.split('__')[0] ?? dirName
}

function formatDateLabel(value: string | null, fallback: string) {
  if (!value) return fallback.replace(/__/g, ' ')
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return fallback.replace(/__/g, ' ')
  const y = date.getFullYear()
  const mo = String(date.getMonth() + 1).padStart(2, '0')
  const d = String(date.getDate()).padStart(2, '0')
  const h = String(date.getHours()).padStart(2, '0')
  const mi = String(date.getMinutes()).padStart(2, '0')
  const s = String(date.getSeconds()).padStart(2, '0')
  return `${y}-${mo}-${d} ${h}:${mi}:${s}`
}

function meanFromRewards(rewardBuckets: Record<string, string[]>) {
  let sum = 0
  let count = 0
  for (const [rewardStr, trials] of Object.entries(rewardBuckets)) {
    const reward = Number.parseFloat(rewardStr)
    if (!Number.isFinite(reward)) continue
    sum += reward * trials.length
    count += trials.length
  }
  return count > 0 ? sum / count : null
}

export async function summarizeJob(jobPath: string): Promise<JobSummary | null> {
  const jobDirName = path.basename(jobPath)
  const configPath = path.join(jobPath, 'config.json')
  const resultPath = path.join(jobPath, 'result.json')

  const config = await safeReadJsonFile<HarborJobConfig>(configPath)
  const result = await safeReadJsonFile<HarborJobResult>(resultPath)
  const binaryMeta = await readMagnitudeMeta(jobPath)

  const modelName = config?.agents?.[0]?.model_name ?? null
  const sanitizedModelName = modelName ? sanitizeModelName(modelName) : null

  if (!config && !result && !binaryMeta) return null

  const taskMap = new Map<string, TaskAggregate & { rewardSum: number; rewardCount: number }>()
  let passed = 0
  let failed = 0
  let errors = result?.stats.n_errors ?? 0
  let meanReward: number | null = null
  let evalName: string | null = null

  if (result) {
    const evalEntries = Object.entries(result.stats?.evals ?? {})
    if (evalEntries.length > 0) {
      evalName = evalEntries[0]?.[0] ?? null
    }

    let aggregateRewardSum = 0
    let aggregateRewardCount = 0

    for (const [name, evalResult] of evalEntries) {
      evalName ??= name
      const exceptionByTrial = new Map<string, string>()
      for (const [exceptionType, trials] of Object.entries(evalResult.exception_stats ?? {})) {
        for (const trialName of trials ?? []) {
          exceptionByTrial.set(trialName, exceptionType)
        }
      }

      for (const [rewardStr, trials] of Object.entries(evalResult.reward_stats?.reward ?? {})) {
        const reward = Number.parseFloat(rewardStr)
        if (!Number.isFinite(reward)) continue
        for (const trialName of trials) {
          const taskName = extractTaskNameFromTrialDir(trialName)
          const current = taskMap.get(taskName) ?? {
            taskName,
            trials: 0,
            passed: 0,
            failed: 0,
            errors: 0,
            meanReward: null,
            rewardSum: 0,
            rewardCount: 0,
          }
          current.trials += 1
          current.rewardSum += reward
          current.rewardCount += 1
          if (exceptionByTrial.has(trialName)) {
            current.errors += 1
          }
          if (reward >= 1) {
            current.passed += 1
            passed += 1
          } else {
            current.failed += 1
            failed += 1
          }
          aggregateRewardSum += reward
          aggregateRewardCount += 1
          taskMap.set(taskName, current)
        }
      }

      if (aggregateRewardCount > 0) {
        meanReward = aggregateRewardSum / aggregateRewardCount
      } else {
        meanReward ??= evalResult.metrics?.[0]?.mean ?? meanFromRewards(evalResult.reward_stats?.reward ?? {})
      }
    }
  }

  const taskBreakdown = Array.from(taskMap.values())
    .map(task => ({
      taskName: task.taskName,
      trials: task.trials,
      passed: task.passed,
      failed: task.failed,
      errors: task.errors,
      meanReward: task.rewardCount > 0 ? task.rewardSum / task.rewardCount : null,
    }))
    .sort((a, b) => a.taskName.localeCompare(b.taskName))

  const startedAt = result?.started_at ?? null
  const finishedAt = result?.finished_at ?? null
  const totalTrialsExpected = result?.n_total_trials ?? null
  const totalTrialsObserved = result?.stats?.n_trials ?? taskBreakdown.reduce((sum, task) => sum + task.trials, 0)
  const taskCount =
    taskBreakdown.length ||
    config?.datasets?.flatMap(dataset => dataset.task_names ?? []).filter(Boolean).length ||
    0

  let status: JobStatus = 'partial'
  if (!result) {
    status = 'partial'
  } else if (result.finished_at == null) {
    status = 'in-progress'
  } else if (result.stats.n_trials >= result.n_total_trials) {
    status = 'complete'
  } else {
    status = 'partial'
  }

  return {
    jobId: result?.id ?? config?.job_name ?? jobDirName,
    jobDirName,
    jobPath,
    configPath,
    resultPath,
    startedAt,
    finishedAt,
    dateLabel: formatDateLabel(startedAt, jobDirName),
    modelName,
    sanitizedModelName,
    taskCount,
    totalTrialsExpected,
    totalTrialsObserved,
    passed,
    failed,
    errors,
    meanReward,
    status,
    evalName,
    taskBreakdown,
    binaryMeta,
  }
}

export async function scanJobs(options: { modelFilter?: string; completedOnly?: boolean } = {}): Promise<JobSummary[]> {
  const dirs = await listJobDirectories()
  const summaries: JobSummary[] = []
  const filter = options.modelFilter?.toLowerCase()

  for (const dir of dirs) {
    const summary = await summarizeJob(path.join(JOBS_DIR, dir))
    if (!summary) continue
    if (filter && !(summary.modelName ?? '').toLowerCase().includes(filter)) continue
    if (options.completedOnly && summary.status !== 'complete') continue
    summaries.push(summary)
  }

  return summaries
}

export async function getTrialDirectories(jobPath: string): Promise<string[]> {
  if (!(await isDirectory(jobPath))) return []
  const entries = await readdir(jobPath, { withFileTypes: true })
  return entries
    .filter(entry => entry.isDirectory())
    .map(entry => entry.name)
    .filter(entry => !entry.startsWith('.'))
    .filter(entry => !['agent', 'verifier'].includes(entry))
    .filter(entry => !/^\d{4}-\d{2}-\d{2}__\d{2}-\d{2}-\d{2}$/.test(entry))
    .sort()
}

export async function validateJobForSubmission(job: JobSummary): Promise<SubmissionValidationIssue[]> {
  const issues: SubmissionValidationIssue[] = []
  const config = await safeReadJsonFile<HarborJobConfig>(job.configPath)

  if (!job.binaryMeta?.binaryHash) {
    issues.push({
      severity: 'error',
      code: 'MISSING_META',
      message: 'Missing or invalid magnitude-meta.json',
      jobId: job.jobDirName,
    })
  }

  if ((config?.timeout_multiplier ?? null) !== 1.0) {
    issues.push({
      severity: 'error',
      code: 'TIMEOUT_MULTIPLIER',
      message: 'timeout_multiplier must equal 1.0',
      jobId: job.jobDirName,
    })
  }

  for (const agent of config?.agents ?? []) {
    if (agent.override_timeout_sec != null || agent.max_timeout_sec != null) {
      issues.push({
        severity: 'error',
        code: 'AGENT_TIMEOUT_OVERRIDE',
        message: 'Agent timeout overrides are not allowed',
        jobId: job.jobDirName,
      })
      break
    }
  }

  if (config?.verifier?.override_timeout_sec != null || config?.verifier?.max_timeout_sec != null) {
    issues.push({
      severity: 'error',
      code: 'VERIFIER_TIMEOUT_OVERRIDE',
      message: 'Verifier timeout overrides are not allowed',
      jobId: job.jobDirName,
    })
  }

  if (
    config?.environment?.override_cpus != null ||
    config?.environment?.override_memory_mb != null ||
    config?.environment?.override_storage_mb != null
  ) {
    issues.push({
      severity: 'error',
      code: 'RESOURCE_OVERRIDE',
      message: 'Environment resource overrides are not allowed',
      jobId: job.jobDirName,
    })
  }

  if (job.status !== 'complete') {
    issues.push({
      severity: 'error',
      code: 'INCOMPLETE_JOB',
      message: `Job status is ${job.status}, must be complete`,
      jobId: job.jobDirName,
    })
  }

  const trialDirs = await getTrialDirectories(job.jobPath)
  for (const trialDir of trialDirs) {
    const trialPath = path.join(job.jobPath, trialDir)
    const resultPath = path.join(trialPath, 'result.json')
    const trialResult = await safeReadJsonFile<HarborTrialResult>(resultPath)
    if (!trialResult) {
      issues.push({
        severity: 'error',
        code: 'MISSING_TRIAL_RESULT',
        message: 'Missing or invalid trial result.json',
        jobId: job.jobDirName,
        taskName: extractTaskNameFromTrialDir(trialDir),
        trialDir,
      })
    }

    const requiredArtifacts = [
      'config.json',
      'result.json',
      'trial.log',
      path.join('agent', 'magnitude.txt'),
    ]
    for (const artifact of requiredArtifacts) {
      if (!(await exists(path.join(trialPath, artifact)))) {
        issues.push({
          severity: 'error',
          code: 'MISSING_TRIAL_ARTIFACTS',
          message: `Missing required artifact: ${artifact}`,
          jobId: job.jobDirName,
          taskName: extractTaskNameFromTrialDir(trialDir),
          trialDir,
        })
      }
    }

    const recommendedArtifacts = [
      path.join('verifier', 'reward.txt'),
      path.join('verifier', 'test-stdout.txt'),
      path.join('verifier', 'ctrf.json'),
    ]
    const missingRecommended: string[] = []
    for (const artifact of recommendedArtifacts) {
      if (!(await exists(path.join(trialPath, artifact)))) {
        missingRecommended.push(artifact)
      }
    }
    if (missingRecommended.length === recommendedArtifacts.length) {
      issues.push({
        severity: 'warning',
        code: 'MISSING_TRIAL_ARTIFACTS',
        message: 'Missing verifier artifacts (reward.txt, test-stdout.txt, ctrf.json)',
        jobId: job.jobDirName,
        taskName: extractTaskNameFromTrialDir(trialDir),
        trialDir,
      })
    }
  }

  return issues
}

export async function validateSubmissionSelection(
  jobs: JobSummary[],
  force: boolean,
): Promise<SubmissionValidationResult> {
  const issues: SubmissionValidationIssue[] = []
  const coverageMap = new Map<string, CoverageRow>()
  const modelNames = new Set(jobs.map(job => job.modelName).filter((v): v is string => Boolean(v)))
  const binaryHashes = Array.from(new Set(jobs.map(job => job.binaryMeta?.binaryHash).filter((v): v is string => Boolean(v))))

  for (const job of jobs) {
    issues.push(...(await validateJobForSubmission(job)))

    const trialDirs = await getTrialDirectories(job.jobPath)
    for (const trialDir of trialDirs) {
      const taskName = extractTaskNameFromTrialDir(trialDir)
      const trialResult = await safeReadJsonFile<HarborTrialResult>(path.join(job.jobPath, trialDir, 'result.json'))
      if (!trialResult) continue

      const row = coverageMap.get(taskName) ?? { taskName, trials: 0, passed: 0, failed: 0, errors: 0 }
      row.trials += 1

      const reward = trialResult.verifier_result?.rewards?.reward
      if (typeof reward === 'number') {
        if (reward >= 1) row.passed += 1
        else row.failed += 1
      }
      if (trialResult.exception_info) row.errors += 1
      coverageMap.set(taskName, row)
    }
  }

  if (modelNames.size > 1) {
    issues.push({
      severity: 'error',
      code: 'MODEL_MISMATCH',
      message: 'Selected jobs have different model names',
    })
  }

  if (binaryHashes.length > 1) {
    issues.push({
      severity: force ? 'warning' : 'error',
      code: 'HASH_MISMATCH',
      message: 'Selected jobs have different binary hashes',
    })
  }

  const coverage = Array.from(coverageMap.values()).sort((a, b) => a.taskName.localeCompare(b.taskName))
  for (const row of coverage) {
    if (row.trials < 5) {
      issues.push({
        severity: 'warning',
        code: 'LOW_COVERAGE',
        message: `Task ${row.taskName} has only ${row.trials} trial(s)`,
        taskName: row.taskName,
      })
    }
  }

  return {
    issues,
    coverage,
    binaryHashes,
    modelName: modelNames.size === 1 ? [...modelNames][0]! : null,
    selectedJobs: jobs,
  }
}

export async function copyJobForSubmission(sourceJobPath: string, destJobPath: string): Promise<void> {
  await cp(sourceJobPath, destJobPath, { recursive: true, force: true, errorOnExist: false })
}

function titleCase(value: string) {
  return value
    .split(/[-_\s]+/)
    .filter(Boolean)
    .map(part => part.charAt(0).toUpperCase() + part.slice(1))
    .join(' ')
}

function humanizeProvider(provider: string) {
  const map: Record<string, string> = {
    anthropic: 'Anthropic',
    openai: 'OpenAI',
    openrouter: 'OpenRouter',
    google: 'Google',
  }
  return map[provider] ?? titleCase(provider)
}

function humanizeModel(model: string) {
  const map: Record<string, string> = {
    'claude-sonnet-4-6': 'Claude Sonnet 4 (6)',
    'gpt-5.4': 'GPT-5.4',
    'gpt-5.3-codex': 'GPT-5.3 Codex',
    'gpt-5.3-codex-spark': 'GPT-5.3 Codex Spark',
  }
  return map[model] ?? model
}

export function generateMetadataYaml(params: {
  modelName: string
  binaryHash: string | null
  jobIds: string[]
  createdAt: string
}): string {
  const [provider, ...modelParts] = params.modelName.split('/')
  const model = modelParts.join('/') || params.modelName
  const lines = [
    'agent_url: https://github.com/magnitude-dev/magnitude',
    'agent_display_name: "Magnitude"',
    'agent_org_display_name: "Magnitude"',
    '',
    'models:',
    `  - model_name: "${model}"`,
    `    model_provider: "${provider}"`,
    `    model_display_name: "${humanizeModel(model)}"`,
    `    model_org_display_name: "${humanizeProvider(provider)}"`,
    '',
    '# Magnitude submission metadata',
    `created_at: "${params.createdAt}"`,
  ]

  if (params.binaryHash) {
    lines.push(`binary_sha256: "${params.binaryHash}"`)
  } else {
    lines.push('binary_sha256: null')
    lines.push('binary_sha256_note: "multiple binary hashes across selected jobs"')
  }

  lines.push('source_jobs:')
  for (const jobId of params.jobIds) {
    lines.push(`  - "${jobId}"`)
  }

  return `${lines.join('\n')}\n`
}

function formatMean(value: number | null) {
  return value == null ? '-' : value.toFixed(3)
}

function colorStatus(status: JobStatus) {
  switch (status) {
    case 'complete':
      return ansis.green(status)
    case 'in-progress':
      return ansis.yellow(status)
    case 'partial':
    default:
      return ansis.yellow(status)
  }
}

function renderTable(headers: string[], rows: string[][], colorFns?: Array<((value: string) => string) | null>) {
  const widths = headers.map((header, index) => Math.max(header.length, ...rows.map(row => row[index]?.length ?? 0)))
  const headerLine = headers.map((header, index) => header.padEnd(widths[index]!)).join('  ')
  const sepLine = widths.map(width => '─'.repeat(width)).join('  ')
  const body = rows.map(row =>
    row
      .map((cell, index) => {
        const padded = cell.padEnd(widths[index]!)
        const fn = colorFns?.[index]
        return fn ? fn(padded) : padded
      })
      .join('  '),
  )
  return [headerLine, sepLine, ...body].join('\n')
}

export async function jobsMain(options: { modelFilter?: string; verbose?: boolean } = {}) {
  if (!(await isDirectory(JOBS_DIR))) {
    clack.log.info('No jobs directory found')
    return
  }

  const jobs = await scanJobs({ modelFilter: options.modelFilter })
  if (jobs.length === 0) {
    clack.log.info('No jobs found')
    return
  }

  const rows = jobs.map(job => [
    job.jobDirName,
    job.modelName ?? 'unknown',
    String(job.taskCount),
    String(job.passed),
    String(job.failed),
    String(job.errors),
    formatMean(job.meanReward),
    job.status,
    job.binaryMeta?.binaryHash?.slice(0, 8) ?? '-',
    job.dateLabel,
  ])

  const statusColor = (v: string) => {
    const s = v.trim()
    if (s === 'complete') return ansis.green(v)
    if (s === 'in-progress') return ansis.yellow(v)
    return ansis.red(v)
  }

  console.log(
    renderTable(
      ['Job ID', 'Model', 'Tasks', 'Passed', 'Failed', 'Errors', 'Mean Reward', 'Status', 'Hash', 'Date'],
      rows,
      [
        null,                                                           // Job ID
        ansis.cyan,                                                     // Model
        null,                                                           // Tasks
        v => v.trim() !== '0' ? ansis.green(v) : v,                    // Passed
        v => v.trim() !== '0' ? ansis.red(v) : ansis.dim(v),           // Failed
        v => v.trim() !== '0' ? ansis.red(v) : ansis.dim(v),           // Errors
        v => {                                                          // Mean Reward
          const n = parseFloat(v.trim())
          if (isNaN(n)) return ansis.dim(v)
          if (n >= 0.7) return ansis.green(v)
          if (n >= 0.4) return ansis.yellow(v)
          return ansis.red(v)
        },
        statusColor,                                                    // Status
        v => ansis.dim(v),                                              // Hash
        v => ansis.dim(v),                                              // Date
      ],
    ),
  )

  if (options.verbose) {
    for (const job of jobs) {
      console.log()
      console.log(ansis.bold(job.jobDirName))
      if (job.taskBreakdown.length === 0) {
        console.log(ansis.dim('  No task breakdown available'))
        continue
      }
      const taskRows = job.taskBreakdown.map(task => [
        task.taskName,
        String(task.trials),
        String(task.passed),
        String(task.failed),
        String(task.errors),
        formatMean(task.meanReward),
      ])
      console.log(
        renderTable(['Task', 'Trials', 'Passed', 'Failed', 'Errors', 'Mean'], taskRows)
          .split('\n')
          .map(line => `  ${line}`)
          .join('\n'),
      )
    }
  }
}

function printValidationReport(validation: SubmissionValidationResult) {
  console.log()
  console.log(ansis.bold('Validation'))
  console.log(`  Model: ${validation.modelName ?? 'unknown'}`)
  console.log(
    `  Binary hashes: ${
      validation.binaryHashes.length > 0 ? validation.binaryHashes.map(hash => hash.slice(0, 12)).join(', ') : 'none'
    }`,
  )

  if (validation.issues.length > 0) {
    for (const issue of validation.issues) {
      const prefix = issue.severity === 'error' ? ansis.red('error') : ansis.yellow('warning')
      const target = [issue.jobId, issue.trialDir].filter(Boolean).join(' / ')
      console.log(`  ${prefix} ${issue.code}${target ? ` [${target}]` : ''}: ${issue.message}`)
    }
  } else {
    console.log(ansis.green('  No validation issues'))
  }

  if (validation.coverage.length > 0) {
    console.log()
    console.log(ansis.bold('Coverage'))
    const rows = validation.coverage.map(row => [
      row.taskName,
      String(row.trials),
      String(row.passed),
      String(row.failed),
      String(row.errors),
    ])
    console.log(renderTable(['Task', 'Trials', 'Passed', 'Failed', 'Errors'], rows))
  }
}

export async function submitMain(opts: SubmitOptions) {
  const outputRoot = opts.outputDir ?? DEFAULT_SUBMISSION_ROOT

  if (!(await isDirectory(JOBS_DIR))) {
    clack.log.error('No jobs directory found')
    process.exit(1)
  }

  const completedJobs = await scanJobs({ completedOnly: true })
  if (completedJobs.length === 0) {
    clack.log.error('No completed jobs found')
    process.exit(1)
  }

  const jobsByModel = new Map<string, JobSummary[]>()
  for (const job of completedJobs) {
    if (!job.modelName) continue
    const list = jobsByModel.get(job.modelName) ?? []
    list.push(job)
    jobsByModel.set(job.modelName, list)
  }

  if (jobsByModel.size === 0) {
    clack.log.error('No completed jobs with model metadata found')
    process.exit(1)
  }

  let model = opts.model
  if (!model) {
    const models = Array.from(jobsByModel.keys()).sort()
    const selected = await clack.select({
      message: 'Select model',
      options: models.map(value => ({
        value,
        label: `${value} (${jobsByModel.get(value)?.length ?? 0} jobs)`,
      })),
    })
    if (clack.isCancel(selected)) {
      clack.cancel('Cancelled')
      return
    }
    model = selected as string
  }

  const modelJobs = jobsByModel.get(model)
  if (!modelJobs) {
    clack.log.error(`No completed jobs found for model: ${model}`)
    console.log(`Available models: ${Array.from(jobsByModel.keys()).sort().join(', ')}`)
    process.exit(1)
  }

  console.log()
  console.log(ansis.bold(`Completed jobs for ${model}`))
  console.log(
    renderTable(
      ['Job ID', 'Date', 'Tasks', 'Pass Rate', 'Mean', 'Binary'],
      modelJobs.map(job => {
        const denominator = job.passed + job.failed
        const passRate = denominator > 0 ? `${job.passed}/${denominator}` : '-'
        const binary = job.binaryMeta?.binaryHash ? job.binaryMeta.binaryHash.slice(0, 12) : '-'
        return [job.jobDirName, job.dateLabel, String(job.taskCount), passRate, formatMean(job.meanReward), binary]
      }),
    ),
  )

  let selectedJobs: JobSummary[] = modelJobs
  if (opts.jobs?.length) {
    const byId = new Map(modelJobs.map(job => [job.jobDirName, job]))
    const missing = opts.jobs.filter(id => !byId.has(id))
    if (missing.length > 0) {
      clack.log.error(`Unknown job IDs: ${missing.join(', ')}`)
      console.log(`Valid IDs: ${modelJobs.map(job => job.jobDirName).join(', ')}`)
      process.exit(1)
    }
    selectedJobs = opts.jobs.map(id => byId.get(id)!).filter(Boolean)
  } else if (opts.interactive !== false) {
    const picks = await clack.multiselect({
      message: 'Select jobs to package',
      required: true,
      initialValues: modelJobs.map(job => job.jobDirName),
      options: modelJobs.map(job => {
        const denominator = job.passed + job.failed
        const passRate = denominator > 0 ? `${job.passed}/${denominator} passed` : 'no trials'
        return {
          value: job.jobDirName,
          label: `${job.jobDirName} · ${job.taskCount} tasks · ${passRate} · mean ${formatMean(job.meanReward)}`,
        }
      }),
    })
    if (clack.isCancel(picks)) {
      clack.cancel('Cancelled')
      return
    }
    const selected = new Set(picks as string[])
    selectedJobs = modelJobs.filter(job => selected.has(job.jobDirName))
  }

  if (selectedJobs.length === 0) {
    clack.log.error('No jobs selected')
    process.exit(1)
  }

  const validation = await validateSubmissionSelection(selectedJobs, opts.force ?? false)
  printValidationReport(validation)

  const hardErrors = validation.issues.filter(issue => issue.severity === 'error')
  if (hardErrors.length > 0) {
    clack.log.error('Submission validation failed')
    process.exit(1)
  }

  const finalDir = path.join(outputRoot, `magnitude__${sanitizeModelName(model)}`)
  const existing = await isDirectory(finalDir)
  if (existing) {
    const entries = await readdir(finalDir).catch(() => [])
    if (entries.length > 0 && !opts.force) {
      clack.log.error(`Submission directory already exists and is not empty: ${finalDir}`)
      process.exit(1)
    }
  }

  console.log()
  console.log(`Destination: ${finalDir}`)

  if (opts.interactive !== false) {
    const confirmed = await clack.confirm({
      message: 'Create submission package?',
      initialValue: true,
    })
    if (clack.isCancel(confirmed) || !confirmed) {
      clack.cancel('Cancelled')
      return
    }
  }

  await Bun.write(path.join(finalDir, '.keep'), '')
  for (const job of selectedJobs) {
    await copyJobForSubmission(job.jobPath, path.join(finalDir, job.jobDirName))
  }

  const metadataYaml = generateMetadataYaml({
    modelName: model,
    binaryHash: validation.binaryHashes.length === 1 ? validation.binaryHashes[0]! : null,
    jobIds: selectedJobs.map(job => job.jobDirName),
    createdAt: new Date().toISOString(),
  })
  await Bun.write(path.join(finalDir, 'metadata.yaml'), metadataYaml)

  clack.outro(`Created submission package at ${finalDir}`)
}