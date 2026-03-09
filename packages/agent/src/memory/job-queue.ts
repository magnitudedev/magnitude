import { promises as fs } from 'fs'
import * as fsSync from 'fs'
import * as os from 'os'
import * as path from 'path'
import { fileURLToPath } from 'url'
import { logger } from '@magnitudedev/logger'
import { createId } from '../util/id'


export interface MemoryExtractionJob {
  jobId: string
  sessionId: string
  cwd: string
  eventsPath: string
  memoryPath: string
  createdAt: string
  attempts: number
  status: 'pending' | 'running'
}

export function getSessionRoot(): string {
  return path.join(os.homedir(), '.magnitude', 'sessions')
}

export function getPendingDir(): string {
  return path.join(getSessionRoot(), '.pending-memory-extraction')
}

export function createMemoryExtractionJob(params: {
  sessionId: string
  cwd: string
  eventsPath: string
  memoryPath: string
}): MemoryExtractionJob {
  return {
    jobId: `${params.sessionId}-${Date.now()}-${createId()}`,
    sessionId: params.sessionId,
    cwd: params.cwd,
    eventsPath: params.eventsPath,
    memoryPath: params.memoryPath,
    createdAt: new Date().toISOString(),
    attempts: 0,
    status: 'pending',
  }
}

export function writePendingJobSync(job: MemoryExtractionJob): string {
  const dir = getPendingDir()
  fsSync.mkdirSync(dir, { recursive: true })
  const file = path.join(dir, `${job.jobId}.json`)
  fsSync.writeFileSync(file, JSON.stringify(job, null, 2), 'utf8')
  return file
}

export async function listPendingJobs(): Promise<string[]> {
  const dir = getPendingDir()
  try {
    const entries = await fs.readdir(dir, { withFileTypes: true })
    return entries
      .filter((e) => e.isFile() && e.name.endsWith('.json'))
      .map((e) => path.join(dir, e.name))
      .sort()
  } catch (error) {
    const code = (error as NodeJS.ErrnoException).code
    if (code === 'ENOENT') return []
    throw error
  }
}

export async function readJob(jobPath: string): Promise<MemoryExtractionJob> {
  const raw = await fs.readFile(jobPath, 'utf8')
  return JSON.parse(raw) as MemoryExtractionJob
}

export async function markJobRunning(jobPath: string, job: MemoryExtractionJob): Promise<void> {
  const next: MemoryExtractionJob = { ...job, status: 'running', attempts: (job.attempts ?? 0) + 1 }
  await fs.writeFile(jobPath, JSON.stringify(next, null, 2), 'utf8')
}

export async function markJobPending(jobPath: string, job: MemoryExtractionJob): Promise<void> {
  const next: MemoryExtractionJob = { ...job, status: 'pending' }
  await fs.writeFile(jobPath, JSON.stringify(next, null, 2), 'utf8')
}

export async function removeJob(jobPath: string): Promise<void> {
  await fs.rm(jobPath, { force: true })
}

export function spawnDetachedMemoryExtractionWorker(jobFilePath: string): void {
  try {
    const scriptPath = fileURLToPath(new URL('./run-extraction-job.ts', import.meta.url))
    const child = Bun.spawn(
      ['bun', 'run', scriptPath, '--job', jobFilePath],
      { detached: true, stdio: ['ignore', 'ignore', 'ignore'] }
    )
    child.unref()
  } catch (error) {
    logger.warn(`[memory] failed to spawn detached extraction worker: ${String(error)}`)
  }
}

const MAX_JOB_ATTEMPTS = 5

export async function drainPendingJobsOnStartup(): Promise<void> {
  const files = await listPendingJobs()
  for (const file of files) {
    try {
      const job = await readJob(file)

      if (job.attempts >= MAX_JOB_ATTEMPTS) {
        logger.warn(`[memory] dropping extraction job after ${job.attempts} attempts: ${file}`)
        await removeJob(file)
        continue
      }

      // Spawn detached worker so it doesn't block the main process
      spawnDetachedMemoryExtractionWorker(file)
    } catch (error) {
      logger.warn(`[memory] startup drain failed for ${file}: ${String(error)}`)
    }
  }
}