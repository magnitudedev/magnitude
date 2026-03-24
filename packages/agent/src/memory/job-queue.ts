import { fileURLToPath } from 'url'
import { logger } from '@magnitudedev/logger'
import type { StorageClient } from '@magnitudedev/storage'
import type { MagnitudeSlot } from '../model-slots'
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

export async function writePendingJob(storage: StorageClient<MagnitudeSlot>, job: MemoryExtractionJob): Promise<string> {
  return storage.memoryJobs.enqueue(job)
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

export async function drainPendingJobsOnStartup(storage: StorageClient<MagnitudeSlot>): Promise<void> {
  const jobFiles = await storage.memoryJobs.list()
  for (const jobFile of jobFiles) {
    try {
      const job = await storage.memoryJobs.read({ filePath: jobFile })

      if (job.attempts >= MAX_JOB_ATTEMPTS) {
        logger.warn(`[memory] dropping extraction job after ${job.attempts} attempts: ${jobFile}`)
        await storage.memoryJobs.remove(job.jobId)
        continue
      }

      await storage.memoryJobs.markRunning(job.jobId, job)
      spawnDetachedMemoryExtractionWorker(jobFile)
    } catch (error) {
      logger.warn(`[memory] startup drain failed for ${jobFile}: ${String(error)}`)
    }
  }
}