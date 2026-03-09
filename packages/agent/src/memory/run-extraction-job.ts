import { writeFile } from 'fs/promises'
import { logger } from '@magnitudedev/logger'
import { initializeProviderState, secondary } from '@magnitudedev/providers'
import { applyMemoryDiff, enforceLineBudget, ensureMemoryFile, readMemory, type MemoryDiff } from './memory-file'
import { buildExtractionTranscript, readEventsJsonl } from './transcript'
import { markJobPending, markJobRunning, readJob, removeJob } from './job-queue'

function parseJsonDiff(raw: unknown): MemoryDiff | null {
  if (!raw || typeof raw !== 'object') return null
  const val = raw as any
  return {
    additions: Array.isArray(val.additions) ? val.additions : [],
    updates: Array.isArray(val.updates) ? val.updates : [],
    deletions: Array.isArray(val.deletions) ? val.deletions : [],
  }
}

export async function runExtractionJobFromFile(jobFilePath: string): Promise<void> {
  const job = await readJob(jobFilePath)
  await markJobRunning(jobFilePath, job)

  try {
    await initializeProviderState()

    await ensureMemoryFile(job.cwd)
    const [events, currentMemory] = await Promise.all([
      readEventsJsonl(job.eventsPath),
      readMemory(job.cwd),
    ])

    const transcript = buildExtractionTranscript(events)

    const { result } = await secondary.extractMemoryDiff(transcript, currentMemory, {
      forkId: null,
      callType: 'extract-memory-diff',
    })

    const diff = parseJsonDiff(result)
    if (!diff) {
      logger.warn('[memory] extraction returned invalid diff object; leaving job pending')
      await markJobPending(jobFilePath, { ...job, status: 'pending', attempts: job.attempts + 1 })
      return
    }

    const applied = applyMemoryDiff(currentMemory, diff)
    const budgeted = enforceLineBudget(applied.updated, 150)
    if (applied.changed || budgeted !== currentMemory) {
      await writeFile(job.memoryPath, budgeted, 'utf8')
    }

    await removeJob(jobFilePath)
  } catch (error) {
    logger.warn(`[memory] extraction job failed (${jobFilePath}): ${String(error)}`)
    await markJobPending(jobFilePath, { ...job, status: 'pending', attempts: job.attempts + 1 }).catch(() => {})
  }
}

async function main(): Promise<void> {
  const args = process.argv.slice(2)
  const idx = args.indexOf('--job')
  if (idx === -1 || !args[idx + 1]) {
    logger.warn('[memory] missing --job path')
    return
  }
  const jobPath = args[idx + 1]!
  await runExtractionJobFromFile(jobPath)
}

if (import.meta.main) {
  main().catch((error) => {
    logger.warn(`[memory] worker crashed: ${String(error)}`)
  })
}