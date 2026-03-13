import { writeFile } from 'fs/promises'
import { Effect, Layer } from 'effect'
import { logger } from '@magnitudedev/logger'
import { bootstrapProviderRuntime, ModelResolver, makeModelResolver, makeNoopTracer, makeProviderRuntimeLive, ExtractMemoryDiff } from '@magnitudedev/providers'
import { applyMemoryDiff, enforceLineBudget, ensureMemoryFile, readMemory, type MemoryDiff } from './memory-file'
import { withTraceScope } from '../tracing'
import { buildExtractionTranscript, readEventsJsonl } from './transcript'
import { markJobPending, markJobRunning, readJob, removeJob } from './job-queue'

function parseJsonDiff(raw: unknown): MemoryDiff | null {
  if (!raw || typeof raw !== 'object') return null
  const val = raw as Record<string, unknown>
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
    const providerRuntime = makeProviderRuntimeLive()
    await Effect.runPromise(bootstrapProviderRuntime.pipe(Effect.provide(providerRuntime)))

    await ensureMemoryFile(job.cwd)
    const [events, currentMemory] = await Promise.all([
      readEventsJsonl(job.eventsPath),
      readMemory(job.cwd),
    ])

    const transcript = buildExtractionTranscript(events)

    const tracerLayer = makeNoopTracer()
    const resolverLayer = Layer.provide(makeModelResolver(), providerRuntime)

    const result = await Effect.runPromise(
      Effect.gen(function* () {
        const runtime = yield* ModelResolver
        const model = yield* runtime.resolve('secondary')
        return yield* withTraceScope(
          { metadata: { callType: 'extract-memory-diff', forkId: null } },
          model.invoke(
            ExtractMemoryDiff,
            { transcript, currentMemory },
          ),
        )
      }).pipe(Effect.provide(Layer.merge(resolverLayer, Layer.merge(providerRuntime, tracerLayer)))),
    )

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
  const jobPath = args[idx + 1]
  if (!jobPath) {
    logger.warn('[memory] missing --job path')
    return
  }
  await runExtractionJobFromFile(jobPath)
}

if (import.meta.main) {
  main().catch((error) => {
    logger.warn(`[memory] worker crashed: ${String(error)}`)
  })
}