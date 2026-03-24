import { Effect, Layer } from 'effect'
import { logger } from '@magnitudedev/logger'
import { createStorageClient } from '@magnitudedev/storage'
import { bootstrapProviderRuntime, ModelResolver, makeModelResolver, makeNoopTracer, makeProviderRuntimeLive, ExtractMemoryDiff } from '@magnitudedev/providers'
import { MAGNITUDE_SLOTS, type MagnitudeSlot } from '../model-slots'
import { applyMemoryDiff, enforceLineBudget, ensureMemoryFile, readMemory, writeMemory, type MemoryDiff } from './memory-file'
import { withTraceScope } from '../tracing'
import { buildExtractionTranscript, readEventsJsonl } from './transcript'

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
  const storage = await createStorageClient<MagnitudeSlot>()
  const job = await storage.memoryJobs.read({ filePath: jobFilePath })
  await storage.memoryJobs.markRunning(job.jobId, job)

  try {
    const providerRuntime = makeProviderRuntimeLive<MagnitudeSlot>()
    await Effect.runPromise(bootstrapProviderRuntime<MagnitudeSlot>({ slots: MAGNITUDE_SLOTS }).pipe(Effect.provide(providerRuntime)))

    await ensureMemoryFile(storage)
    const [events, currentMemory] = await Promise.all([
      readEventsJsonl(job.eventsPath),
      readMemory(storage),
    ])

    const transcript = buildExtractionTranscript(events)

    const tracerLayer = makeNoopTracer()
    const resolverLayer = Layer.provide(makeModelResolver<MagnitudeSlot>(), providerRuntime)

    const result = await Effect.runPromise(
      Effect.gen(function* () {
        const runtime = yield* ModelResolver
        const model = yield* runtime.resolve('lead')
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
      await storage.memoryJobs.markPending(job.jobId, { ...job, status: 'pending', attempts: job.attempts + 1 })
      return
    }

    const applied = applyMemoryDiff(currentMemory, diff)
    const budgeted = enforceLineBudget(applied.updated, 150)
    if (applied.changed || budgeted !== currentMemory) {
      await writeMemory(storage, budgeted)
    }

    await storage.memoryJobs.remove(job.jobId)
  } catch (error) {
    logger.warn(`[memory] extraction job failed (${jobFilePath}): ${String(error)}`)
    await storage.memoryJobs.markPending(job.jobId, { ...job, status: 'pending', attempts: job.attempts + 1 }).catch(() => {})
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