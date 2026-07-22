/**
 * DetachedShellRegistry — Live implementation (Effect-native).
 *
 * Owns the entire child process lifecycle: spawn, pipes, detach,
 * exit handling, event publishing, and stream teardown.
 *
 * Uses `Bun.spawn` (the native Bun subprocess API), not Node's
 * `child_process.spawn`. Under Bun, a missing shell binary via
 * `child_process.spawn` raises an unhandled EventEmitter `'error'`
 * event that escapes all JS-level catches and crashes the process.
 * `Bun.spawn` throws a JS exception synchronously on missing command,
 * which Effect catches normally. See bugs/26-06-21/ps-spawn-enoent-leak.md.
 *
 * Invariants:
 *   1. Exit handler fiber is the ONLY path that calls finalize().
 *   2. Kill sequences (SIGTERM → SIGKILL) only escalate signals.
 *   3. SIGKILL always produces an exit (Bun reaps via waitpid).
 *   4. Stream lifecycle: tee → read to queue + read to file → flush → end.
 *      Output files are preserved.
 *   5. All cleanup runs in Effect.uninterruptible.
 */

import { Context, Effect, Layer, Deferred, Scope, Cause, Option, Exit, Queue, Stream, SubscriptionRef, Runtime } from 'effect'
import type { ReadableSubprocess, FileSink } from 'bun'
import { DetachedShellRegistry, type DetachedShellRegistryService, type ExecuteDetachedOutput, type ShellOutputChunk } from './detached-process-registry'
import type { WorkerBusService } from '@magnitudedev/event-core'
import type { AppEvent } from '../events'
import { mkdirSync, readFileSync, rmSync } from 'fs'
import * as path from 'path'
import { logger } from '@magnitudedev/logger'
import { discoverDescendants } from './ps-tree'

// ── Types ────────────────────────────────────────────────────────────

interface TrackedProcess {
  readonly pid: number
  readonly child: ReadableSubprocess
  readonly command: string
  readonly forkId: string | null
  readonly ownerAgentId: string | undefined
  readonly turnId: string
  readonly toolCallId: string
  readonly stdoutPath: string
  readonly stderrPath: string
  readonly startedAt: number
  readonly stdoutWriter: FileSink
  readonly stderrWriter: FileSink
}

interface ProcessTimers {
  readonly detach: ReturnType<typeof setTimeout> | undefined
  readonly sigkill: ReturnType<typeof setTimeout> | undefined
}

// ── Signal helpers ──────────────────────────────────────────────────

function signalToNum(signal: string | NodeJS.Signals): number {
  const map: Record<string, number> = {
    SIGHUP: 1, SIGINT: 2, SIGQUIT: 3, SIGILL: 4, SIGTRAP: 5,
    SIGABRT: 6, SIGBUS: 7, SIGFPE: 8, SIGKILL: 9, SIGUSR1: 10,
    SIGSEGV: 11, SIGUSR2: 12, SIGPIPE: 13, SIGALRM: 14, SIGTERM: 15,
  }
  return map[signal] ?? 1
}

function ensureProcessesDir(scratchpadPath: string): string {
  const dir = path.join(scratchpadPath, 'processes')
  mkdirSync(dir, { recursive: true })
  return dir
}

// ── File sink lifecycle ─────────────────────────────────────────────

/**
 * Flush and close a Bun FileSink, ensuring all buffered data is written
 * to disk. Logs and continues on_failure — file write errors during
 * cleanup are non-fatal; the exit event has already been published.
 */
function closeFileSink(writer: FileSink): Effect.Effect<void, never, never> {
  return Effect.promise(async () => {
      await writer.flush()
      await writer.end()
  }).pipe(Effect.catchAllCause(() => Effect.void))
}

/**
 * Close a FileSink, then read and delete the backing file.
 * Returns the file content (empty string if file missing or close failed).
 */
function readAndDeleteAfterClose(
  writer: FileSink,
  filePath: string,
): Effect.Effect<string, never, never> {
  return Effect.gen(function* () {
    yield* closeFileSink(writer)
    const content = yield* Effect.sync(() => readFileSync(filePath, 'utf8')).pipe(
      Effect.catchAllCause(() => Effect.succeed('')),
    )
    yield* Effect.sync(() => rmSync(filePath)).pipe(
      Effect.catchAllCause(() => Effect.void),
    )
    return content
  })
}

// ── Subprocess exit await ───────────────────────────────────────────

/**
 * Wait for a Bun Subprocess to exit, returning its exit code and signal.
 *
 * `proc.exited` resolves with the numeric exit code on natural exit or
 * signal termination (Bun reaps via waitpid). On a proc-level error
 * (rare — Bun reports spawning failures as synchronous throws before
 * `proc` is returned), `exited` rejects; we treat that as exit code 1.
 */
function waitForProcExit(
  child: ReadableSubprocess,
): Effect.Effect<[number, NodeJS.Signals | null], never, never> {
  return Effect.promise(async () => {
      const code = await child.exited
      return [code, child.signalCode] as [number, NodeJS.Signals | null]
  }).pipe(
    Effect.catchAllCause(() => Effect.succeed<[number, NodeJS.Signals | null]>([1, null])),
  )
}

// ── Stream consumer (queue + file) ──────────────────────────────────

/**
 * Lifecycle result for a single stream consumer.
 * Ends with 'end' when the stream closes naturally (child exit),
 * or 'error' with a message if the stream errored.
 */
type StreamConsumerResult = { readonly _tag: 'end' } | { readonly _tag: 'error'; readonly reason: string }

/**
 * Consume a Web ReadableStream<Uint8Array>, writing each chunk to a Bun
 * FileSink and offering decoded text to the output queue.
 *
 * Returns a Promise that resolves when the stream ends. The consumer
 * ends naturally when the child process exits and Bun closes the pipe.
 */
function consumeStreamToFileAndQueue(
  stream: ReadableStream<Uint8Array> | null,
  writer: FileSink,
  queue: Queue.Queue<ShellOutputChunk>,
  isStderr: boolean,
): Promise<StreamConsumerResult> {
  if (stream === null) {
    return Promise.resolve<StreamConsumerResult>({ _tag: 'end' })
  }
  const reader = stream.getReader()
  const decoder = new TextDecoder()
  return (async (): Promise<StreamConsumerResult> => {
    try {
      // eslint-disable-next-line no-constant-condition
      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        if (value) {
          writer.write(value)
          const text = decoder.decode(value, { stream: true })
          queue.unsafeOffer(isStderr ? { stdout: '', stderr: text } : { stdout: text, stderr: '' })
        }
      }
      return { _tag: 'end' }
    } catch (error) {
      return { _tag: 'error', reason: String(error) }
    } finally {
      reader.releaseLock()
    }
  })()
}

// ── Spawn helper ───────────────────────────────────────────────────

/**
 * Spawn the detached shell. Returns the Subprocess on success, or an
 * Error if the shell binary could not be spawned (ENOENT, etc.).
 *
 * Bun.spawn throws synchronously on spawn-time failure — unlike Node's
 * `child_process.spawn`, which raises an unhandled EventEmitter 'error'
 * event that escapes all JS-level catches. The try/catch here is the
 * boundary: callers receive a typed error and surface it to the LLM
 * instead of the process crashing.
 */
function spawnDetachedShell(
  shellPath: string,
  command: string,
  cwd: string,
  env: Record<string, string | undefined>,
): ReadableSubprocess {
  return Bun.spawn({
    cmd: [shellPath, '-c', command],
    cwd,
    env,
    stdin: 'ignore',
    stdout: 'pipe',
    stderr: 'pipe',
  })
}

// ── Implementation ──────────────────────────────────────────────────

export const makeDetachedShellRegistryService: Effect.Effect<DetachedShellRegistryService> = Effect.gen(function* () {
  const processes = new Map<number, TrackedProcess>()
  const byFork = new Map<string | null, Set<number>>()
  const timersByPid = new Map<number, ProcessTimers>()
  let bus: WorkerBusService<AppEvent> | null = null
  const activeCount = yield* SubscriptionRef.make(0)
  const runtime = yield* Effect.runtime<never>()
  const publishActiveCount = Effect.sync(() => processes.size).pipe(
    Effect.flatMap((count) => SubscriptionRef.set(activeCount, count)),
  )

  // ── Event Publishing ───────────────────────────────────────────────

  function publish(event: AppEvent): void {
    if (bus === null) {
      logger.warn({ type: event.type }, '[DetachedShellRegistry] Bus not bound, dropping event')
      return
    }
    Runtime.runFork(runtime)(
      bus.publish(event).pipe(
        Effect.tapErrorCause((cause) =>
          Effect.sync(() =>
            logger.error(
              { cause, eventType: event.type },
              '[DetachedShellRegistry] Event publish failed',
            ),
          ),
        ),
        Effect.catchAllCause(() => Effect.void),
      ),
    )
  }

  // ── Finalize (exit handler fiber ONLY) ──────────────────────────

  /**
   * Finalize a process that has exited while under registry management.
   * Called exclusively from the registry exit handler fiber. Runs
   * uninterruptible — must complete stream teardown even if the parent
   * Effect is interrupted.
   */
  function finalize(proc: TrackedProcess, exitCode: number): Effect.Effect<void, never, never> {
    return Effect.gen(function* () {
      // 1. Clear escalation timers
      const t = timersByPid.get(proc.pid)
      if (t) {
        if (t.sigkill) clearTimeout(t.sigkill)
        if (t.detach) clearTimeout(t.detach)
        timersByPid.delete(proc.pid)
      }

      // 2. Delete from tracking maps immediately — prevents re-kill races
      processes.delete(proc.pid)
      byFork.get(proc.forkId)?.delete(proc.pid)
      yield* publishActiveCount

      // 3. Flush + close file sinks so all buffered data is on disk
      yield* Effect.all(
        [
          closeFileSink(proc.stdoutWriter),
          closeFileSink(proc.stderrWriter),
        ],
        { concurrency: 2, discard: true },
      )

      // 4. Output files are preserved so the LLM can read them after the
      //    process exits. The exit notification only carries file paths,
      //    not content — deleting here would make those paths stale.
      // 5. Publish completion events
      publish({
        type: 'shell_process_exited',
        forkId: proc.forkId,
        pid: proc.pid,
        command: proc.command,
        exitCode,
      })
      publish({
        type: 'shell_completed',
        forkId: proc.forkId,
        pid: proc.pid,
        command: proc.command,
        exitCode,
      })
    }).pipe(Effect.uninterruptible)
  }

  // ── Kill escalation ──────────────────────────────────────────────

  /**
   * Kill a process and all its descendants.
   *
   * Discovers the process tree via `ps`, kills descendants bottom-up
   * with SIGTERM, then kills the root. A 2-second SIGKILL escalation
   * timer is set on the root PID as a backstop for processes that
   * ignore SIGTERM.
   */
  function performKill(proc: TrackedProcess): Effect.Effect<void, never, never> {
    return Effect.gen(function* () {
      if (!processes.has(proc.pid)) return

      // Discover and kill descendants bottom-up
      const descendants = yield* discoverDescendants(proc.pid)
      for (const pid of [...descendants].reverse()) {
        yield* Effect.sync(() => process.kill(pid, 'SIGTERM')).pipe(
          Effect.catchAllCause(() => Effect.void),
        )
      }

      // Kill root
      yield* Effect.sync(() => proc.child.kill('SIGTERM')).pipe(
        Effect.catchAllCause(() => Effect.void),
      )

      const sigkill = setTimeout(() => {
        if (processes.has(proc.pid)) {
          try {
            proc.child.kill('SIGKILL')
          } catch {
            // Process already dead — nothing to escalate to.
          }
        }
      }, 2000)

      timersByPid.set(proc.pid, {
        detach: timersByPid.get(proc.pid)?.detach,
        sigkill,
      })
    })
  }

  // ── Public Service Object ──────────────────────────────────────

  return {
    bindBus: (b) =>
      Effect.sync(() => {
        if (bus === null) bus = b
      }),

    executeDetached: (params) =>
      Effect.gen(function* () {
        const processesDir = yield* Effect.sync(() =>
          ensureProcessesDir(params.scratchpadPath),
        )
        const stdoutPath = path.join(processesDir, `${params.toolCallId}.stdout`)
        const stderrPath = path.join(processesDir, `${params.toolCallId}.stderr`)
        const stdoutPathDisplay = `$M/processes/${params.toolCallId}.stdout`
        const stderrPathDisplay = `$M/processes/${params.toolCallId}.stderr`

        const resultDeferred = yield* Deferred.make<ExecuteDetachedOutput, never>()
        const closeableScope = yield* Scope.make()

        const outputQueue = yield* Queue.unbounded<ShellOutputChunk>().pipe(
          Effect.provideService(Scope.Scope, closeableScope),
        )

        // ── Spawn the shell ────────────────────────────────────────
        // Bun.spawn throws synchronously if the shell is missing. The
        // outer Effect.catchAllCause below catches it and surfaces the
        // error to the LLM as a Completed result with exitCode 1.
        const shellPath = process.env.SHELL ?? '/bin/sh'
        const child = spawnDetachedShell(
          shellPath,
          params.command,
          params.cwd,
          params.agentEnv,
        )
        const pid = child.pid

        // ── File sinks + stream consumers ────────────────────────
        // Two concurrent consumers per stream: one writes raw bytes to
        // the output file (preserved for the LLM after exit), the other
        // offers decoded text to the live queue (Stream.fromQueue for the
        // tool's streaming output). The consumers end naturally when the
        // child exits and Bun closes the pipes.
        const stdoutWriter = Bun.file(stdoutPath).writer()
        const stderrWriter = Bun.file(stderrPath).writer()
        void consumeStreamToFileAndQueue(child.stdout, stdoutWriter, outputQueue, false)
        void consumeStreamToFileAndQueue(child.stderr, stderrWriter, outputQueue, true)

        const startedAt = Date.now()

        // ── Early registration ────────────────────────────────────
        // Ensures killAll(forkId) can find and kill the process even
        // during the detach wait window — before the exit handler fiber
        // is forked.
        let set = byFork.get(params.forkId)
        if (!set) {
          set = new Set()
          byFork.set(params.forkId, set)
        }
        set.add(pid)
        processes.set(pid, {
          pid,
          child,
          command: params.command,
          forkId: params.forkId,
          ownerAgentId: params.ownerAgentId,
          turnId: params.turnId,
          toolCallId: params.toolCallId,
          stdoutPath,
          stderrPath,
          startedAt,
          stdoutWriter,
          stderrWriter,
        })
        yield* publishActiveCount

        // ── Scope finalizer for child lifecycle ────────────────
        // Runs when closeableScope is closed. If Deferred resolved
        // with Detached → no-op. If Completed → idempotent cleanup.
        // If never resolved (race interrupted) → full teardown.
        yield* Scope.addFinalizer(
          closeableScope,
          Effect.gen(function* () {
            const polled = yield* Deferred.poll(resultDeferred)
            if (Option.isSome(polled)) {
              const result = yield* polled.value
              if (result._tag === 'Detached') return
            } else {
              processes.delete(pid)
              byFork.get(params.forkId)?.delete(pid)
              yield* publishActiveCount
            }

            // Kill the child if still running
            yield* Effect.sync(() => child.kill('SIGTERM')).pipe(
              Effect.catchAllCause(() => Effect.void),
            )

            // Flush + close file sinks
            yield* Effect.all(
              [
                closeFileSink(stdoutWriter),
                closeFileSink(stderrWriter),
              ],
              { concurrency: 2, discard: true },
            )

            // Delete temp files
            yield* Effect.sync(() => rmSync(stdoutPath)).pipe(
              Effect.catchAllCause(() => Effect.void),
            )
            yield* Effect.sync(() => rmSync(stderrPath)).pipe(
              Effect.catchAllCause(() => Effect.void),
            )
          }).pipe(Effect.uninterruptible),
        )

        // ── Race: completed vs detached ────────────────────────
        const completedPath = Effect.gen(function* () {
          const [code, signal] = yield* waitForProcExit(child)
          const exitCode = code ?? (signal ? 128 + signalToNum(signal) : 1)

          // Remove from tracking immediately — prevents the exit fiber
          // from calling finalize on a process we're handling here.
          processes.delete(pid)
          byFork.get(params.forkId)?.delete(pid)
          yield* publishActiveCount

          const [stdout, stderr] = yield* Effect.all(
            [
              readAndDeleteAfterClose(stdoutWriter, stdoutPath),
              readAndDeleteAfterClose(stderrWriter, stderrPath),
            ],
            { concurrency: 2 },
          )

          const result: ExecuteDetachedOutput = {
            _tag: 'Completed',
            stdout,
            stderr,
            exitCode,
          }
          yield* Deferred.succeed(resultDeferred, result)
          return result
        })

        const detachedPath = Effect.gen(function* () {
          // Wait for detach trigger. Bun.spawn returns synchronously with
          // a real pid — no 'spawn' event to wait for. If the shell was
          // missing, Bun.spawn threw before we got here.
          if (params.detachAfter > 0) {
            yield* Effect.sleep(`${params.detachAfter} seconds`)
          }

          // Fork the exit handler fiber as a daemon — it must outlive the
          // executeDetached call. This is the sole exit listener for the
          // detached process.
          yield* Effect.forkDaemon(
            Effect.gen(function* () {
              const [code, signal] = yield* waitForProcExit(child)
              const exitCode = code ?? (signal ? 128 + signalToNum(signal) : 1)
              const tracked = processes.get(pid)
              if (!tracked) return
              yield* finalize(tracked, exitCode)
            }),
          )

          // Publish registration event (model-facing $M/ paths)
          publish({
            type: 'shell_process_registered',
            forkId: params.forkId,
            pid,
            command: params.command,
            ownerAgentId: params.ownerAgentId,
            startedAt,
            stdoutPath: stdoutPathDisplay,
            stderrPath: stderrPathDisplay,
          })

          const result: ExecuteDetachedOutput = {
            _tag: 'Detached',
            pid,
            stdoutPath: stdoutPathDisplay,
            stderrPath: stderrPathDisplay,
          }
          yield* Deferred.succeed(resultDeferred, result)
          return result
        })

        // Fork the race as a daemon fiber. When the race resolves, it
        // shuts down the queue (terminating the stream) and closes the
        // scope (triggering cleanup).
        yield* Effect.forkDaemon(
          Effect.gen(function* () {
            const winner = yield* Effect.race(completedPath, detachedPath)

            // Shut down the queue — terminates Stream.fromQueue after
            // draining all remaining items.
            yield* Queue.shutdown(outputQueue)

            // Close the scope — triggers finalizer.
            yield* Scope.close(closeableScope, Exit.void)
            return winner
          }),
        )

        return {
          result: resultDeferred,
          outputStream: Stream.fromQueue(outputQueue),
        }
      }).pipe(
        // Catch spawn errors (shell not found, invalid env, etc.) —
        // reachable because Bun.spawn throws synchronously on missing
        // command, unlike child_process.spawn which raises an unhandled
        // EventEmitter 'error' event that escapes all JS-level catches.
        Effect.catchAllCause((cause) =>
          Effect.gen(function* () {
            const resultDeferred = yield* Deferred.make<ExecuteDetachedOutput, never>()
            yield* Deferred.succeed(resultDeferred, {
              _tag: 'Completed' as const,
              stdout: '',
              stderr: Cause.pretty(cause),
              exitCode: 1,
            })
            return {
              result: resultDeferred,
              outputStream: Stream.empty,
            }
          }),
        ),
      ),

    kill: (pid, forkId) =>
      Effect.gen(function* () {
        const proc = processes.get(pid)
        if (!proc || proc.forkId !== forkId) return false
        yield* performKill(proc)
        return true
      }),

    killAll: (forkId) =>
      Effect.gen(function* () {
        const set = byFork.get(forkId)
        if (!set) return
        for (const pid of [...set]) {
          const proc = processes.get(pid)
          if (proc) yield* performKill(proc)
        }
      }),

    interruptAll: (forkId) =>
      Effect.gen(function* () {
        const set = byFork.get(forkId)
        if (!set) return
        for (const pid of [...set]) {
          const proc = processes.get(pid)
          if (proc) yield* performKill(proc)
        }
      }),
    activeCount: SubscriptionRef.get(activeCount),
    changes: activeCount.changes,
  }
})
