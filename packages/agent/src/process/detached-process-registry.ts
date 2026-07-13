/**
 * DetachedShellRegistry — Effect service for detached shell process lifecycle.
 *
 * Created by ExecutionManager, provided via fork layers.
 * The shell tool calls `executeDetached()` — one method that owns
 * spawn through cleanup.
 *
 * Design:
 * - Exit handler is the single authority for finalize. Kill timers
 *   only escalate signals (SIGTERM → SIGKILL). No force-finalize timer.
 * - SIGKILL always produces an exit event in Node.js (libuv reaps),
 *   so no zombie scenario exists that requires a force timer.
 * - Timer tracking via Map<number, ProcessTimers> — no property injection.
 */

import { Context, Effect, Layer, Deferred, Stream } from 'effect'
import { type WorkerBusService } from '@magnitudedev/event-core'
import type { AppEvent } from '../events'
import { logger } from '@magnitudedev/logger'

// ── Public Types ──────────────────────────────────────────────────────

export interface ExecuteDetachedParams {
  readonly command: string
  readonly forkId: string | null
  readonly turnId: string
  readonly toolCallId: string
  readonly scratchpadPath: string
  readonly cwd: string
  readonly agentEnv: Record<string, string | undefined>
  readonly detachAfter: number // seconds
  readonly ownerAgentId: string | undefined // undefined = root agent
}

export type ExecuteDetachedOutput =
  | { readonly _tag: 'Completed'; readonly stdout: string; readonly stderr: string; readonly exitCode: number }
  | { readonly _tag: 'Detached'; readonly pid: number; readonly stdoutPath: string; readonly stderrPath: string }

export interface ShellOutputChunk {
  readonly stdout: string
  readonly stderr: string
}

export interface ExecuteDetachedHandle {
  readonly result: Deferred.Deferred<ExecuteDetachedOutput, never>
  readonly outputStream: Stream.Stream<ShellOutputChunk, never, never>
}

export interface DetachedShellRegistryService {
  readonly executeDetached: (params: ExecuteDetachedParams) => Effect.Effect<ExecuteDetachedHandle, never, never>
  readonly kill: (pid: number, forkId: string | null) => Effect.Effect<boolean>
  readonly killAll: (forkId: string | null) => Effect.Effect<void>
  readonly interruptAll: (forkId: string | null) => Effect.Effect<void>
  readonly bindBus: (bus: WorkerBusService<AppEvent>) => Effect.Effect<void>
}

export class DetachedShellRegistry extends Context.Tag('DetachedShellRegistry')<
  DetachedShellRegistry,
  DetachedShellRegistryService
>() {}
