import type { ExtensionUIContext } from "@earendil-works/pi-coding-agent"
import { Text } from "@earendil-works/pi-tui"
import type { MagnitudeObservation, MagnitudeProgress, MagnitudeTimings } from "./protocol"

export const MAGNITUDE_SUMMARY_WIDGET_KEY = "magnitude-inference-summary"

const elapsedSeconds = (milliseconds: number): string => `${(Math.max(0, milliseconds) / 1_000).toFixed(1)}s`

const compactNumber = (value: number): string => {
  if (value < 1_000) return String(Math.round(value))
  if (value < 1_000_000) return `${(value / 1_000).toFixed(1).replace(/\.0$/, "")}k`
  return `${(value / 1_000_000).toFixed(1).replace(/\.0$/, "")}m`
}

const rate = (value: number): string => Number.isFinite(value) ? value.toFixed(1) : "0.0"

const workDuration = (milliseconds: number): string => {
  if (milliseconds < 1_000) return "<1 second"
  const totalSeconds = Math.floor(milliseconds / 1_000)
  if (totalSeconds < 60) return `${totalSeconds} second${totalSeconds === 1 ? "" : "s"}`
  const minutes = Math.floor(totalSeconds / 60)
  const seconds = totalSeconds % 60
  return seconds === 0
    ? `${minutes} minute${minutes === 1 ? "" : "s"}`
    : `${minutes}:${String(seconds).padStart(2, "0")}`
}

type ActiveStatus =
  | {
      readonly phase: "model_loading"
      readonly startedAt: number
      readonly modelName: string
      readonly fraction: number
    }
  | {
      readonly phase: "prefill"
      readonly startedAt: number
      readonly completedTokens: number
      readonly totalTokens: number
      readonly cachedTokens: number
    }
  | { readonly phase: "generating"; readonly startedAt: number }

interface AgentRun {
  readonly startedAt: number
  modelName?: string
  firstTimeToFirstTokenMs?: number
  generatedTokens: number
  decodeMs: number
  settledAt?: number
  cancelled: boolean
}

export interface ProgressTrackerOptions {
  readonly now?: () => number
  readonly setInterval?: (callback: () => void, milliseconds: number) => ReturnType<typeof globalThis.setInterval>
  readonly clearInterval?: (timer: ReturnType<typeof globalThis.setInterval>) => void
  readonly tickMilliseconds?: number
}

export interface ProgressRequest {
  readonly observe: (observation: MagnitudeObservation) => void
  readonly finish: () => void
  readonly fail: () => void
}

export class MagnitudeProgressTracker {
  readonly #ui: Pick<ExtensionUIContext, "setWidget" | "setWorkingMessage">
  readonly #now: () => number
  readonly #schedule: (callback: () => void, milliseconds: number) => ReturnType<typeof globalThis.setInterval>
  readonly #clearInterval: (timer: ReturnType<typeof globalThis.setInterval>) => void
  readonly #tickMilliseconds: number
  #timer: ReturnType<typeof globalThis.setInterval> | undefined
  #active: ActiveStatus | undefined
  #run: AgentRun | undefined
  #requestInFlight = false
  #requestId = 0
  #disposed = false

  constructor(
    ui: Pick<ExtensionUIContext, "setWidget" | "setWorkingMessage">,
    options: ProgressTrackerOptions = {},
  ) {
    this.#ui = ui
    this.#now = options.now ?? Date.now
    this.#schedule = options.setInterval ?? globalThis.setInterval
    this.#clearInterval = options.clearInterval ?? globalThis.clearInterval
    this.#tickMilliseconds = options.tickMilliseconds ?? 100
  }

  startRun(modelName?: string): void {
    if (this.#disposed || this.#run !== undefined) return
    this.#run = {
      startedAt: this.#now(),
      ...(modelName === undefined ? {} : { modelName }),
      generatedTokens: 0,
      decodeMs: 0,
      cancelled: false,
    }
    this.#clearSummary()
  }

  settleRun(): void {
    if (this.#disposed || this.#run === undefined) return
    this.#run.settledAt = this.#now()
    if (!this.#requestInFlight) this.#renderSummary()
  }

  begin(modelName: string): ProgressRequest {
    this.startRun(modelName)
    if (this.#run !== undefined && this.#run.modelName === undefined) this.#run.modelName = modelName
    const requestId = ++this.#requestId
    this.#requestInFlight = true
    this.#active = undefined
    this.#stopTimer()
    this.#clearSummary()
    this.#ui.setWorkingMessage()
    return {
      observe: (observation) => {
        if (!this.#isCurrent(requestId)) return
        if (observation.progress !== undefined) this.#observeProgress(observation.progress, modelName)
        if (observation.timings !== undefined) this.#recordTimings(observation.timings)
      },
      finish: () => {
        if (!this.#isCurrent(requestId)) return
        this.#requestInFlight = false
        this.#active = undefined
        this.#stopTimer()
        this.#ui.setWorkingMessage()
        if (this.#run?.settledAt !== undefined) this.#renderSummary()
      },
      fail: () => {
        if (!this.#isCurrent(requestId)) return
        this.#requestInFlight = false
        if (this.#run !== undefined) this.#run.cancelled = true
        this.#active = undefined
        this.#stopTimer()
        this.#ui.setWorkingMessage()
        this.#clearSummary()
        if (this.#run?.settledAt !== undefined) this.#renderSummary()
      },
    }
  }

  clear(): void {
    this.#requestId++
    this.#requestInFlight = false
    this.#active = undefined
    this.#run = undefined
    this.#stopTimer()
    this.#ui.setWorkingMessage()
    this.#clearSummary()
  }

  dispose(): void {
    if (this.#disposed) return
    this.clear()
    this.#disposed = true
  }

  #isCurrent(requestId: number): boolean {
    return !this.#disposed && requestId === this.#requestId
  }

  #observeProgress(progress: MagnitudeProgress, modelName: string): void {
    const now = this.#now()
    switch (progress.phase) {
      case "model_loading":
        this.#active = {
          phase: "model_loading",
          modelName,
          fraction: Math.max(0, Math.min(1, progress.fraction)),
          startedAt: this.#active?.phase === "model_loading" ? this.#active.startedAt : now,
        }
        break
      case "queued":
      case "preparing":
        this.#active = undefined
        this.#stopTimer()
        this.#ui.setWorkingMessage()
        return
      case "prefill":
        this.#active = {
          phase: "prefill",
          completedTokens: progress.completed_tokens,
          totalTokens: progress.total_tokens,
          cachedTokens: progress.cached_tokens,
          startedAt: this.#active?.phase === "prefill" ? this.#active.startedAt : now,
        }
        break
      case "generating":
        this.#active = {
          phase: "generating",
          startedAt: this.#active?.phase === "generating" ? this.#active.startedAt : now,
        }
        break
    }
    this.#startTimer()
    this.#render()
  }

  #recordTimings(timings: MagnitudeTimings): void {
    this.#active = undefined
    this.#stopTimer()
    this.#ui.setWorkingMessage()
    if (this.#run === undefined) return
    this.#run.firstTimeToFirstTokenMs ??= timings.time_to_first_token_ms
    this.#run.generatedTokens += timings.predicted_n
    this.#run.decodeMs += timings.predicted_ms
    this.#run.cancelled = false
  }

  #render(): void {
    if (this.#disposed || this.#active === undefined) return
    const elapsed = elapsedSeconds(this.#now() - this.#active.startedAt)
    switch (this.#active.phase) {
      case "model_loading":
        this.#ui.setWorkingMessage(
          `Loading ${this.#active.modelName} into memory · ${Math.round(this.#active.fraction * 100)}% · ${elapsed}`,
        )
        break
      case "prefill": {
        const cached = Math.min(this.#active.cachedTokens, this.#active.totalTokens)
        const completed = Math.min(Math.max(this.#active.completedTokens, cached), this.#active.totalTokens)
        const effectiveCompleted = completed - cached
        const effectiveTotal = this.#active.totalTokens - cached
        this.#ui.setWorkingMessage(
          `Prefilling prompt · ${compactNumber(effectiveCompleted)} / ${compactNumber(effectiveTotal)} tokens · ${compactNumber(cached)} cached · ${elapsed}`,
        )
        break
      }
      case "generating":
        this.#ui.setWorkingMessage(`Working · ${elapsed}`)
        break
    }
  }

  #startTimer(): void {
    if (this.#timer === undefined) {
      this.#timer = this.#schedule(() => this.#render(), this.#tickMilliseconds)
    }
  }

  #stopTimer(): void {
    if (this.#timer === undefined) return
    this.#clearInterval(this.#timer)
    this.#timer = undefined
  }

  #clearSummary(): void {
    this.#ui.setWidget(MAGNITUDE_SUMMARY_WIDGET_KEY, undefined)
  }

  #renderSummary(): void {
    const run = this.#run
    if (run === undefined || run.settledAt === undefined) return
    this.#run = undefined
    if (run.cancelled || run.modelName === undefined || run.firstTimeToFirstTokenMs === undefined) {
      this.#clearSummary()
      return
    }
    const tokensPerSecond = run.generatedTokens > 0 && run.decodeMs > 0
      ? run.generatedTokens * 1_000 / run.decodeMs
      : undefined
    const summary = `● ${run.modelName} worked for ${workDuration(run.settledAt - run.startedAt)}`
      + ` · ${elapsedSeconds(run.firstTimeToFirstTokenMs)} TTFT`
      + (tokensPerSecond === undefined ? "" : ` · ${rate(tokensPerSecond)} tok/s`)
    this.#ui.setWidget(
      MAGNITUDE_SUMMARY_WIDGET_KEY,
      (_tui, theme) => new Text(theme.fg("muted", summary), 0, 0),
    )
  }
}
