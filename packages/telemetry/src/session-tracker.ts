/**
 * Session Tracker — accumulates usage counters during a session
 * for the session_end telemetry event.
 */

export interface ModelUsage {
  providerId: string
  modelId: string
  inputTokens: number
  outputTokens: number
}

export class SessionTracker {
  private startTime = Date.now()
  private turnCount = 0
  private userMessageCount = 0
  private totalInputTokens = 0
  private totalOutputTokens = 0
  private compactionCount = 0
  private modelsUsedMap = new Map<string, { providerId: string; modelId: string; inputTokens: number; outputTokens: number }>()

  recordUserMessage(): void {
    this.userMessageCount++
  }

  recordTurn(providerId: string | null, modelId: string | null, inputTokens: number | null, outputTokens: number | null): void {
    this.turnCount++
    if (inputTokens !== null) this.totalInputTokens += inputTokens
    if (outputTokens !== null) this.totalOutputTokens += outputTokens
    if (providerId && modelId) {
      const key = `${providerId}:${modelId}`
      const existing = this.modelsUsedMap.get(key)
      if (existing) {
        if (inputTokens !== null) existing.inputTokens += inputTokens
        if (outputTokens !== null) existing.outputTokens += outputTokens
      } else {
        this.modelsUsedMap.set(key, {
          providerId,
          modelId,
          inputTokens: inputTokens ?? 0,
          outputTokens: outputTokens ?? 0,
        })
      }
    }
  }

  recordCompaction(): void {
    this.compactionCount++
  }

  getSummary() {
    return {
      durationSeconds: Math.round((Date.now() - this.startTime) / 1000),
      totalTurns: this.turnCount,
      totalUserMessages: this.userMessageCount,
      totalInputTokens: this.totalInputTokens,
      totalOutputTokens: this.totalOutputTokens,
      modelsUsed: Array.from(this.modelsUsedMap.values()),
      compactionCount: this.compactionCount,
    }
  }
}
