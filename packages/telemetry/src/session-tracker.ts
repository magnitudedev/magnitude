/**
 * Session Tracker — accumulates usage counters during a session
 * for the session_end telemetry event.
 */

export class SessionTracker {
  private startTime = Date.now()
  private turnCount = 0
  private toolCount = 0
  private userMessageCount = 0
  private totalInputTokens = 0
  private totalOutputTokens = 0
  private totalLinesWritten = 0
  private totalLinesAdded = 0
  private totalLinesRemoved = 0
  private agentCount = 0

  recordUserMessage(): void {
    this.userMessageCount++
  }

  recordTurn(inputTokens: number | null, outputTokens: number | null, toolCallCount: number): void {
    this.turnCount++
    this.toolCount += toolCallCount
    if (inputTokens !== null) this.totalInputTokens += inputTokens
    if (outputTokens !== null) this.totalOutputTokens += outputTokens
  }

  recordLinesWritten(n: number): void {
    this.totalLinesWritten += n
  }

  recordLinesAdded(n: number): void {
    this.totalLinesAdded += n
  }

  recordLinesRemoved(n: number): void {
    this.totalLinesRemoved += n
  }

  recordAgentSpawned(): void {
    this.agentCount++
  }

  getSummary() {
    return {
      durationSeconds: Math.round((Date.now() - this.startTime) / 1000),
      totalTurns: this.turnCount,
      totalTools: this.toolCount,
      totalUserMessages: this.userMessageCount,
      totalInputTokens: this.totalInputTokens,
      totalOutputTokens: this.totalOutputTokens,
      totalLinesWritten: this.totalLinesWritten,
      totalLinesAdded: this.totalLinesAdded,
      totalLinesRemoved: this.totalLinesRemoved,
      agentCount: this.agentCount,
    }
  }
}
