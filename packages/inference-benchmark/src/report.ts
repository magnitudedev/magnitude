import * as FileSystem from "@effect/platform/FileSystem"
import { Data, Effect, Schema } from "effect"
import type { BenchmarkResult, ComparisonResult, MetricSummary, TrialAnalysis } from "./domain"

export class ReportError extends Data.TaggedError("ReportError")<{
  readonly output: string
  readonly message: string
}> {}

function number(value: number | undefined, digits = 2): string {
  return value === undefined || !Number.isFinite(value) ? "unsupported" : value.toFixed(digits)
}

function bytes(value: number | undefined): string {
  if (value === undefined) return "unsupported"
  const units = ["B", "KiB", "MiB", "GiB", "TiB"]
  let amount = value
  let unit = 0
  while (amount >= 1024 && unit < units.length - 1) {
    amount /= 1024
    unit++
  }
  return `${amount.toFixed(2)} ${units[unit]}`
}

function median(summary: MetricSummary | undefined): string {
  return summary ? number(summary.median) : "unsupported"
}

function preferredPeakMemory(analysis: TrialAnalysis): number | undefined {
  return analysis.memory?.peakDeviceBytes
    ? Object.values(analysis.memory.peakDeviceBytes).reduce((sum, value) => sum + value, 0)
    : analysis.memory?.peakBytes
}

function analysisRow(analysis: TrialAnalysis): string {
  const nonValid = Object.entries(analysis.outcomes).filter(([outcome]) => outcome !== "valid").reduce((sum, [, count]) => sum + count, 0)
  return `| ${analysis.trialId} | ${analysis.pattern} | ${analysis.measuredRequests} | ${analysis.validRequests} | ${nonValid} | ${median(analysis.promptTokens)} | ${median(analysis.completionTokens)} | ${median(analysis.responsivenessMs)} | ${number(analysis.responsivenessMs?.p95)} | ${median(analysis.prefillTokensPerSecond)} | ${median(analysis.decodeTokensPerSecond)} | ${number(analysis.achievedCompletionTokensPerSecond)} | ${number(analysis.cacheReuseRatio?.median, 3)} | ${number(analysis.completionMs?.p95)} | ${bytes(preferredPeakMemory(analysis))} |`
}

export function renderBenchmarkMarkdown(result: BenchmarkResult): string {
  return [
    `# Inference benchmark: ${result.target.id}`,
    "",
    `Plan: \`${result.planDigest}\``,
    ...(result.target.speculativeBackend !== undefined ? [`Speculative method: **${result.target.speculativeBackend}**`] : []),
    "",
    "| Trial | Pattern | Measured | Semantically valid | Non-valid | Actual input p50 (tokens) | Actual output p50 (tokens) | TTFT p50 (ms) | TTFT p95 (ms) | Native prefill p50 (tok/s) | Native decode p50 (tok/s) | Achieved output (tok/s) | Cache reuse | Completion p95 (ms) | Peak attributed footprint |",
    "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ...result.analyses.map(analysisRow),
    "",
    ...(result.warnings.length > 0 ? ["## Warnings", "", ...result.warnings.map((warning) => `- ${warning}`), ""] : []),
  ].join("\n")
}

export function renderComparisonMarkdown(comparison: ComparisonResult): string {
  const targetNames = comparison.results.map((result) => result.target.id).join(" vs ")
  const [candidate, reference] = comparison.results
  const matched = candidate && reference
    ? candidate.analyses.flatMap((left) => {
        const right = reference.analyses.find((analysis) => analysis.trialId === left.trialId)
        if (!right) return []
        const values: readonly [string, number | undefined, number | undefined][] = [
          ["TTFT median (ms)", left.responsivenessMs?.median, right.responsivenessMs?.median],
          ["Native prefill median (tok/s)", left.prefillTokensPerSecond?.median, right.prefillTokensPerSecond?.median],
          ["Decode median (tok/s)", left.decodeTokensPerSecond?.median, right.decodeTokensPerSecond?.median],
          ["Achieved completion (tok/s)", left.achievedCompletionTokensPerSecond, right.achievedCompletionTokensPerSecond],
          [
            "Peak attributable footprint (bytes)",
            left.memory?.source === right.memory?.source ? preferredPeakMemory(left) : undefined,
            left.memory?.source === right.memory?.source ? preferredPeakMemory(right) : undefined,
          ],
        ]
        return values.flatMap(([metric, candidateValue, referenceValue]) =>
          candidateValue !== undefined && referenceValue !== undefined && referenceValue !== 0
            ? [`| ${left.trialId} | ${metric} | ${number(candidateValue)} | ${number(referenceValue)} | ${number(candidateValue / referenceValue, 3)} |`]
            : [])
      })
    : []
  const sections = comparison.results.flatMap((result) => [renderBenchmarkMarkdown(result), ""])
  return [
    `# Inference comparison: ${targetNames}`,
    "",
    `Shared plan: \`${comparison.plan.digest}\``,
    `Model: \`${comparison.plan.model.id}\``,
    `Profile: \`${comparison.plan.profile}\``,
    `Comparison kind: **${comparison.comparisonKind}**`,
    `Comparison protocol: **${comparison.comparisonProtocol}**`,
    ...(comparison.comparisonProtocol === "speculative-decoding"
      ? ["Methods: " + comparison.results.map(({ target }) => `\`${target.id}\`=${target.speculativeBackend ?? "unreported"}`).join(", ")]
      : []),
    ...(comparison.differences.length > 0
      ? ["", "Differences:", "", ...comparison.differences.map((difference) => `- ${difference}`)]
      : []),
    "",
    ...(candidate && reference ? [
      "## Matched comparisons",
      "",
      `Ratios are \`${candidate.target.id} / ${reference.target.id}\` at identical trial points.`,
      "",
      "| Trial | Metric | Candidate | Reference | Ratio |",
      "| --- | --- | ---: | ---: | ---: |",
      ...matched,
      "",
    ] : []),
    ...sections,
  ].join("\n")
}

export const writeReport = (
  output: string,
  value: BenchmarkResult | ComparisonResult,
): Effect.Effect<void, ReportError, FileSystem.FileSystem> =>
  Effect.gen(function* () {
    const fs = yield* FileSystem.FileSystem
    const json = yield* Schema.encode(Schema.parseJson(Schema.Unknown, { space: 2 }))(value).pipe(
      Effect.map((encoded) => `${encoded}\n`),
    )
    yield* fs.writeFileString(output, json)
    const markdown = output.endsWith(".json") ? `${output.slice(0, -5)}.md` : `${output}.md`
    yield* fs.writeFileString(
      markdown,
      "results" in value ? renderComparisonMarkdown(value) : renderBenchmarkMarkdown(value),
    )
  }).pipe(Effect.mapError((error) => new ReportError({
    output,
    message: error instanceof Error ? error.message : String(error),
  })))
