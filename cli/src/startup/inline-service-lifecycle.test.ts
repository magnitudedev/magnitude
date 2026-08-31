import { describe, expect, it } from "vitest"
import * as Terminal from "@effect/platform/Terminal"
import { Effect, Schema } from "effect"
import { AcnLifecycleStateSchema } from "@magnitudedev/sdk"
import {
  inlineServiceCompletionColor,
  makeInlineServiceStartupPresenter,
  serviceStartupChildPhase,
} from "./inline-service-lifecycle"
import { defaultCliThemes } from "../utils/theme"

const state = (input: unknown) => Schema.decodeUnknownSync(AcnLifecycleStateSchema)(input)

describe("inline service lifecycle copy", () => {
  it("uses the Markdown sea-foam token for completed checks", () => {
    expect(inlineServiceCompletionColor(defaultCliThemes.dark)).toBe(
      defaultCliThemes.dark.markdown.inlineCode,
    )
    expect(inlineServiceCompletionColor(defaultCliThemes.dark)).not.toBe(
      defaultCliThemes.dark.status.success,
    )
  })

  it("absorbs client preparation and inference resolution into the parent", () => {
    const preparing = state({ _tag: "Starting", phase: "PreparingAcn" })
    const resolving = state({ _tag: "Starting", phase: "ResolvingLocalInference" })
    expect(serviceStartupChildPhase(preparing)).toBeNull()
    expect(serviceStartupChildPhase(resolving)).toBeNull()
  })

  it("names backend hardware in active and completed phases", () => {
    const preparing = state({
      _tag: "Starting",
      phase: {
        _tag: "PreparingBackend",
        backend: { _tag: "Metal", hardwareLabel: "Apple M3 Max" },
      },
    })
    expect(serviceStartupChildPhase(preparing)).toEqual({
      key: "backend",
      active: "Preparing Metal backend for Apple M3 Max",
      completed: "Metal backend ready for Apple M3 Max",
      progress: null,
    })
  })

  it("keeps download measurement on one line and completes total over total", () => {
    const downloading = state({
      _tag: "Installing",
      phase: "DownloadingInferenceEngine",
      overallProgress: 0.63,
      detailIsExact: true,
      detail: {
        completed: 3_100_000_000,
        totalBytes: 4_900_000_000,
        unit: "Bytes",
      },
    })
    expect(serviceStartupChildPhase(downloading)).toEqual({
      key: "inference-download",
      active: "Downloading inference engine... 63% (3.1 GB / 4.9 GB)",
      completed: "Inference engine downloaded 100% (4.9 GB / 4.9 GB)",
      progress: 0.63,
    })
  })

  it("does not show synthetic progress while starting the service", () => {
    const starting = state({
      _tag: "Installing",
      phase: "StartingMagnitude",
      overallProgress: 0.97,
      detailIsExact: false,
    })
    expect(serviceStartupChildPhase(starting)).toEqual({
      key: "local-inference",
      active: "Starting inference engine",
      completed: "Inference engine started",
      progress: null,
    })
  })

  it("uses the same inference-engine child for the non-installing launch path", () => {
    const launching = state({ _tag: "Starting", phase: "LaunchingLocalInference" })
    expect(serviceStartupChildPhase(launching)).toEqual({
      key: "local-inference",
      active: "Starting inference engine",
      completed: "Inference engine started",
      progress: null,
    })
  })

  it("nests service acquisition beneath the parent startup operation", async () => {
    const output: Array<string> = []
    const terminal = Terminal.Terminal.of({
      columns: Effect.succeed(80),
      rows: Effect.succeed(24),
      isTTY: Effect.succeed(false),
      readInput: Effect.die("unused"),
      readLine: Effect.die("unused"),
      display: (text) => Effect.sync(() => { output.push(text) }),
    })

    await Effect.runPromise(Effect.gen(function* () {
      const presenter = yield* makeInlineServiceStartupPresenter(defaultCliThemes.dark)
      yield* presenter.acquisitionObserver.report({
        _tag: "Planned",
        plan: {
          daemonBytes: 39_400_000,
          inferenceEngineBytes: 0,
          inferenceEngineBytesExact: true,
        },
      })
      yield* presenter.acquisitionObserver.report({
        _tag: "Artifact",
        event: {
          _tag: "Downloading",
          progress: {
            strategy: "Sequential",
            acceptedBytes: 24_822_000,
            totalBytes: 39_400_000,
            attempt: 1,
          },
        },
      })
      yield* presenter.acquisitionSucceeded
    }).pipe(Effect.provideService(Terminal.Terminal, terminal)))

    expect(output.join("")).toBe([
      "○ Starting Magnitude service",
      "  Downloading Magnitude service... 63% (24.8 MB / 39.4 MB)",
      "  ✓ Magnitude service downloaded 100% (39.4 MB / 39.4 MB)",
      "",
    ].join("\n"))
  })

  it("leaves acquisition progress durable for an outer startup failure", async () => {
    const output: Array<string> = []
    const terminal = Terminal.Terminal.of({
      columns: Effect.succeed(80),
      rows: Effect.succeed(24),
      isTTY: Effect.succeed(true),
      readInput: Effect.die("unused"),
      readLine: Effect.die("unused"),
      display: (text) => Effect.sync(() => { output.push(text) }),
    })

    await Effect.runPromise(Effect.gen(function* () {
      const presenter = yield* makeInlineServiceStartupPresenter(defaultCliThemes.dark)
      yield* presenter.acquisitionObserver.report({
        _tag: "Planned",
        plan: {
          daemonBytes: 39_400_000,
          inferenceEngineBytes: 0,
          inferenceEngineBytesExact: true,
        },
      })
      yield* presenter.acquisitionObserver.report({
        _tag: "Artifact",
        event: {
          _tag: "Downloading",
          progress: {
            strategy: "Sequential",
            acceptedBytes: 24_822_000,
            totalBytes: 39_400_000,
            attempt: 1,
          },
        },
      })
    }).pipe(Effect.provideService(Terminal.Terminal, terminal)))

    expect(output).toHaveLength(1)
    expect(output[0]).toContain("Starting Magnitude service")
  })
})
