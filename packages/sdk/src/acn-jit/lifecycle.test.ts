import { Effect, Option, TestClock, TestContext } from "effect";
import { describe, expect, it } from "vitest";
import { AcnCandidateExitedAfterAdmission } from "./errors";
import { formatAcnEnsuranceError, makeAcnLifecycle } from "./lifecycle";

const plan = {
  daemonBytes: 100,
  inferenceEngineBytes: 300,
  inferenceEngineBytesExact: true,
} as const;

const run = <A, E>(effect: Effect.Effect<A, E>) =>
  Effect.runPromise(effect.pipe(Effect.provide(TestContext.TestContext)));

describe("ACN installation lifecycle", () => {
  it("formats candidate exits as diagnostics without duplicating the CLI heading", () => {
    expect(formatAcnEnsuranceError(new AcnCandidateExitedAfterAdmission({
      pid: 42,
      code: 1,
      stderr: "  local inference failed  ",
    }))).toBe("local inference failed");
    expect(formatAcnEnsuranceError(new AcnCandidateExitedAfterAdmission({
      pid: 42,
      code: 1,
      stderr: "",
    }))).toBe("The service process exited before becoming ready");
  });

  it("weights download progress by bundle size without regressing on retry", async () => {
    await run(
      Effect.gen(function* () {
        const lifecycle = yield* makeAcnLifecycle();
        yield* lifecycle.report({
          _tag: "Installing",
          phase: "DownloadingDaemon",
          plan,
          progress: Option.some({
            completed: 50,
            totalBytes: 100,
            unit: "Bytes",
            attempt: Option.some(1),
          }),
        });
        const halfway = yield* lifecycle.get;
        expect(halfway._tag).toBe("Installing");
        if (halfway._tag !== "Installing") return;
        expect(halfway.overallProgress).toBeCloseTo(0.1125);

        yield* lifecycle.report({
          _tag: "Installing",
          phase: "DownloadingDaemon",
          plan,
          progress: Option.some({
            completed: 25,
            totalBytes: 100,
            unit: "Bytes",
            attempt: Option.some(2),
          }),
        });
        const retried = yield* lifecycle.get;
        expect(retried._tag).toBe("Installing");
        if (retried._tag !== "Installing") return;
        expect(retried.overallProgress).toBe(halfway.overallProgress);

        yield* lifecycle.report({
          _tag: "Installing",
          phase: "DownloadingInferenceEngine",
          plan,
          progress: Option.some({
            completed: 150,
            totalBytes: 300,
            unit: "Bytes",
            attempt: Option.some(1),
          }),
        });
        const inferenceHalfway = yield* lifecycle.get;
        expect(inferenceHalfway._tag).toBe("Installing");
        if (inferenceHalfway._tag !== "Installing") return;
        expect(inferenceHalfway.overallProgress).toBeCloseTo(0.5625);
      })
    );
  });

  it("gives Starting Magnitude the final ten percent asymptotically", async () => {
    await run(
      Effect.gen(function* () {
        const lifecycle = yield* makeAcnLifecycle();
        yield* lifecycle.report({
          _tag: "Installing",
          phase: "DownloadingInferenceEngine",
          plan,
          progress: Option.some({
            completed: 300,
            totalBytes: 300,
            unit: "Bytes",
            attempt: Option.some(1),
          }),
        });
        yield* lifecycle.report({
          _tag: "Installing",
          phase: "StartingMagnitude",
          plan,
          progress: Option.none(),
        });

        const started = yield* lifecycle.get;
        expect(started._tag).toBe("Installing");
        if (started._tag !== "Installing") return;
        expect(started.overallProgress).toBeCloseTo(0.9);

        yield* TestClock.adjust("7500 millis");
        const expectedDuration = yield* lifecycle.get;
        expect(expectedDuration._tag).toBe("Installing");
        if (expectedDuration._tag !== "Installing") return;
        expect(expectedDuration.overallProgress).toBeCloseTo(0.99);

        yield* TestClock.adjust("1 hour");
        const approached = yield* lifecycle.get;
        expect(approached._tag).toBe("Installing");
        if (approached._tag !== "Installing") return;
        expect(approached.overallProgress).toBeLessThan(1);

        yield* lifecycle.ready;
        const ready = yield* lifecycle.get;
        expect(ready._tag).toBe("Ready");
      })
    );
  });

  it("keeps the installation screen active while services start", async () => {
    await run(
      Effect.gen(function* () {
        const lifecycle = yield* makeAcnLifecycle();
        yield* lifecycle.report({
          _tag: "Installing",
          phase: "DownloadingDaemon",
          plan,
          progress: Option.some({
            completed: 100,
            totalBytes: 100,
            unit: "Bytes",
            attempt: Option.none(),
          }),
        });
        yield* lifecycle.report({
          _tag: "Starting",
          phase: "PreparingAcn",
        });
        const launching = yield* lifecycle.get;
        expect(launching._tag).toBe("Installing");
        if (launching._tag !== "Installing") return;
        expect(launching.phase).toBe("DownloadingDaemon");
      })
    );
  });

  it("replaces installation progress with exact backend preparation", async () => {
    await run(
      Effect.gen(function* () {
        const lifecycle = yield* makeAcnLifecycle();
        yield* lifecycle.report({
          _tag: "Installing",
          phase: "StartingMagnitude",
          plan,
          progress: Option.none(),
        });
        yield* lifecycle.report({
          _tag: "Starting",
          phase: {
            _tag: "PreparingBackend",
            backend: { _tag: "Cuda", hardwareLabel: "NVIDIA GPU" },
          },
        });

        expect(yield* lifecycle.get).toEqual({
          _tag: "Starting",
          phase: {
            _tag: "PreparingBackend",
            backend: { _tag: "Cuda", hardwareLabel: "NVIDIA GPU" },
          },
        });
      })
    );
  });
});
