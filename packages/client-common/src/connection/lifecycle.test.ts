import {
  ConnectionStateSchema,
  type ServiceStartProgress,
} from "@magnitudedev/sdk";
import { Option, Schema } from "effect";
import { describe, expect, it } from "vitest";
import {
  ServiceChecking,
  reduceLifecycle,
  renderLifecycle,
  lifecycleIsTimeDependent,
  type LifecycleModel,
} from "./lifecycle";

const plan = {
  daemonBytes: 100,
  inferenceEngineBytes: 300,
  inferenceEngineBytesExact: true,
};
const report = (
  state: LifecycleModel,
  activity: ServiceStartProgress,
  now = 0
) =>
  reduceLifecycle(
    state,
    { _tag: "Connecting", reason: "initial", activity: Option.some(activity) },
    now
  );
const progress = (completed: number, totalBytes: number) =>
  Option.some({
    completed,
    totalBytes,
    unit: "Bytes" as const,
    attempt: Option.none<number>(),
  });
const ready = Schema.decodeUnknownSync(ConnectionStateSchema)({
  _tag: "Ready",
  service: { id: "one", version: "0.0.1", rpcVersion: 1 },
});

describe("service installation presentation", () => {
  it("weights measured downloads and does not regress on a download retry", () => {
    const halfway = report(new ServiceChecking({}), {
      _tag: "Installing",
      phase: "DownloadingDaemon",
      plan,
      progress: progress(50, 100),
    });
    expect(renderLifecycle(halfway, 0)).toMatchObject({
      _tag: "Installing",
      overallProgress: 0.1125,
    });
    expect(lifecycleIsTimeDependent(halfway)).toBe(false);
    const retried = report(halfway, {
      _tag: "Installing",
      phase: "DownloadingDaemon",
      plan,
      progress: progress(25, 100),
    });
    expect(renderLifecycle(retried, 0)).toMatchObject({
      overallProgress: 0.1125,
    });
    const inference = report(retried, {
      _tag: "Installing",
      phase: "DownloadingInferenceEngine",
      plan,
      progress: progress(150, 300),
    });
    expect(renderLifecycle(inference, 0)).toMatchObject({
      overallProgress: 0.5625,
    });
  });

  it("estimates only the final ten percent and never reports completion before Ready", () => {
    const state = report(new ServiceChecking({}), {
      _tag: "Installing",
      phase: "StartingMagnitude",
      plan,
      progress: Option.none(),
    });
    expect(renderLifecycle(state, 0)).toMatchObject({ overallProgress: 0.9 });
    expect(lifecycleIsTimeDependent(state)).toBe(true);
    expect(renderLifecycle(state, 7500)).toMatchObject({
      overallProgress: 0.99,
    });
    const late = renderLifecycle(state, 3_600_000);
    expect(late._tag === "Installing" && late.overallProgress < 1).toBe(true);
    expect(reduceLifecycle(state, ready, 3_600_000)._tag).toBe("Ready");
    expect(
      lifecycleIsTimeDependent(reduceLifecycle(state, ready, 3_600_000))
    ).toBe(false);
  });

  it("keeps installation through generic launch observations, but shows exact backend preparation", () => {
    const installing = report(new ServiceChecking({}), {
      _tag: "Installing",
      phase: "StartingMagnitude",
      plan,
      progress: Option.none(),
    });
    expect(
      report(installing, { _tag: "Starting", phase: "PreparingAcn" })
    ).toBe(installing);
    const phase = {
      _tag: "PreparingBackend" as const,
      backend: { _tag: "Cuda" as const, hardwareLabel: "NVIDIA GPU" },
    };
    expect(report(installing, { _tag: "Starting", phase })).toEqual({
      _tag: "Starting",
      phase,
    });
  });
});
