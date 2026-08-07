import { describe, expect, it } from "vitest";
import { Effect, Option, Schema } from "effect";
import { AcnHealthResponseSchema } from "@magnitudedev/acn-protocol";
import { makeHealthResponse } from "./identity";

describe("ACN identity", () => {
  it("exposes the exact registry owner identity through health", () => {
    expect(
      makeHealthResponse("1.2.3", { _tag: "Ready" }, "owner-1", 1234, 42)
    ).toEqual({
      service: "magnitude-acn",
      version: "1.2.3",
      revision: 42,
      id: "owner-1",
      pid: 1234,
      state: { _tag: "Ready" },
    });
  });

  it("encodes optional startup progress as an ordinary optional wire field", async () => {
    const encode = Schema.encode(AcnHealthResponseSchema);
    const withoutProgress = await Effect.runPromise(
      encode(
        makeHealthResponse(
          "1.2.3",
          {
            _tag: "Starting",
            activity: "Resolving",
            progress: Option.none(),
          },
          "owner-1",
          1234
        )
      )
    );
    const withProgress = await Effect.runPromise(
      encode(
        makeHealthResponse(
          "1.2.3",
          {
            _tag: "Starting",
            activity: {
              _tag: "Installing",
              phase: "DownloadingInferenceEngine",
              plan: {
                daemonBytes: 30,
                inferenceEngineBytes: 70,
                inferenceEngineBytesExact: true,
              },
            },
            progress: Option.some({
              completed: 10,
              totalBytes: 20,
              unit: "Bytes",
              attempt: Option.some(1),
            }),
          },
          "owner-1",
          1234
        )
      )
    );

    expect(withoutProgress.state).toEqual({
      _tag: "Starting",
      activity: "Resolving",
    });
    expect(withProgress.state).toEqual({
      _tag: "Starting",
      activity: {
        _tag: "Installing",
        phase: "DownloadingInferenceEngine",
        plan: {
          daemonBytes: 30,
          inferenceEngineBytes: 70,
          inferenceEngineBytesExact: true,
        },
      },
      progress: {
        completed: 10,
        totalBytes: 20,
        unit: "Bytes",
        attempt: 1,
      },
    });
  });
});
