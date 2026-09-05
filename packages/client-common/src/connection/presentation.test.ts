import {
  ConnectionStateSchema,
  ServiceUnavailable,
  type ConnectionState,
} from "@magnitudedev/sdk";
import { Option, Schema } from "effect";
import { describe, expect, it } from "vitest";
import { initialPresentation, reducePresentation } from "./presentation";

const ready = Schema.decodeUnknownSync(ConnectionStateSchema)({
  _tag: "Ready",
  service: { id: "one", version: "0.0.1", rpcVersion: 1 },
});
const connecting = (
  reason: "initial" | "recovery" | "retry"
): ConnectionState => ({ _tag: "Connecting", reason, activity: Option.none() });
const failed: ConnectionState = {
  _tag: "Failed",
  error: new ServiceUnavailable({
    origin: "http://localhost:10100",
    message: "offline",
  }),
};

describe("first-party presentation fold", () => {
  it("keeps failed initial retries in bootstrap", () => {
    const failure = reducePresentation(initialPresentation, failed, 0);
    expect(failure).toMatchObject({
      _tag: "Bootstrapping",
      lifecycle: { _tag: "Failed" },
    });
    expect(reducePresentation(failure, connecting("retry"), 1)).toMatchObject({
      _tag: "Bootstrapping",
      lifecycle: { _tag: "Checking" },
    });
  });
  it("uses the SDK reason, and gives a failed recovery and its retries just one occurrence", () => {
    const first = reducePresentation(initialPresentation, ready, 0);
    expect(first).toMatchObject({ _tag: "Online", occurrence: 0 });
    const recovery = reducePresentation(first, connecting("recovery"), 1);
    expect(recovery).toMatchObject({ _tag: "Recovering", occurrence: 1 });
    const failure = reducePresentation(recovery, failed, 2);
    expect(failure).toMatchObject({
      _tag: "Recovering",
      occurrence: 1,
      lifecycle: { _tag: "Failed" },
    });
    const retry = reducePresentation(failure, connecting("recovery"), 3);
    expect(retry).toMatchObject({
      _tag: "Recovering",
      occurrence: 1,
      lifecycle: { _tag: "Checking" },
    });
    const online = reducePresentation(retry, ready, 4);
    expect(online).toMatchObject({ _tag: "Online", occurrence: 1 });
    expect(reducePresentation(online, ready, 5)).toBe(online);
    expect(reducePresentation(online, connecting("recovery"), 6)).toMatchObject(
      { _tag: "Recovering", occurrence: 2 }
    );
  });
  it("honors a recovery reason even without locally observing an earlier Ready", () => {
    expect(
      reducePresentation(initialPresentation, connecting("recovery"), 0)
    ).toMatchObject({ _tag: "Recovering", occurrence: 1 });
  });
  it("retains the last presentation when closed", () => {
    const online = reducePresentation(initialPresentation, ready, 0);
    expect(reducePresentation(online, { _tag: "Closed" }, 1)).toBe(online);
  });
});
