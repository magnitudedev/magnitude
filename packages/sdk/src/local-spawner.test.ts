import { describe, expect, it } from "vitest";
import { Option } from "effect";
import type { AcnRegistration } from "@magnitudedev/protocol";
import { decideDaemonAction, type HealthResponse } from "./local-spawner";

const registration: AcnRegistration = {
  id: "acn-1",
  version: "1.0.0",
  url: "http://127.0.0.1:1234",
  pid: 1234,
  timestamp: 1,
};

const health = (version: string): HealthResponse => ({
  service: "magnitude-acn",
  version,
  id: registration.id,
  pid: registration.pid,
});

describe("decideDaemonAction", () => {
  it("spawns when registration is missing", () => {
    expect(
      decideDaemonAction({
        registration: Option.none(),
        health: Option.none(),
      })
    ).toEqual({ type: "spawn", reason: "missing" });
  });

  it("spawns when registration is stale", () => {
    expect(
      decideDaemonAction({
        registration: Option.some(registration),
        health: Option.none(),
      })
    ).toEqual({ type: "spawn", reason: "stale" });
  });

  it("connects to a same-version healthy ACN", () => {
    expect(
      decideDaemonAction({
        registration: Option.some(registration),
        health: Option.some(health("1.0.0")),
      })
    ).toEqual({
      type: "connect",
      url: registration.url,
      reason: "same-version",
    });
  });

  it("spawns when health belongs to a different ACN owner", () => {
    expect(
      decideDaemonAction({
        registration: Option.some(registration),
        health: Option.some({ ...health("1.0.0"), id: "other-acn" }),
      })
    ).toEqual({ type: "spawn", reason: "stale" });
  });
});
