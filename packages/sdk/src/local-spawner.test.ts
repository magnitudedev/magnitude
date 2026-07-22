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
        candidateVersion: "1.0.0",
        registration: Option.none(),
        health: Option.none(),
      })
    ).toEqual({ type: "unavailable", reason: "missing" });
  });

  it("spawns when registration is stale", () => {
    expect(
      decideDaemonAction({
        candidateVersion: "1.0.0",
        registration: Option.some(registration),
        health: Option.none(),
      })
    ).toEqual({ type: "unavailable", reason: "stale" });
  });

  it("connects to a same-version healthy ACN", () => {
    expect(
      decideDaemonAction({
        candidateVersion: "1.0.0",
        registration: Option.some(registration),
        health: Option.some(health("1.0.0")),
      })
    ).toEqual({
      type: "reuse",
      url: registration.url,
      reason: "same-release",
    });
  });

  it("spawns when health belongs to a different ACN owner", () => {
    expect(
      decideDaemonAction({
        candidateVersion: "1.0.0",
        registration: Option.some(registration),
        health: Option.some({ ...health("1.0.0"), id: "other-acn" }),
      })
    ).toEqual({ type: "unavailable", reason: "stale" });
  });

  it("reuses a newer healthy ACN for forward compatibility", () => {
    expect(decideDaemonAction({
      candidateVersion: "1.0.0",
      registration: Option.some({ ...registration, version: "2.0.0" }),
      health: Option.some(health("2.0.0")),
    })).toEqual({
      type: "reuse",
      url: registration.url,
      reason: "newer-release",
    });
  });

  it("replaces only an older healthy ACN", () => {
    expect(decideDaemonAction({
      candidateVersion: "2.0.0",
      registration: Option.some(registration),
      health: Option.some(health("1.0.0")),
    })).toEqual({ type: "replace", owner: registration });
  });

  it("replaces an older Magnitude dev build", () => {
    const incumbent = "0.0.1-alpha.22+dev.2c5b178.1784755698047";
    const candidate = "0.0.1-alpha.22+dev.2c5b178.1784757574495";
    const owner = { ...registration, version: incumbent };

    expect(decideDaemonAction({
      candidateVersion: candidate,
      registration: Option.some(owner),
      health: Option.some(health(incumbent)),
    })).toEqual({ type: "replace", owner });
  });

  it("reuses a newer Magnitude dev build", () => {
    const candidate = "0.0.1-alpha.22+dev.2c5b178.1784755698047";
    const incumbent = "0.0.1-alpha.22+dev.2c5b178.1784757574495";

    expect(decideDaemonAction({
      candidateVersion: candidate,
      registration: Option.some({ ...registration, version: incumbent }),
      health: Option.some(health(incumbent)),
    })).toEqual({
      type: "reuse",
      url: registration.url,
      reason: "newer-release",
    });
  });
});
