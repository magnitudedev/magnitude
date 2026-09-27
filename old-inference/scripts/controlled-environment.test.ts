import { describe, expect, test } from "vitest";
import {
  controlledEnvironment,
  controlledEnvironmentEvidence,
} from "./controlled-environment";

describe("controlled parity environment", () => {
  test("inherits only declared build inputs and installs deterministic locale values", () => {
    const environment = controlledEnvironment(
      { CMAKE_BUILD_PARALLEL_LEVEL: "4" },
      {
        PATH: "/tools",
        HOME: "/home/test",
        CFLAGS: "-fsanitize=address",
        SECRET_TOKEN: "must-not-leak",
        LANG: "host-locale",
      }
    );

    expect(environment).toMatchObject({
      PATH: "/tools",
      HOME: "/home/test",
      LC_ALL: "C",
      LANG: "C",
      TZ: "UTC",
      CMAKE_BUILD_PARALLEL_LEVEL: "4",
    });
    expect(environment).not.toHaveProperty("CFLAGS");
    expect(environment).not.toHaveProperty("SECRET_TOKEN");
  });

  test("records names and a value-sensitive digest without recording values", () => {
    const first = controlledEnvironmentEvidence({ PATH: "/a", TZ: "UTC" });
    const second = controlledEnvironmentEvidence({ TZ: "UTC", PATH: "/b" });

    expect(first.names).toEqual(["PATH", "TZ"]);
    expect(first).not.toHaveProperty("values");
    expect(first.sha256).toMatch(/^[a-f0-9]{64}$/);
    expect(first.sha256).not.toBe(second.sha256);
  });
});
