import { describe, expect, it } from "vitest";
import { derivedChangeset } from "./derived-changeset";

describe("derived changeset", () => {
  it("generates nothing when every change is already declared", () => {
    expect(derivedChangeset({ rpcChanged: true, releases: [] })).toBeUndefined();
  });
  it("bumps only the plugin for a plugin-only content change", () => {
    const text = derivedChangeset({
      rpcChanged: false,
      releases: ["@magnitudedev/pi-extension"],
    });
    expect(text).toContain('"@magnitudedev/pi-extension": patch');
    expect(text).not.toContain("@magnitudedev/cli");
    expect(text).toContain("Rebuild bundled harness plugins");
  });
  it("bumps the CLI and the plugin when the RPC contract moved", () => {
    const text = derivedChangeset({
      rpcChanged: true,
      releases: ["@magnitudedev/cli", "@magnitudedev/pi-extension"],
    });
    expect(text).toContain('"@magnitudedev/cli": patch');
    expect(text).toContain('"@magnitudedev/pi-extension": patch');
    expect(text).toContain("Update the RPC contract");
  });
});
