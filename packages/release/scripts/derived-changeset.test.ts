import { describe, expect, it } from "vitest";
import { derivedChangeset, derivedChangesetName, isDerivedChangeset } from "./derived-changeset";

describe("derived changeset", () => {
  it("generates nothing when every affected package is already declared", () => {
    expect(derivedChangeset({ rpcVersion: 2, releases: [] })).toBeUndefined();
  });
  it("bumps the CLI and the plugin for a new RPC contract", () => {
    const text = derivedChangeset({
      rpcVersion: 2,
      releases: ["@magnitudedev/cli", "@magnitudedev/pi-extension"],
    });
    expect(text).toContain('"@magnitudedev/cli": patch');
    expect(text).toContain('"@magnitudedev/pi-extension": patch');
    expect(text).toContain("Update the RPC contract to v2");
  });
  it("bumps only the undeclared package", () => {
    const text = derivedChangeset({ rpcVersion: 2, releases: ["@magnitudedev/pi-extension"] });
    expect(text).toContain('"@magnitudedev/pi-extension": patch');
    expect(text).not.toContain("@magnitudedev/cli");
  });
  it("names one file per RPC version and recognizes only those files as derived", () => {
    expect(derivedChangesetName(2)).toBe("rpc-v2.md");
    expect(isDerivedChangeset("rpc-v2.md")).toBe(true);
    expect(isDerivedChangeset("rpc-plugins.md")).toBe(false);
    expect(isDerivedChangeset("major-pumas-happen.md")).toBe(false);
  });
});
