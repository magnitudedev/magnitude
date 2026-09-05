import { Effect, Option } from "effect";
import { describe, expect, it } from "vitest";
import {
  allocateRevision,
  allocateRpcVersion,
  isAwaitingPublication,
  planPlugin,
  validatePreparedRelease,
  type PreparedRelease,
  type PublicBaseline,
} from "../src/release-plan";
import { publishTag } from "./release-channel";

const hash = "a".repeat(64);
const artifact = {
  host: "pi" as const,
  name: "@magnitudedev/pi-extension",
  version: "1.2.3",
  rpcVersion: 8,
  contentFingerprint: hash,
  filename: "pi.tgz",
  integrity: `sha512-${"a".repeat(86)}==`,
};
const baseline = Option.some<PublicBaseline>({
  tag: "@magnitudedev/cli@1.0.0",
  sourceCommit: "a".repeat(40),
  cliVersion: "1.0.0",
  rpc: Option.some({ version: 8, fingerprint: hash }),
});
const published = Option.some(artifact);
const metadata = {
  name: artifact.name,
  version: artifact.version,
  rpcVersion: artifact.rpcVersion,
  contentFingerprint: hash,
  files: {},
};
describe("public-baseline release allocation", () => {
  const prepared: PreparedRelease = {
    format: 2,
    baseline,
    cliVersion: "1.1.0",
    revision: 28,
    rpc: { version: 8, fingerprint: hash },
    semanticBreaks: [],
    plugins: [{ artifact, publish: false }],
  };
  it("does not block pre exit on an unpublished alpha allocation", () => {
    expect(isAwaitingPublication(prepared, "1.1.0", baseline)).toBe(true);
    expect(
      isAwaitingPublication(
        { ...prepared, cliVersion: "1.1.0-alpha.4" },
        "1.1.0-alpha.4",
        baseline
      )
    ).toBe(false);
  });
  it("advances the revision exactly once per CLI version, prereleases included", async () => {
    const advance = (previous: PreparedRelease, cliVersion: string) =>
      Effect.runPromise(allocateRevision(previous, cliVersion));
    expect(await advance(prepared, "1.1.0")).toBe(28);
    expect(await advance(prepared, "1.2.0-alpha.0")).toBe(29);
    const alpha = { ...prepared, cliVersion: "1.2.0-alpha.0", revision: 29 };
    expect(await advance(alpha, "1.2.0-alpha.0")).toBe(29);
    expect(await advance(alpha, "1.2.0-alpha.1")).toBe(30);
    expect(await advance({ ...alpha, cliVersion: "1.2.0-alpha.1", revision: 30 }, "1.2.0")).toBe(31);
    const exhausted = { ...prepared, revision: Number.MAX_SAFE_INTEGER };
    expect(
      (await Effect.runPromise(Effect.either(allocateRevision(exhausted, "9.0.0"))))._tag
    ).toBe("Left");
  });

  it("requires one selection per declared host, independently of package naming", async () => {
    const renamed = {
      ...prepared,
      plugins: [
        {
          artifact: { ...artifact, name: "@magnitudedev/renamed-pi" },
          publish: false,
        },
      ],
    };
    expect(await Effect.runPromise(validatePreparedRelease(renamed))).toBe(
      renamed
    );
    await expect(
      Effect.runPromise(validatePreparedRelease({ ...prepared, plugins: [] }))
    ).rejects.toThrow("Missing plugin selection for hosts: pi");
    await expect(
      Effect.runPromise(
        validatePreparedRelease({
          ...prepared,
          plugins: [...prepared.plugins, ...renamed.plugins],
        })
      )
    ).rejects.toThrow("Plugin selection must be unique");
  });
  it("publishes plugins only in the plan's own channel", async () => {
    const outcome = (cliVersion: string, version: string) =>
      Effect.runPromise(
        Effect.either(
          validatePreparedRelease({
            ...prepared,
            cliVersion,
            plugins: [{ artifact: { ...artifact, version }, publish: true }],
          })
        )
      ).then((result) => result._tag);
    expect(await outcome("1.1.0-beta.0", "1.2.3")).toBe("Left");
    expect(await outcome("1.1.0-beta.0", "1.2.4-beta.0")).toBe("Right");
    expect(await outcome("1.1.0", "1.2.4-beta.0")).toBe("Left");
    expect(await outcome("1.1.0", "1.2.4")).toBe("Right");
    expect(publishTag("1.2.4-beta.0")).toBe("beta");
    expect(publishTag("1.2.4")).toBe("latest");
  });
  it("advances the RPC version from the previous plan in every channel", async () => {
    const allocate = (previous: PreparedRelease, fingerprint: string, breaks: string[] = []) =>
      Effect.runPromise(allocateRpcVersion(previous, fingerprint, breaks)).then((rpc) => rpc.version);
    expect(await allocate(prepared, hash)).toBe(8);
    expect(await allocate(prepared, "b".repeat(64))).toBe(9);
    expect(await allocate(prepared, hash, ["behavior"])).toBe(9);
    const alpha = { ...prepared, cliVersion: "1.2.0-alpha.0", rpc: { version: 9, fingerprint: "b".repeat(64) } };
    expect(await allocate(alpha, "b".repeat(64))).toBe(9);
    expect(await allocate(alpha, "c".repeat(64))).toBe(10);
    // A reverted contract never reuses a number.
    expect(await allocate(alpha, hash)).toBe(10);
    const exhausted = { ...prepared, rpc: { version: Number.MAX_SAFE_INTEGER, fingerprint: hash } };
    expect(
      (await Effect.runPromise(Effect.either(allocateRpcVersion(exhausted, "b".repeat(64), []))))._tag
    ).toBe("Left");
  });
  it("retains an unchanged published artifact, ignoring an incidental package bump", async () => {
    const plan = await Effect.runPromise(
      planPlugin({ ...metadata, version: "1.2.4" }, published)
    );
    expect(plan.publish).toBe(false);
    expect(plan.version).toBe("1.2.3");
  });
  it("publishes a first plugin at its own version when npm has none", async () => {
    const plan = await Effect.runPromise(planPlugin(metadata, Option.none()));
    expect(plan.publish).toBe(true);
    expect(plan.version).toBe("1.2.3");
  });
  it("bumps for shipped code, dependency or protocol changes and respects a larger human bump", async () => {
    for (const changed of [
      { contentFingerprint: "b".repeat(64) },
      { rpcVersion: 9 },
    ]) {
      const plan = await Effect.runPromise(
        planPlugin({ ...metadata, ...changed }, published)
      );
      expect(plan.publish).toBe(true);
      expect(plan.version).toBe("1.2.4");
    }
    expect(
      (
        await Effect.runPromise(
          planPlugin({ ...metadata, version: "2.0.0", rpcVersion: 9 }, published)
        )
      ).version
    ).toBe("2.0.0");
  });
});
