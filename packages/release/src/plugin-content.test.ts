import * as FileSystem from "@effect/platform/FileSystem";
import { BunContext } from "@effect/platform-bun";
import { Effect, Option, Schema } from "effect";
import { describe, expect, it } from "vitest";
import { packageContentFingerprint, sha256 } from "./plugin-content";
import { artifactIntegrity } from "./plugin-artifacts";
import {
  readAcceptedPluginCandidate,
  readPluginCandidate,
} from "./plugin-candidate";
import { PreparedReleaseSchema } from "./release-plan";
import type { JsonValue } from "@magnitudedev/utils/schema";

describe("plugin artifact identity", () => {
  it("fingerprints install behavior and bundled bytes, not incidental release/tool metadata", () => {
    const manifest = {
      name: "@magnitudedev/pi-extension",
      version: "1.0.0",
      dependencies: { effect: "3.21.2" },
    };
    const files = { "dist/magnitude.js": sha256("bundle") };
    const fingerprint = packageContentFingerprint(manifest, 1, files);
    expect(
      packageContentFingerprint(
        {
          ...manifest,
          version: "1.0.1",
          devDependencies: { tool: "new" },
          scripts: { build: "different" },
        },
        1,
        files
      )
    ).toBe(fingerprint);
    const changes: Record<string, JsonValue>[] = [
      { dependencies: { effect: "4.0.0" } },
      { scripts: { install: "different" } },
      { exports: "./dist/other.js" },
    ];
    for (const changed of changes) {
      expect(
        packageContentFingerprint({ ...manifest, ...changed }, 1, files)
      ).not.toBe(fingerprint);
    }
    expect(packageContentFingerprint(manifest, 2, files)).not.toBe(fingerprint);
    expect(
      packageContentFingerprint(manifest, 1, {
        "dist/magnitude.js": sha256("new bundle"),
      })
    ).not.toBe(fingerprint);
  });
  it("ties acceptance to both the exact plan and the unchanged tarball", async () => {
    await Effect.runPromise(
      Effect.scoped(
        Effect.gen(function* () {
          const fs = yield* FileSystem.FileSystem;
          const root = yield* fs.makeTempDirectoryScoped({
            prefix: "magnitude-plugin-receipt-",
          });
          const bytes = new TextEncoder().encode("packed candidate");
          const artifact = {
            host: "pi" as const,
            name: "@magnitudedev/pi-extension",
            version: "1.0.0",
            rpcVersion: 1,
            contentFingerprint: sha256("bundle"),
            filename: "pi.tgz",
            integrity: artifactIntegrity(bytes),
          };
          const plan = {
            format: 2 as const,
            baseline: Option.none(),
            cliVersion: "1.0.0",
            revision: 1,
            rpc: { version: 1, fingerprint: sha256("wire") },
            semanticBreaks: [],
            plugins: [{ artifact, publish: true }],
          };
          const writePlan = (cliVersion: string) =>
            Schema.encode(Schema.parseJson(PreparedReleaseSchema))({
              ...plan,
              cliVersion,
            }).pipe(
              Effect.flatMap((text) =>
                fs.writeFileString(`${root}/release-plan.json`, text)
              )
            );
          yield* writePlan("1.0.0");
          yield* fs.writeFile(`${root}/pi.tgz`, bytes);
          const candidate = yield* readPluginCandidate(root);
          yield* fs.writeFileString(
            `${root}/accepted.json`,
            JSON.stringify(candidate.receipt)
          );
          yield* readAcceptedPluginCandidate(root);
          yield* writePlan("1.0.1");
          expect(
            (yield* Effect.either(readAcceptedPluginCandidate(root)))._tag
          ).toBe("Left");
          yield* writePlan("1.0.0");
          yield* fs.writeFileString(`${root}/pi.tgz`, "tampered");
          expect(
            (yield* Effect.either(readAcceptedPluginCandidate(root)))._tag
          ).toBe("Left");
        })
      ).pipe(Effect.provide(BunContext.layer))
    );
  });
});
