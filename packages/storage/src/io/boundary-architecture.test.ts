import { readFile } from "node:fs/promises";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";

const repositoryRoot = resolve(import.meta.dirname, "../../../..");

const boundaryFiles = [
  "packages/sdk/src/acn-jit/local-daemon-spawner.ts",
  "packages/acn/src/daemon-registration.ts",
  "packages/acn/src/session-runtime-options.ts",
  "packages/storage/src/io/structured-file.ts",
  "packages/client-common/src/sync/display-view-store.ts",
  "packages/client-common/src/sync/apply-stream-event.ts",
] as const;

describe("external-data boundary architecture", () => {
  it("does not use throwing codecs or swallow defects", async () => {
    for (const relativePath of boundaryFiles) {
      const source = await readFile(
        resolve(repositoryRoot, relativePath),
        "utf8"
      );
      expect(source, relativePath).not.toMatch(
        /Schema\.(?:decodeUnknownSync|decodeSync|validateSync)\b/
      );
      expect(source, relativePath).not.toMatch(/\bJSON\.parse\s*\(/);
      expect(source, relativePath).not.toMatch(/Effect\.catchAllDefect\b/);
    }
  });

  it("keeps the removed JSON lock protocol out of production coordination", async () => {
    for (const relativePath of [
      "packages/sdk/src/acn-jit/local-daemon-spawner.ts",
      "packages/acn/src/daemon-registration.ts",
      "packages/acn/src/daemon-lifecycle.ts",
    ]) {
      const source = await readFile(
        resolve(repositoryRoot, relativePath),
        "utf8"
      );
      expect(source, relativePath).not.toContain("registry.lock");
      expect(source, relativePath).not.toContain("owner.json");
    }
  });
});
