import { afterEach, describe, expect, test } from "vitest";
import { createHash } from "node:crypto";
import { mkdtemp, mkdir, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { Effect } from "effect";
import {
  defaultExcludedDirectoryNames,
  sourceInventory,
} from "./source-inventory";

const temporaryDirectories: Array<string> = [];

afterEach(async () => {
  await Promise.all(
    temporaryDirectories.splice(0).map((directory) =>
      rm(directory, { force: true, recursive: true })
    )
  );
});

describe("sourceInventory", () => {
  test("reports and excludes generated dependency/cache directories", async () => {
    const directory = await mkdtemp(join(tmpdir(), "icn-source-inventory-"));
    temporaryDirectories.push(directory);
    await writeFile(join(directory, "source.cpp"), "int main() {}\n");
    await writeFile(join(directory, "target"), "a source file, not a directory\n");
    const excludedDirectoryNames = defaultExcludedDirectoryNames;
    for (const excluded of excludedDirectoryNames) {
      const generatedDirectory = join(directory, "nested", excluded, "debug");
      await mkdir(generatedDirectory, { recursive: true });
      await writeFile(join(generatedDirectory, "generated.bin"), excluded);
    }

    const inventory = await Effect.runPromise(sourceInventory(directory));

    expect(inventory.excludedDirectoryNames).toEqual(excludedDirectoryNames);
    expect(inventory.entries.map((entry) => entry.sourcePath)).toEqual([
      "source.cpp",
      "target",
    ]);
  });

  test("uses an explicit exclusion contract that a manifest verifier can replay", async () => {
    const directory = await mkdtemp(join(tmpdir(), "icn-source-inventory-"));
    temporaryDirectories.push(directory);
    await mkdir(join(directory, "native"), { recursive: true });
    await mkdir(join(directory, "crates", "probe"), { recursive: true });
    await writeFile(join(directory, "Cargo.toml"), "[workspace]\n");
    await writeFile(join(directory, "native", "generated.cpp"), "generated\n");
    await writeFile(join(directory, "crates", "probe", "lib.rs"), "source\n");

    const inventory = await Effect.runPromise(
      sourceInventory(directory, {
        excludedDirectoryNames: [...defaultExcludedDirectoryNames, "native"],
      })
    );

    expect(inventory.excludedDirectoryNames).toEqual([
      ...defaultExcludedDirectoryNames,
      "native",
    ]);
    expect(inventory.entries.map((entry) => entry.sourcePath)).toEqual([
      "Cargo.toml",
      "crates/probe/lib.rs",
    ]);
  });

  test("rejects ambiguous exclusion names", async () => {
    const directory = await mkdtemp(join(tmpdir(), "icn-source-inventory-"));
    temporaryDirectories.push(directory);
    await expect(
      Effect.runPromise(
        sourceInventory(directory, {
          excludedDirectoryNames: ["target", "nested/cache"],
        })
      )
    ).rejects.toThrow("unique, non-empty directory base names");
  });

  test("orders normalized path bytes rather than filesystem components", async () => {
    const directory = await mkdtemp(join(tmpdir(), "icn-source-inventory-"));
    temporaryDirectories.push(directory);
    await mkdir(join(directory, "a"), { recursive: true });
    await writeFile(join(directory, "a-b"), "dash");
    await writeFile(join(directory, "a", "b"), "slash");

    const inventory = await Effect.runPromise(sourceInventory(directory));
    const dash = createHash("sha256").update("dash").digest("hex");
    const slash = createHash("sha256").update("slash").digest("hex");
    const expected = createHash("sha256")
      .update(`a-b\0${4}\0${dash}\na/b\0${5}\0${slash}\n`)
      .digest("hex");

    expect(inventory.entries.map((entry) => entry.sourcePath)).toEqual([
      "a-b",
      "a/b",
    ]);
    expect(inventory.sha256).toBe(expected);
  });
});
