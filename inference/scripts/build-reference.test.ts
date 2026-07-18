import { afterEach, describe, expect, test } from "vitest";
import { mkdir, mkdtemp, rm, symlink, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import {
  assertInventoryUnchanged,
  buildVerificationEvidence,
  findArtifact,
  inventoryUpstreamSource,
} from "./build-reference";

const temporaryDirectories: string[] = [];

afterEach(async () => {
  await Promise.all(
    temporaryDirectories.splice(0).map((directory) =>
      rm(directory, { recursive: true, force: true })
    )
  );
});

const dryRun = async (target: string, runTests = false) => {
  const child = Bun.spawn(
    [
      "bun",
      "run",
      "scripts/build-reference.ts",
      "--backend",
      "cpu",
      "--lane",
      "cargo-equivalent",
      "--target",
      target,
      "--dry-run",
      ...(runTests ? ["--run-tests"] : []),
    ],
    {
      cwd: resolve(import.meta.dirname, ".."),
      stdout: "pipe",
      stderr: "pipe",
    }
  );
  const [exitCode, stdout, stderr] = await Promise.all([
    child.exited,
    new Response(child.stdout).text(),
    new Response(child.stderr).text(),
  ]);
  expect(stderr).toBe("");
  expect(exitCode).toBe(0);
  return stdout;
};

describe("reference build integrity", () => {
  test("dry-run keeps focused upstream configuration free of oracle injection", async () => {
    const output = await dryRun("focused-tests");

    expect(output).toContain("cpu-cargo-equivalent-upstream");
    expect(output).not.toContain("CMAKE_PROJECT_INCLUDE");
    expect(output).not.toContain("ICN_PARITY_ORACLE_SOURCE_DIR");
  });

  test("dry-run injects the oracle only into its isolated build", async () => {
    const output = await dryRun("oracle");

    expect(output).toContain("cpu-cargo-equivalent-oracle");
    expect(output).toContain("CMAKE_PROJECT_INCLUDE");
    expect(output).toContain("ICN_PARITY_ORACLE_SOURCE_DIR");
  });

  test("direct CTest dry-run suppresses network-capable fixture setup", async () => {
    const output = await dryRun("focused-tests", true);

    expect(output).toContain("--fixture-exclude-setup");
    expect(output).toContain("^(test-download-model)$");
  });

  test("manifest inventory removes known test residue before hashing", async () => {
    const source = await mkdtemp(join(tmpdir(), "icn-upstream-source-"));
    temporaryDirectories.push(source);
    await writeFile(join(source, "native.cpp"), "int native = 1;\n");
    await writeFile(join(source, "test-grammar-output.tmp"), "stale\n");
    await writeFile(join(source, "test-json-schema-input.tmp"), "stale\n");

    const inventory = await inventoryUpstreamSource(source);

    expect(inventory.entries.map((entry) => entry.sourcePath)).toEqual([
      "native.cpp",
    ]);
    await expect(Bun.file(join(source, "test-grammar-output.tmp")).exists()).resolves.toBe(false);
    await expect(Bun.file(join(source, "test-json-schema-input.tmp")).exists()).resolves.toBe(false);
  });

  test("rejects source mutation across a build boundary", () => {
    const before = { sha256: "a", fileCount: 1, totalBytes: 2 };
    expect(() =>
      assertInventoryUnchanged(before, before, "source")
    ).not.toThrow();
    expect(() =>
      assertInventoryUnchanged(
        before,
        { sha256: "b", fileCount: 1, totalBytes: 2 },
        "source"
      )
    ).toThrow("source changed during the reference build");
  });

  test("derives assertion and sanitizer claims from compile and link evidence", async () => {
    const build = await mkdtemp(join(tmpdir(), "icn-reference-flags-"));
    temporaryDirectories.push(build);
    await mkdir(join(build, "CMakeFiles/tool.dir"), { recursive: true });
    await writeFile(
      join(build, "compile_commands.json"),
      JSON.stringify([{ arguments: ["cc", "-O3", "-DNDEBUG", "tool.c"] }])
    );
    await writeFile(join(build, "CMakeFiles/tool.dir/link.txt"), "cc tool.o -o tool\n");

    await expect(buildVerificationEvidence(build, {})).resolves.toMatchObject({
      assertions: false,
      sanitizers: [],
      evidence: { compileCommandCount: 1, linkCommandCount: 1 },
    });

    await writeFile(
      join(build, "compile_commands.json"),
      JSON.stringify([{ command: "cc -fsanitize=address tool.c" }])
    );
    await expect(buildVerificationEvidence(build, {})).resolves.toMatchObject({
      assertions: null,
      sanitizers: ["address"],
    });
  });

  test("rejects an artifact symlink that escapes its fresh build tree", async () => {
    const root = await mkdtemp(join(tmpdir(), "icn-reference-artifact-"));
    temporaryDirectories.push(root);
    const build = join(root, "build");
    const outside = join(root, "outside-tool");
    await mkdir(join(build, "bin"), { recursive: true });
    await writeFile(outside, "binary\n");
    await symlink(outside, join(build, "bin", "llama-bench"));

    await expect(findArtifact(build, "llama-bench")).rejects.toThrow(
      "escapes its build directory"
    );
  });
});
