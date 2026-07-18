import { afterEach, describe, expect, test } from "vitest";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
  ctestSelectionArguments,
  verifyCTestJunit,
} from "./ctest-evidence";

const temporaryDirectories: string[] = [];

afterEach(async () => {
  await Promise.all(
    temporaryDirectories.splice(0).map((directory) =>
      rm(directory, { force: true, recursive: true })
    )
  );
});

describe("CTest evidence", () => {
  test("creates an anchored exact-name selection and suppresses setup", () => {
    const arguments_ = ctestSelectionArguments({
      buildDirectory: "/tmp/build",
      testNames: ["test-a", "test-b+1"],
      setupNames: ["download-model"],
    });
    expect(arguments_).toContain("^(test-a|test-b\\+1)$");
    expect(arguments_).toContain("--fixture-exclude-setup");
    expect(arguments_).toContain("^(download-model)$");
  });

  test("accepts exact JUnit cardinality regardless of XML test order", async () => {
    const directory = await mkdtemp(join(tmpdir(), "icn-ctest-junit-"));
    temporaryDirectories.push(directory);
    const path = join(directory, "ctest.xml");
    await writeFile(
      path,
      `<?xml version="1.0"?><testsuite tests="2" failures="0" disabled="0" skipped="0"><testcase name="test-b"/><testcase name="test-a"/></testsuite>`
    );
    await expect(verifyCTestJunit(path, ["test-a", "test-b"])).resolves.toMatchObject({
      tests: 2,
      names: ["test-b", "test-a"],
    });
  });

  test("rejects missing or duplicate JUnit cases", async () => {
    const directory = await mkdtemp(join(tmpdir(), "icn-ctest-junit-"));
    temporaryDirectories.push(directory);
    const path = join(directory, "ctest.xml");
    await writeFile(
      path,
      `<testsuite tests="2" failures="0" disabled="0" skipped="0"><testcase name="test-a"/><testcase name="test-a"/></testsuite>`
    );
    await expect(verifyCTestJunit(path, ["test-a", "test-b"])).rejects.toThrow(
      "duplicate"
    );
  });
});
