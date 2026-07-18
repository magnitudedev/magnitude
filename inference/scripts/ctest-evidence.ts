import { createHash } from "node:crypto";
import { realpath, stat } from "node:fs/promises";
import { isAbsolute, relative, sep } from "node:path";

export interface CTestInventoryEntry {
  readonly name: string;
  readonly command: ReadonlyArray<string>;
  readonly config: string | null;
}

export interface CTestInventory {
  readonly schemaVersion: 1;
  readonly generatedAt: string;
  readonly names: ReadonlyArray<string>;
  readonly tests: ReadonlyArray<CTestInventoryEntry>;
  readonly rawSha256: string;
  readonly rawJson: string;
  readonly raw: unknown;
}

export interface CTestJunitSummary {
  readonly tests: number;
  readonly failures: number;
  readonly disabled: number;
  readonly skipped: number;
  readonly names: ReadonlyArray<string>;
}

const escapeRegex = (value: string) =>
  value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

export const ctestSelectionArguments = (options: {
  readonly buildDirectory: string;
  readonly testNames: ReadonlyArray<string>;
  readonly setupNames: ReadonlyArray<string>;
}) => {
  if (options.testNames.length === 0) {
    throw new Error("CTest selection requires at least one declared test");
  }
  const testExpression = `^(${options.testNames.map(escapeRegex).join("|")})$`;
  const setupExpression = `^(${options.setupNames.map(escapeRegex).join("|")})$`;
  return [
    "--test-dir",
    options.buildDirectory,
    "--build-config",
    "Release",
    "-R",
    testExpression,
    ...(options.setupNames.length === 0
      ? []
      : ["--fixture-exclude-setup", setupExpression]),
  ];
};

const capture = async (
  command: ReadonlyArray<string>,
  cwd: string,
  environment?: Readonly<Record<string, string>>
) => {
  const child = Bun.spawn([...command], {
    cwd,
    ...(environment === undefined ? {} : { env: environment }),
    stdout: "pipe",
    stderr: "pipe",
  });
  const [exitCode, stdout, stderr] = await Promise.all([
    child.exited,
    new Response(child.stdout).text(),
    new Response(child.stderr).text(),
  ]);
  if (exitCode !== 0) {
    throw new Error(stderr.trim() || `${command[0]} exited with ${exitCode}`);
  }
  return stdout;
};

const sameSetInOrder = (
  actual: ReadonlyArray<string>,
  expected: ReadonlyArray<string>
) => {
  const sortedActual = [...actual].sort();
  const sortedExpected = [...expected].sort();
  return (
    sortedActual.length === sortedExpected.length &&
    sortedActual.every((value, index) => value === sortedExpected[index])
  );
};

const assertUnique = (values: ReadonlyArray<string>, description: string) => {
  if (new Set(values).size !== values.length) {
    throw new Error(`${description} contains duplicate names`);
  }
};

const pathIsWithin = (root: string, path: string) => {
  const child = relative(root, path);
  return (
    child.length > 0 &&
    child !== ".." &&
    !child.startsWith(`..${sep}`) &&
    !isAbsolute(child)
  );
};

/**
 * Resolve CTest's selection before execution and require a one-to-one mapping
 * from every declared test name to a concrete executable inside the build.
 */
export const captureCTestInventory = async (options: {
  readonly cwd: string;
  readonly buildDirectory: string;
  readonly testNames: ReadonlyArray<string>;
  readonly setupNames: ReadonlyArray<string>;
  readonly environment?: Readonly<Record<string, string>>;
}): Promise<CTestInventory> => {
  assertUnique(options.testNames, "declared CTest selection");
  const selection = ctestSelectionArguments(options);
  const rawText = await capture(
    ["ctest", ...selection, "--show-only=json-v1"],
    options.cwd,
    options.environment
  );
  const raw = JSON.parse(rawText) as Record<string, unknown>;
  if (raw.kind !== "ctestInfo" || !Array.isArray(raw.tests)) {
    throw new Error("CTest --show-only returned an unsupported document");
  }
  const tests = raw.tests.map((unknownTest, index) => {
    if (
      unknownTest === null ||
      typeof unknownTest !== "object" ||
      Array.isArray(unknownTest)
    ) {
      throw new Error(`CTest inventory test ${index} is not an object`);
    }
    const test = unknownTest as Record<string, unknown>;
    if (
      typeof test.name !== "string" ||
      !Array.isArray(test.command) ||
      test.command.length === 0 ||
      !test.command.every((entry) => typeof entry === "string")
    ) {
      throw new Error(`CTest inventory test ${index} has no concrete command`);
    }
    return {
      name: test.name,
      command: test.command as string[],
      config: typeof test.config === "string" ? test.config : null,
    } satisfies CTestInventoryEntry;
  });
  const names = tests.map((test) => test.name);
  assertUnique(names, "resolved CTest selection");
  if (!sameSetInOrder(names, options.testNames)) {
    throw new Error(
      `resolved CTest names differ from the declaration: expected ${options.testNames.join(", ")}; got ${names.join(", ")}`
    );
  }
  const canonicalBuild = await realpath(options.buildDirectory);
  for (const test of tests) {
    const executable = test.command[0]!;
    if (!isAbsolute(executable)) {
      throw new Error(`CTest ${test.name} command is not absolute: ${executable}`);
    }
    const canonicalExecutable = await realpath(executable);
    if (
      !pathIsWithin(canonicalBuild, canonicalExecutable) ||
      !(await stat(canonicalExecutable)).isFile()
    ) {
      throw new Error(
        `CTest ${test.name} does not resolve to a build artifact: ${executable}`
      );
    }
  }
  return {
    schemaVersion: 1,
    generatedAt: new Date().toISOString(),
    names,
    tests,
    rawSha256: createHash("sha256").update(rawText).digest("hex"),
    rawJson: rawText,
    raw,
  };
};

const decodeXml = (value: string) =>
  value
    .replaceAll("&quot;", '"')
    .replaceAll("&apos;", "'")
    .replaceAll("&lt;", "<")
    .replaceAll("&gt;", ">")
    .replaceAll("&amp;", "&");

const integerAttribute = (document: string, name: string) => {
  const value = new RegExp(`\\b${name}="([0-9]+)"`).exec(document)?.[1];
  if (value === undefined) {
    throw new Error(`CTest JUnit is missing ${name}`);
  }
  return Number(value);
};

/** Verify that CTest's post-run JUnit record contains exactly the declared tests. */
export const verifyCTestJunit = async (
  path: string,
  expectedNames: ReadonlyArray<string>
): Promise<CTestJunitSummary> => {
  const document = await Bun.file(path).text();
  const names = [...document.matchAll(/<testcase\b[^>]*\bname="([^"]+)"/g)].map(
    (match) => decodeXml(match[1]!)
  );
  assertUnique(names, "CTest JUnit");
  if (!sameSetInOrder(names, expectedNames)) {
    throw new Error(
      `CTest JUnit names differ from the declaration: expected ${expectedNames.join(", ")}; got ${names.join(", ")}`
    );
  }
  const summary = {
    tests: integerAttribute(document, "tests"),
    failures: integerAttribute(document, "failures"),
    disabled: integerAttribute(document, "disabled"),
    skipped: integerAttribute(document, "skipped"),
    names,
  };
  if (summary.tests !== expectedNames.length) {
    throw new Error(
      `CTest JUnit reported ${summary.tests} tests; expected ${expectedNames.length}`
    );
  }
  return summary;
};
