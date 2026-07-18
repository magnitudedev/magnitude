import { Effect } from "effect";
import { createHash, randomBytes } from "node:crypto";
import { mkdir, readFile, readdir, unlink, writeFile } from "node:fs/promises";
import { isAbsolute, relative, resolve, sep } from "node:path";
import {
  loadTargetManifest,
  selectReferenceTargets,
  uniqueStrings,
} from "./build-parity-config";
import {
  prepareOfflineUpstreamFixtures,
  verifyOfflineUpstreamFixtures,
} from "./upstream-model-fixture";
import { verifyNativePin } from "./verify-native-pin";
import {
  captureCTestInventory,
  ctestSelectionArguments,
  verifyCTestJunit,
} from "./ctest-evidence";
import {
  controlledEnvironment,
  controlledEnvironmentEvidence,
} from "./controlled-environment";

type Backend = "cpu" | "metal";

interface Arguments {
  readonly backend: Backend;
  readonly targets: readonly string[];
  readonly outputDirectory?: string;
  readonly modelDirectory?: string;
  readonly parallel: number;
  readonly help: boolean;
}

const inferenceRoot = resolve(import.meta.dirname, "..");
const usage = `Build and run unchanged pinned upstream CTests.

Usage:
  bun run scripts/test-upstream.ts [options]

Options:
  --backend <cpu|metal>  Backend to test (default: metal on macOS, cpu elsewhere)
  --target <id>          Target-manifest entry with CTests (default: focused-tests; repeatable)
  --output-dir <path>    Empty evidence directory (default: unique results/parity/upstream/<run>)
  --model-dir <path>     Registry artifact root (default: ICN_PARITY_MODEL_DIR)
  --parallel <count>     Parallel native build jobs (default: 8)
  --help                 Show this help
`;

const requireNext = (args: readonly string[], index: number, option: string) => {
  const value = args[index + 1];
  if (value === undefined) throw new Error(`${option} requires a value`);
  return value;
};

const parseArguments = (): Arguments => {
  let backend: Backend = process.platform === "darwin" ? "metal" : "cpu";
  const targets: string[] = [];
  let outputDirectory: string | undefined;
  let modelDirectory: string | undefined;
  let parallel = 8;
  let help = false;
  const args = Bun.argv.slice(2);
  for (let index = 0; index < args.length; index += 1) {
    const argument = args[index];
    if (argument === "--backend") {
      const value = requireNext(args, index, argument);
      if (value !== "cpu" && value !== "metal") {
        throw new Error(`Unsupported backend: ${value}`);
      }
      backend = value;
      index += 1;
    } else if (argument === "--target") {
      targets.push(requireNext(args, index, argument));
      index += 1;
    } else if (argument === "--output-dir") {
      outputDirectory = requireNext(args, index, argument);
      index += 1;
    } else if (argument === "--model-dir") {
      modelDirectory = requireNext(args, index, argument);
      index += 1;
    } else if (argument === "--parallel") {
      const raw = requireNext(args, index, argument);
      parallel = Number(raw);
      if (!Number.isSafeInteger(parallel) || parallel < 1) {
        throw new Error(`Invalid parallel job count: ${raw}`);
      }
      index += 1;
    } else if (argument === "--help") {
      help = true;
    } else {
      throw new Error(`Unsupported argument: ${argument}`);
    }
  }
  return { backend, targets, outputDirectory, modelDirectory, parallel, help };
};

const run = async (
  command: readonly string[],
  cwd: string,
  environment: Readonly<Record<string, string>>
) => {
  const child = Bun.spawn([...command], {
    cwd,
    env: environment,
    stdin: "inherit",
    stdout: "inherit",
    stderr: "inherit",
  });
  return child.exited;
};

const referenceBuildDirectory = async (manifestPath: string) => {
  const value: unknown = JSON.parse(
    new TextDecoder().decode(await readFile(manifestPath))
  );
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    throw new Error("Reference builder produced an invalid manifest");
  }
  const manifest = value as Record<string, unknown>;
  if (manifest.schemaVersion !== 3 || typeof manifest.buildDirectory !== "string") {
    throw new Error("Reference builder did not record a schema-v3 build directory");
  }
  const base = resolve(inferenceRoot, "target/reference/llama.cpp");
  const build = resolve(inferenceRoot, manifest.buildDirectory);
  const child = relative(base, build);
  if (
    child.length === 0 ||
    child === ".." ||
    child.startsWith(`..${sep}`) ||
    isAbsolute(child)
  ) {
    throw new Error("Reference build directory escapes its dedicated target root");
  }
  return build;
};

const cleanupGeneratedTestFiles = async (source: string) => {
  await Promise.all(
    ["test-grammar-output.tmp", "test-json-schema-input.tmp"].map((name) =>
      unlink(resolve(source, name)).catch((error: NodeJS.ErrnoException) => {
        if (error.code !== "ENOENT") throw error;
      })
    )
  );
};

const outputDirectoryFor = (configured?: string) => {
  if (configured !== undefined) {
    return isAbsolute(configured) ? configured : resolve(process.cwd(), configured);
  }
  const timestamp = new Date().toISOString().replaceAll(/[-:.TZ]/g, "");
  const suffix = randomBytes(4).toString("hex");
  return resolve(
    inferenceRoot,
    "results/parity/upstream",
    `${timestamp}-${suffix}`
  );
};

const prepareOutputDirectory = async (path: string, explicit: boolean) => {
  if (explicit) {
    try {
      const entries = await readdir(path);
      if (entries.length > 0) {
        throw new Error(`Evidence output directory is not empty: ${path}`);
      }
      return;
    } catch (error) {
      const code = (error as NodeJS.ErrnoException).code;
      if (code !== "ENOENT") throw error;
    }
  }
  await mkdir(path, { recursive: true });
};

const main = async () => {
  const args = parseArguments();
  if (args.help) {
    console.log(usage);
    return;
  }
  const manifest = await loadTargetManifest(inferenceRoot);
  const selected = selectReferenceTargets(
    manifest,
    args.targets.length === 0 ? ["focused-tests"] : args.targets
  );
  const testNames = uniqueStrings(
    selected.flatMap((target) => [...target.ctest_names])
  );
  const setupNames = uniqueStrings(
    selected.flatMap((target) => [...target.ctest_setup_names])
  );
  const modelFixtures = selected.flatMap((target) => [
    ...target.ctest_model_fixtures,
  ]);
  if (testNames.length === 0) {
    throw new Error("Selected reference targets do not declare any CTests");
  }

  const outputDirectory = outputDirectoryFor(args.outputDirectory);
  const referenceOutput = resolve(outputDirectory, "reference");
  const environment = controlledEnvironment({
    CMAKE_BUILD_PARALLEL_LEVEL: String(args.parallel),
  });
  await prepareOutputDirectory(
    outputDirectory,
    args.outputDirectory !== undefined
  );
  const buildCommand = [
    "bun",
    "run",
    "scripts/build-reference.ts",
    "--backend",
    args.backend,
    "--lane",
    "cargo-equivalent",
    "--parallel",
    String(args.parallel),
    "--output-dir",
    referenceOutput,
    ...selected.flatMap((target) => ["--target", target.id]),
  ];
  const buildExitCode = await run(buildCommand, inferenceRoot, environment);
  if (buildExitCode !== 0) {
    throw new Error(`Reference build exited with status ${buildExitCode}`);
  }

  const pin = await Effect.runPromise(verifyNativePin());
  const source = resolve(inferenceRoot, pin.llama_cpp.checkout_path);
  const build = await referenceBuildDirectory(
    resolve(
      referenceOutput,
      `reference-build-${args.backend}-cargo-equivalent.json`
    )
  );
  const fixturePreparation = await prepareOfflineUpstreamFixtures({
    registryPath: resolve(inferenceRoot, "parity/models/registry.toml"),
    fixtures: modelFixtures,
    buildDirectory: build,
    configuredModelDirectory: args.modelDirectory,
    cwd: process.cwd(),
  });
  const selectionArguments = ctestSelectionArguments({
    buildDirectory: build,
    testNames,
    setupNames,
  });
  const junitPath = resolve(outputDirectory, `upstream-ctest-${args.backend}.xml`);
  const inventoryPath = resolve(
    outputDirectory,
    `upstream-ctest-inventory-${args.backend}.json`
  );
  const testCommand = [
    "ctest",
    ...selectionArguments,
    "--output-on-failure",
    "--no-tests=error",
    "--output-junit",
    junitPath,
  ];
  const ctestInventory = await captureCTestInventory({
    cwd: inferenceRoot,
    buildDirectory: build,
    testNames,
    setupNames,
    environment,
  });
  await writeFile(inventoryPath, `${JSON.stringify(ctestInventory, null, 2)}\n`, {
    flag: "wx",
  });
  const startedAt = new Date().toISOString();
  let testExitCode: number;
  try {
    testExitCode = await run(testCommand, inferenceRoot, environment);
  } finally {
    await Promise.all([
      cleanupGeneratedTestFiles(source),
      verifyOfflineUpstreamFixtures(fixturePreparation),
    ]);
  }
  const completedAt = new Date().toISOString();
  const junitSummary = await verifyCTestJunit(junitPath, testNames);
  const evidenceArtifact = async (path: string) => {
    const file = Bun.file(path);
    const bytes = new Uint8Array(await file.arrayBuffer());
    return {
      path,
      bytes: bytes.byteLength,
      sha256: createHash("sha256").update(bytes).digest("hex"),
    };
  };
  const result = {
    schemaVersion: 3,
    startedAt,
    completedAt,
    backend: args.backend,
    lane: "cargo-equivalent",
    buildDirectory: build,
    environment: controlledEnvironmentEvidence(environment),
    selectedTargets: selected.map((target) => target.id),
    llamaCppRevision: pin.llama_cpp.revision,
    bindingsRevision: pin.llama_cpp_rs.revision,
    tests: testNames,
    setupTests: setupNames,
    suppressedSetupTests: setupNames,
    expectedExecutedTests: testNames.length,
    executedTests: junitSummary.tests,
    ctestInventory: await evidenceArtifact(inventoryPath),
    ctestJunit: await evidenceArtifact(junitPath),
    junitSummary,
    offline: true,
    fixturePreparation,
    artifacts: fixturePreparation.artifacts.map((artifact) => ({
      kind: "model",
      id: artifact.registryId,
      role: artifact.role,
      bytes: artifact.bytes,
      sha256: artifact.sha256,
      sourceCachePath: artifact.sourceCachePath,
      stagedPath: artifact.stagedPath,
      stagingMethod: artifact.stagingMethod,
    })),
    ctestArguments: testCommand.slice(1),
    status:
      testExitCode === 0 &&
      junitSummary.failures === 0 &&
      junitSummary.disabled === 0 &&
      junitSummary.skipped === 0
        ? "passed"
        : "failed",
    exitCode: testExitCode,
  };
  const resultPath = resolve(
    outputDirectory,
    `upstream-tests-${args.backend}.json`
  );
  await writeFile(resultPath, `${JSON.stringify(result, null, 2)}\n`, {
    flag: "wx",
  });
  console.log(`Upstream test evidence: ${resultPath}`);
  if (testExitCode !== 0) {
    throw new Error(`CTest exited with status ${testExitCode}`);
  }
  if (
    junitSummary.failures !== 0 ||
    junitSummary.disabled !== 0 ||
    junitSummary.skipped !== 0
  ) {
    throw new Error(
      "CTest did not execute every declared test successfully; see the preserved JUnit evidence"
    );
  }
};

main().catch((error) => {
  console.error(error instanceof Error ? error.message : error);
  process.exitCode = 1;
});
