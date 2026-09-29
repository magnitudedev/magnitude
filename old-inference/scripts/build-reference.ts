import { Effect } from "effect";
import { randomBytes } from "node:crypto";
import {
  lstat,
  mkdir,
  readFile,
  readdir,
  realpath,
  stat,
  unlink,
  writeFile,
} from "node:fs/promises";
import { isAbsolute, relative, resolve, sep } from "node:path";
import {
  type BuildLane,
  loadBuildProfiles,
  loadTargetManifest,
  selectReferenceTargets,
  uniqueStrings,
} from "./build-parity-config";
import { sourceInventory, type SourceInventory } from "./source-inventory";
import {
  prepareOfflineUpstreamFixtures,
  verifyOfflineUpstreamFixtures,
  type OfflineFixturePreparation,
} from "./upstream-model-fixture";
import {
  captureCTestInventory,
  ctestSelectionArguments,
  verifyCTestJunit,
  type CTestInventory,
  type CTestJunitSummary,
} from "./ctest-evidence";
import { verifyNativePin } from "./verify-native-pin";
import {
  controlledEnvironment,
  controlledEnvironmentEvidence,
} from "./controlled-environment";

type Backend = "cpu" | "metal" | "cuda" | "vulkan";

interface Arguments {
  readonly backend: Backend;
  readonly lane: BuildLane;
  readonly targets: readonly string[];
  readonly parallel: number;
  readonly outputDirectory?: string;
  readonly modelDirectory?: string;
  readonly configureOnly: boolean;
  readonly runTests: boolean;
  readonly dryRun: boolean;
  readonly list: boolean;
  readonly help: boolean;
}

const inferenceRoot = resolve(import.meta.dirname, "..");
const targetManifestPath = resolve(
  inferenceRoot,
  "parity/upstream/targets.toml"
);
const buildProfilesPath = resolve(
  inferenceRoot,
  "parity/upstream/build-profiles.toml"
);
const oracleDirectory = resolve(inferenceRoot, "parity/oracle");
const oracleInjection = resolve(oracleDirectory, "inject.cmake");
const configurationInputs = [
  { name: "targetsManifest", path: targetManifestPath },
  { name: "buildProfiles", path: buildProfilesPath },
  { name: "nativePin", path: resolve(inferenceRoot, "native-pin.toml") },
  {
    name: "referenceBuilder",
    path: resolve(inferenceRoot, "scripts/build-reference.ts"),
  },
  {
    name: "configLoader",
    path: resolve(inferenceRoot, "scripts/build-parity-config.ts"),
  },
  {
    name: "sourceInventory",
    path: resolve(inferenceRoot, "scripts/source-inventory.ts"),
  },
  {
    name: "ctestEvidence",
    path: resolve(inferenceRoot, "scripts/ctest-evidence.ts"),
  },
  {
    name: "offlineFixtureStaging",
    path: resolve(inferenceRoot, "scripts/upstream-model-fixture.ts"),
  },
  {
    name: "controlledEnvironment",
    path: resolve(inferenceRoot, "scripts/controlled-environment.ts"),
  },
  {
    name: "nativePinVerifier",
    path: resolve(inferenceRoot, "scripts/verify-native-pin.ts"),
  },
] as const;

const usage = `Build pinned upstream tests, tools, and the primitive oracle.

Usage:
  bun run scripts/build-reference.ts [options]

Options:
  --backend <cpu|metal|cuda|vulkan>  Backend to compile (default: metal on macOS, cpu elsewhere)
  --lane <upstream-default|cargo-equivalent>
                                      Native build profile (default: upstream-default)
  --target <id|all>                 Select a manifest target; may be repeated
  --run-tests                       Run selected CTests with offline fixture staging
  --model-dir <path>                Registry artifact root (default: ICN_PARITY_MODEL_DIR)
  --parallel <count>                Parallel CMake build jobs (default: 8)
  --output-dir <path>               Empty evidence directory (default: unique results/parity/reference/<run>)
  --configure-only                  Configure CMake without compiling targets
  --dry-run                         Verify configuration and print commands only
  --list                            List target-manifest entries
  --help                            Show this help
`;

const parseBackend = (value: string): Backend => {
  if (
    value === "cpu" ||
    value === "metal" ||
    value === "cuda" ||
    value === "vulkan"
  ) {
    return value;
  }
  throw new Error(`Unsupported backend: ${value}`);
};

const parseLane = (value: string): BuildLane => {
  if (value === "upstream-default" || value === "cargo-equivalent") {
    return value;
  }
  throw new Error(`Unsupported build lane: ${value}`);
};

const requireNext = (args: readonly string[], index: number, option: string) => {
  const value = args[index + 1];
  if (value === undefined) throw new Error(`${option} requires a value`);
  return value;
};

const parseArguments = (): Arguments => {
  let backend: Backend = process.platform === "darwin" ? "metal" : "cpu";
  let lane: BuildLane = "upstream-default";
  const targets: string[] = [];
  let parallel = 8;
  let outputDirectory: string | undefined;
  let modelDirectory: string | undefined;
  let configureOnly = false;
  let runTests = false;
  let dryRun = false;
  let list = false;
  let help = false;
  const args = Bun.argv.slice(2);

  for (let index = 0; index < args.length; index += 1) {
    const argument = args[index];
    if (argument === "--backend") {
      backend = parseBackend(requireNext(args, index, argument));
      index += 1;
    } else if (argument === "--lane") {
      lane = parseLane(requireNext(args, index, argument));
      index += 1;
    } else if (argument === "--target") {
      targets.push(requireNext(args, index, argument));
      index += 1;
    } else if (argument === "--parallel") {
      const raw = requireNext(args, index, argument);
      parallel = Number(raw);
      if (!Number.isSafeInteger(parallel) || parallel < 1) {
        throw new Error(`Invalid parallel job count: ${raw}`);
      }
      index += 1;
    } else if (argument === "--output-dir") {
      outputDirectory = requireNext(args, index, argument);
      index += 1;
    } else if (argument === "--model-dir") {
      modelDirectory = requireNext(args, index, argument);
      index += 1;
    } else if (argument === "--configure-only") {
      configureOnly = true;
    } else if (argument === "--run-tests") {
      runTests = true;
    } else if (argument === "--dry-run") {
      dryRun = true;
    } else if (argument === "--list") {
      list = true;
    } else if (argument === "--help") {
      help = true;
    } else {
      throw new Error(`Unsupported argument: ${argument}`);
    }
  }
  if (configureOnly && runTests) {
    throw new Error("--configure-only and --run-tests cannot be combined");
  }
  return {
    backend,
    lane,
    targets,
    parallel,
    outputDirectory,
    modelDirectory,
    configureOnly,
    runTests,
    dryRun,
    list,
    help,
  };
};

const cmakeDefinitions = (definitions: Readonly<Record<string, string>>) =>
  Object.entries(definitions).map(([name, value]) => `-D${name}=${value}`);

const backendDefinitions = (backend: Backend): readonly string[] => [
  `-DGGML_METAL=${backend === "metal" ? "ON" : "OFF"}`,
  `-DGGML_CUDA=${backend === "cuda" ? "ON" : "OFF"}`,
  `-DGGML_VULKAN=${backend === "vulkan" ? "ON" : "OFF"}`,
];

const formatCommand = (command: readonly string[]) =>
  command.map((part) => JSON.stringify(part)).join(" ");

const runCommandStatus = async (
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

const runCommand = async (
  command: readonly string[],
  cwd: string,
  environment: Readonly<Record<string, string>>
) => {
  const exitCode = await runCommandStatus(command, cwd, environment);
  if (exitCode !== 0) {
    throw new Error(`${command[0]} exited with status ${exitCode}`);
  }
};

const captureCommand = async (
  command: readonly string[],
  cwd: string,
  environment: Readonly<Record<string, string>>
) => {
  const child = Bun.spawn([...command], {
    cwd,
    env: environment,
    stdout: "pipe",
    stderr: "pipe",
  });
  const [exitCode, stdout, stderr] = await Promise.all([
    child.exited,
    new Response(child.stdout).text(),
    new Response(child.stderr).text(),
  ]);
  if (exitCode !== 0) {
    throw new Error(
      stderr.trim() || `${command[0]} exited with status ${exitCode}`
    );
  }
  return stdout.trim();
};

export const cleanupGeneratedTestFiles = async (source: string) => {
  await Promise.all(
    ["test-grammar-output.tmp", "test-json-schema-input.tmp"].map((name) =>
      unlink(resolve(source, name)).catch((error: NodeJS.ErrnoException) => {
        if (error.code !== "ENOENT") throw error;
      })
    )
  );
};

export const inventoryUpstreamSource = async (source: string) => {
  await cleanupGeneratedTestFiles(source);
  return Effect.runPromise(sourceInventory(source));
};

export const assertInventoryUnchanged = (
  before: Pick<SourceInventory, "sha256" | "fileCount" | "totalBytes">,
  after: Pick<SourceInventory, "sha256" | "fileCount" | "totalBytes">,
  description: string
) => {
  if (
    before.sha256 !== after.sha256 ||
    before.fileCount !== after.fileCount ||
    before.totalBytes !== after.totalBytes
  ) {
    throw new Error(`${description} changed during the reference build`);
  }
};

export const oracleCmakeArguments = (
  oracleSelected: boolean
): readonly string[] =>
  oracleSelected
    ? [
        "-DCMAKE_PROJECT_TOP_LEVEL_INCLUDES=",
        `-DCMAKE_PROJECT_INCLUDE=${oracleInjection}`,
        `-DICN_PARITY_ORACLE_SOURCE_DIR=${oracleDirectory}`,
      ]
    : [];

export const findArtifact = async (buildDirectory: string, name: string) => {
  const executable = process.platform === "win32" ? `${name}.exe` : name;
  const candidates = [
    resolve(buildDirectory, "bin", executable),
    resolve(buildDirectory, "bin", "Release", executable),
    resolve(buildDirectory, "parity-oracle", executable),
    resolve(buildDirectory, executable),
    resolve(buildDirectory, "Release", executable),
  ];
  const canonicalBuildDirectory = await realpath(buildDirectory);
  for (const path of candidates) {
    try {
      await lstat(path);
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code === "ENOENT") continue;
      throw error;
    }
    const canonicalPath = await realpath(path);
    const child = relative(canonicalBuildDirectory, canonicalPath);
    if (
      child.length === 0 ||
      child === ".." ||
      child.startsWith(`..${sep}`) ||
      isAbsolute(child)
    ) {
      throw new Error(`Built artifact escapes its build directory: ${path}`);
    }
    if (!(await stat(canonicalPath)).isFile()) {
      throw new Error(`Built artifact is not a regular file: ${path}`);
    }
    return canonicalPath;
  }
  throw new Error(`Unable to find built ${name} under ${buildDirectory}`);
};

const artifactRecordFromBytes = (path: string, bytes: Uint8Array) => {
  const hasher = new Bun.CryptoHasher("sha256");
  hasher.update(bytes);
  return {
    path: relative(inferenceRoot, path),
    bytes: bytes.byteLength,
    sha256: hasher.digest("hex"),
  };
};

const artifactRecord = async (path: string) =>
  artifactRecordFromBytes(path, await readFile(path));

const snapshotConfigurationInputs = async () =>
  Promise.all(
    configurationInputs.map(async (input) => {
      const bytes = await readFile(input.path);
      return {
        ...input,
        text: new TextDecoder().decode(bytes),
        record: artifactRecordFromBytes(input.path, bytes),
      };
    })
  );

const uniqueRunName = () => {
  const timestamp = new Date().toISOString().replaceAll(/[-:.TZ]/g, "");
  return `${timestamp}-${randomBytes(4).toString("hex")}`;
};

const resolveOutputDirectory = (configured?: string) => {
  if (configured !== undefined) {
    return isAbsolute(configured) ? configured : resolve(process.cwd(), configured);
  }
  return resolve(
    inferenceRoot,
    "results/parity/reference",
    uniqueRunName()
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

const reserveFreshBuildDirectory = async (path: string) => {
  await mkdir(resolve(path, ".."), { recursive: true });
  await mkdir(path, { recursive: false });
};

const readCmakeCache = async (buildDirectory: string) => {
  const cache = await Bun.file(resolve(buildDirectory, "CMakeCache.txt")).text();
  const values: Record<string, string> = {};
  for (const line of cache.split("\n")) {
    if (line.startsWith("//") || line.startsWith("#")) continue;
    const match = /^([^:=]+):[^=]*=(.*)$/.exec(line);
    if (match?.[1] !== undefined && match[2] !== undefined) {
      values[match[1]] = match[2];
    }
  }
  return values;
};

const compilerIdentity = async (
  buildDirectory: string,
  cache: Readonly<Record<string, string>>,
  language: "C" | "CXX",
  environment: Readonly<Record<string, string>>
) => {
  const glob = new Bun.Glob(`CMakeFiles/*/CMake${language}Compiler.cmake`);
  const files: string[] = [];
  for await (const path of glob.scan({ cwd: buildDirectory, onlyFiles: true })) {
    files.push(path);
  }
  files.sort();
  const metadataPath = files.at(-1);
  const metadata =
    metadataPath === undefined
      ? ""
      : await Bun.file(resolve(buildDirectory, metadataPath)).text();
  const compiler = cache[`CMAKE_${language}_COMPILER`];
  if (compiler === undefined || compiler.length === 0) {
    throw new Error(`CMake did not record CMAKE_${language}_COMPILER`);
  }
  const id = new RegExp(`set\\(CMAKE_${language}_COMPILER_ID "([^"]*)"\\)`).exec(
    metadata
  )?.[1];
  const version = new RegExp(
    `set\\(CMAKE_${language}_COMPILER_VERSION "([^"]*)"\\)`
  ).exec(metadata)?.[1];
  return {
    path: compiler,
    id: id ?? "unknown",
    version: version ?? "unknown",
    banner: (
      await captureCommand([compiler, "--version"], inferenceRoot, environment)
    ).split("\n")[0],
    metadataPath:
      metadataPath === undefined
        ? null
        : relative(inferenceRoot, resolve(buildDirectory, metadataPath)),
  };
};

const relevantFlagNames = [
  "CMAKE_C_FLAGS",
  "CMAKE_C_FLAGS_RELEASE",
  "CMAKE_CXX_FLAGS",
  "CMAKE_CXX_FLAGS_RELEASE",
  "CMAKE_CUDA_FLAGS",
  "CMAKE_CUDA_FLAGS_RELEASE",
  "CMAKE_EXE_LINKER_FLAGS",
  "CMAKE_EXE_LINKER_FLAGS_RELEASE",
  "CMAKE_SHARED_LINKER_FLAGS",
  "CMAKE_SHARED_LINKER_FLAGS_RELEASE",
] as const;

const sanitizerNames = (documents: readonly string[]) => {
  const names = new Set<string>();
  for (const document of documents) {
    for (const match of document.matchAll(/(?:-f|\/)sanitize=([A-Za-z0-9_,+-]+)/g)) {
      for (const name of match[1]!.split(",")) {
        if (name.length > 0) names.add(name);
      }
    }
  }
  return [...names].sort();
};

export const buildVerificationEvidence = async (
  buildDirectory: string,
  cache: Readonly<Record<string, string>>
) => {
  const compileCommandsPath = resolve(buildDirectory, "compile_commands.json");
  const compileBytes = await readFile(compileCommandsPath);
  const rawCommands: unknown = JSON.parse(new TextDecoder().decode(compileBytes));
  if (!Array.isArray(rawCommands) || rawCommands.length === 0) {
    throw new Error("CMake compile_commands.json is empty or invalid");
  }
  const compileCommands = rawCommands.map((value, index) => {
    if (value === null || typeof value !== "object" || Array.isArray(value)) {
      throw new Error(`compile_commands.json entry ${index} is invalid`);
    }
    const entry = value as Record<string, unknown>;
    if (
      Array.isArray(entry.arguments) &&
      entry.arguments.length > 0 &&
      entry.arguments.every((argument) => typeof argument === "string")
    ) {
      return (entry.arguments as string[]).join(" ");
    }
    if (typeof entry.command === "string" && entry.command.length > 0) {
      return entry.command;
    }
    throw new Error(`compile_commands.json entry ${index} has no command`);
  });
  const linkPaths: string[] = [];
  const glob = new Bun.Glob("**/link.txt");
  for await (const path of glob.scan({ cwd: buildDirectory, onlyFiles: true })) {
    linkPaths.push(path);
  }
  linkPaths.sort();
  const linkCommands = await Promise.all(
    linkPaths.map(async (path) => ({
      path,
      text: await Bun.file(resolve(buildDirectory, path)).text(),
    }))
  );
  const cachedFlags = relevantFlagNames.map((name) => cache[name] ?? "");
  const ndebug = /(?:^|\s)(?:-D|\/D)NDEBUG(?:=[^\s]+)?(?:\s|$)/;
  const undefNdebug = /(?:^|\s)(?:-U|\/U)NDEBUG(?:\s|$)/;
  const assertions = compileCommands.every(
    (command) => ndebug.test(command) && !undefNdebug.test(command)
  )
    ? false
    : null;
  const sanitizers =
    linkCommands.length === 0
      ? null
      : sanitizerNames([
          ...compileCommands,
          ...linkCommands.map((command) => command.text),
          ...cachedFlags,
        ]);
  const linkHasher = new Bun.CryptoHasher("sha256");
  for (const command of linkCommands) {
    linkHasher.update(command.path);
    linkHasher.update("\0");
    linkHasher.update(command.text);
    linkHasher.update("\0");
  }
  return {
    assertions,
    sanitizers,
    evidence: {
      compileCommands: artifactRecordFromBytes(
        compileCommandsPath,
        compileBytes
      ),
      compileCommandCount: compileCommands.length,
      linkCommandCount: linkCommands.length,
      linkCommandsSha256: linkHasher.digest("hex"),
      inspectedCacheFlags: Object.fromEntries(
        relevantFlagNames.map((name) => [name, cache[name] ?? ""])
      ),
    },
  };
};

const main = async () => {
  const args = parseArguments();
  if (args.help) {
    console.log(usage);
    return;
  }

  const configurationBefore = await snapshotConfigurationInputs();
  const configurationSource = (name: (typeof configurationInputs)[number]["name"]) => {
    const input = configurationBefore.find((candidate) => candidate.name === name);
    if (input === undefined) throw new Error(`Missing configuration snapshot: ${name}`);
    return input.text;
  };
  const [targetManifest, buildProfiles] = await Promise.all([
    loadTargetManifest(inferenceRoot, configurationSource("targetsManifest")),
    loadBuildProfiles(inferenceRoot, configurationSource("buildProfiles")),
  ]);
  if (args.list) {
    for (const target of targetManifest.targets) {
      const marker = targetManifest.default_targets.includes(target.id)
        ? "default"
        : "explicit";
      console.log(`${target.id}\t${target.kind}\t${marker}\t${target.description}`);
    }
    return;
  }

  const selected = selectReferenceTargets(targetManifest, args.targets);
  const oracleSelected = selected.some((target) => target.id === "oracle");
  const profile = buildProfiles.profiles[args.lane];
  const pin = await Effect.runPromise(verifyNativePin());
  const sourceDirectory = resolve(inferenceRoot, pin.llama_cpp.checkout_path);
  const resultSuffix =
    args.lane === "upstream-default"
      ? args.backend
      : `${args.backend}-${args.lane}`;
  const buildDirectory = resolve(
    inferenceRoot,
    "target/reference/llama.cpp",
    `${resultSuffix}-${oracleSelected ? "oracle" : "upstream"}`,
    uniqueRunName()
  );
  const outputDirectory = resolveOutputDirectory(args.outputDirectory);
  const requiresTests = selected.some((target) => target.requires_tests);
  const requiresTools = selected.some((target) => target.requires_tools);
  const requiresServer = selected.some((target) => target.requires_server);
  const selectedIds = selected.map((target) => target.id);
  const buildTargets = uniqueStrings(
    selected.flatMap((target) => [...target.cmake_targets])
  );
  const artifactNames = uniqueStrings(
    selected.flatMap((target) => [...target.artifacts])
  );
  const testNames = uniqueStrings(
    selected.flatMap((target) => [...target.ctest_names])
  );
  const setupTestNames = uniqueStrings(
    selected.flatMap((target) => [...target.ctest_setup_names])
  );
  const modelFixtures = selected.flatMap((target) => [
    ...target.ctest_model_fixtures,
  ]);
  const laneDefinitions = {
    ...profile.definitions,
    ...(process.platform === "darwin" ? profile.darwin_definitions : {}),
    ...(args.backend === "metal" ? profile.metal_definitions : {}),
  };
  const configureCommand = [
    "cmake",
    "-S",
    sourceDirectory,
    "-B",
    buildDirectory,
    "-DCMAKE_BUILD_TYPE=Release",
    "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
    "-DBUILD_SHARED_LIBS=OFF",
    "-DGGML_NATIVE=OFF",
    "-DLLAMA_BUILD_APP=OFF",
    "-DLLAMA_BUILD_COMMON=ON",
    "-DLLAMA_BUILD_EXAMPLES=OFF",
    `-DLLAMA_BUILD_SERVER=${requiresServer ? "ON" : "OFF"}`,
    `-DLLAMA_BUILD_TESTS=${requiresTests ? "ON" : "OFF"}`,
    `-DLLAMA_BUILD_TOOLS=${requiresTools ? "ON" : "OFF"}`,
    "-DLLAMA_BUILD_UI=OFF",
    "-DLLAMA_USE_PREBUILT_UI=OFF",
    "-DLLAMA_CURL=OFF",
    ...oracleCmakeArguments(oracleSelected),
    ...backendDefinitions(args.backend),
    ...cmakeDefinitions(laneDefinitions),
    ...(process.platform === "darwin" && profile.osx_architecture_from_host
      ? [
          `-DCMAKE_OSX_ARCHITECTURES=${
            process.arch === "arm64" ? "arm64" : "x86_64"
          }`,
        ]
      : []),
  ];
  const buildCommand = [
    "cmake",
    "--build",
    buildDirectory,
    "--config",
    "Release",
    "--target",
    ...buildTargets,
    "--parallel",
    String(args.parallel),
  ];
  const ctestSelection =
    testNames.length === 0
      ? []
      : ctestSelectionArguments({
          buildDirectory,
          testNames,
          setupNames: setupTestNames,
        });
  const ctestJunitPath = resolve(
    outputDirectory,
    `reference-ctest-${resultSuffix}.xml`
  );
  const testCommand = [
    "ctest",
    ...ctestSelection,
    "--output-on-failure",
    "--no-tests=error",
    "--output-junit",
    ctestJunitPath,
  ];
  const buildEnvironment = controlledEnvironment({
    CMAKE_BUILD_PARALLEL_LEVEL: String(args.parallel),
  });

  console.log(`Selected reference targets: ${selectedIds.join(", ")}`);
  console.log(formatCommand(configureCommand));
  if (!args.configureOnly) console.log(formatCommand(buildCommand));
  if (args.runTests && testNames.length > 0) console.log(formatCommand(testCommand));
  if (args.runTests && testNames.length === 0) {
    throw new Error("Selected targets do not declare any CTests");
  }
  if (args.dryRun) return;

  await prepareOutputDirectory(
    outputDirectory,
    args.outputDirectory !== undefined
  );
  await reserveFreshBuildDirectory(buildDirectory);

  const [nativeSource, oracleSource] = await Promise.all([
    inventoryUpstreamSource(sourceDirectory),
    oracleSelected
      ? Effect.runPromise(sourceInventory(oracleDirectory))
      : Promise.resolve(null),
  ]);
  await runCommand(configureCommand, inferenceRoot, buildEnvironment);
  if (!args.configureOnly) {
    await runCommand(buildCommand, inferenceRoot, buildEnvironment);
  }
  let fixturePreparation: OfflineFixturePreparation = {
    registryPath: resolve(inferenceRoot, "parity/models/registry.toml"),
    modelDirectory: null,
    modelDirectorySource: null,
    artifacts: [],
  };
  let testsStartedAt: string | undefined;
  let testsCompletedAt: string | undefined;
  let ctestExitCode: number | undefined;
  let ctestInventory: CTestInventory | undefined;
  let ctestJunit: CTestJunitSummary | undefined;
  const ctestInventoryPath = resolve(
    outputDirectory,
    `reference-ctest-inventory-${resultSuffix}.json`
  );
  if (args.runTests) {
    fixturePreparation = await prepareOfflineUpstreamFixtures({
      registryPath: resolve(inferenceRoot, "parity/models/registry.toml"),
      fixtures: modelFixtures,
      buildDirectory,
      configuredModelDirectory: args.modelDirectory,
      cwd: process.cwd(),
    });
    ctestInventory = await captureCTestInventory({
      cwd: inferenceRoot,
      buildDirectory,
      testNames,
      setupNames: setupTestNames,
      environment: buildEnvironment,
    });
    await writeFile(
      ctestInventoryPath,
      `${JSON.stringify(ctestInventory, null, 2)}\n`,
      { flag: "wx" }
    );
    testsStartedAt = new Date().toISOString();
    try {
      ctestExitCode = await runCommandStatus(
        testCommand,
        inferenceRoot,
        buildEnvironment
      );
    } finally {
      await Promise.all([
        cleanupGeneratedTestFiles(sourceDirectory),
        verifyOfflineUpstreamFixtures(fixturePreparation),
      ]);
    }
    testsCompletedAt = new Date().toISOString();
    ctestJunit = await verifyCTestJunit(ctestJunitPath, testNames);
  }

  const inventoryPath = resolve(
    outputDirectory,
    `reference-build-inputs-${resultSuffix}.json`
  );
  await writeFile(
    inventoryPath,
    `${JSON.stringify(nativeSource, null, 2)}\n`,
    { flag: "wx" }
  );
  const binaryArtifacts = args.configureOnly
    ? []
    : await Promise.all(
        artifactNames.map(async (name) => ({
          name,
          ...(await artifactRecord(await findArtifact(buildDirectory, name))),
        }))
      );
  const cmakeVersion = await captureCommand(
    ["cmake", "--version"],
    inferenceRoot,
    buildEnvironment
  );
  const cmakeCache = await readCmakeCache(buildDirectory);
  const resolvedCmakeCache = Object.fromEntries(
    Object.entries(cmakeCache)
      .filter(([name]) =>
        name === "BUILD_SHARED_LIBS" ||
        name === "CMAKE_BUILD_TYPE" ||
        name === "CMAKE_GENERATOR" ||
        name === "CMAKE_OSX_ARCHITECTURES" ||
        name === "CMAKE_PROJECT_INCLUDE" ||
        name === "CMAKE_PROJECT_TOP_LEVEL_INCLUDES" ||
        relevantFlagNames.includes(name as (typeof relevantFlagNames)[number]) ||
        name === "ICN_PARITY_ORACLE_SOURCE_DIR" ||
        name.startsWith("GGML_") ||
        name.startsWith("LLAMA_")
      )
      .sort(([left], [right]) => left.localeCompare(right))
  );
  const compilers = {
    c: await compilerIdentity(
      buildDirectory,
      cmakeCache,
      "C",
      buildEnvironment
    ),
    cxx: await compilerIdentity(
      buildDirectory,
      cmakeCache,
      "CXX",
      buildEnvironment
    ),
  };
  const verification = await buildVerificationEvidence(
    buildDirectory,
    cmakeCache
  );
  const manifest = {
    schemaVersion: 3,
    createdAt: new Date().toISOString(),
    host: { platform: process.platform, architecture: process.arch },
    backend: args.backend,
    lane: args.lane,
    profileDescription: profile.description,
    buildType: "Release",
    buildDirectory: relative(inferenceRoot, buildDirectory),
    selectedTargets: selected.map((target) => ({
      id: target.id,
      kind: target.kind,
      description: target.description,
      requiresTests: target.requires_tests,
      requiresTools: target.requires_tools,
      requiresServer: target.requires_server,
      cmakeTargets: target.cmake_targets,
      ctestNames: target.ctest_names,
      ctestSetupNames: target.ctest_setup_names,
      ctestModelFixtures: target.ctest_model_fixtures,
    })),
    bindings: {
      path: pin.llama_cpp_rs.checkout_path,
      revision: pin.llama_cpp_rs.revision,
    },
    llamaCpp: {
      path: pin.llama_cpp.checkout_path,
      revision: pin.llama_cpp.revision,
      sourceTree: {
        algorithm: nativeSource.algorithm,
        sha256: nativeSource.sha256,
        fileCount: nativeSource.fileCount,
        totalBytes: nativeSource.totalBytes,
        excludedDirectoryNames: nativeSource.excludedDirectoryNames,
        inventory: await artifactRecord(inventoryPath),
      },
    },
    oracle:
      oracleSource === null
        ? null
        : {
            sourceTree: {
              algorithm: oracleSource.algorithm,
              sha256: oracleSource.sha256,
              fileCount: oracleSource.fileCount,
              totalBytes: oracleSource.totalBytes,
              excludedDirectoryNames: oracleSource.excludedDirectoryNames,
            },
          },
    configuration: Object.fromEntries(
      configurationBefore.map((input) => [input.name, input.record])
    ),
    environment: controlledEnvironmentEvidence(buildEnvironment),
    cmake: {
      version: cmakeVersion.split("\n")[0] ?? cmakeVersion,
      generator: cmakeCache.CMAKE_GENERATOR ?? "unknown",
      configureArguments: configureCommand.slice(1),
      buildArguments: args.configureOnly ? null : buildCommand.slice(1),
      resolvedCache: resolvedCmakeCache,
    },
    compilers,
    verification,
    ctest: args.runTests
      ? {
          startedAt: testsStartedAt,
          completedAt: testsCompletedAt,
          status:
            ctestExitCode === 0 &&
            ctestJunit?.failures === 0 &&
            ctestJunit.disabled === 0 &&
            ctestJunit.skipped === 0
              ? "passed"
              : "failed",
          exitCode: ctestExitCode,
          names: testNames,
          setupNames: setupTestNames,
          suppressedSetupNames: setupTestNames,
          expectedExecutedTests: testNames.length,
          executedTests: ctestJunit?.tests,
          inventory: await artifactRecord(ctestInventoryPath),
          junit: await artifactRecord(ctestJunitPath),
          junitSummary: ctestJunit,
          offline: true,
          fixturePreparation,
          arguments: testCommand.slice(1),
        }
      : {
          status: "not-run",
          names: testNames,
          setupNames: setupTestNames,
        },
    fixtureArtifacts: fixturePreparation.artifacts,
    artifacts: binaryArtifacts,
  };
  const manifestPath = resolve(
    outputDirectory,
    `reference-build-${resultSuffix}.json`
  );
  const [nativeSourceAfter, oracleSourceAfter, configurationAfter] =
    await Promise.all([
      inventoryUpstreamSource(sourceDirectory),
      oracleSelected
        ? Effect.runPromise(sourceInventory(oracleDirectory))
        : Promise.resolve(null),
      Promise.all(configurationInputs.map((input) => artifactRecord(input.path))),
    ]);
  assertInventoryUnchanged(
    nativeSource,
    nativeSourceAfter,
    "Pinned llama.cpp source"
  );
  if (oracleSource !== null && oracleSourceAfter !== null) {
    assertInventoryUnchanged(
      oracleSource,
      oracleSourceAfter,
      "Parity oracle source"
    );
  }
  for (let index = 0; index < configurationBefore.length; index += 1) {
    const before = configurationBefore[index]!.record;
    const after = configurationAfter[index]!;
    if (before.sha256 !== after.sha256 || before.bytes !== after.bytes) {
      throw new Error(
        `Reference build input changed during execution: ${before.path}`
      );
    }
  }
  await writeFile(manifestPath, `${JSON.stringify(manifest, null, 2)}\n`, {
    flag: "wx",
  });
  console.log(`Reference build manifest: ${manifestPath}`);
  if (args.runTests && ctestExitCode !== 0) {
    throw new Error(`CTest exited with status ${ctestExitCode}`);
  }
  if (
    args.runTests &&
    ctestJunit !== undefined &&
    (ctestJunit.failures !== 0 ||
      ctestJunit.disabled !== 0 ||
      ctestJunit.skipped !== 0)
  ) {
    throw new Error(
      "CTest did not execute every declared test successfully; see the preserved JUnit evidence"
    );
  }
};

if (import.meta.main) {
  void main().catch((error) => {
    console.error(error instanceof Error ? error.message : error);
    process.exitCode = 1;
  });
}
