import { Effect } from "effect";
import { randomBytes } from "node:crypto";
import { constants as fileConstants } from "node:fs";
import {
  chmod,
  copyFile,
  mkdir,
  readFile,
  readdir,
  realpath,
  stat,
  writeFile,
} from "node:fs/promises";
import {
  isAbsolute,
  relative,
  resolve,
  sep,
} from "node:path";
import {
  defaultExcludedDirectoryNames,
  sourceInventory,
  type SourceInventory,
} from "./source-inventory";
import {
  controlledEnvironment,
  controlledEnvironmentEvidence,
} from "./controlled-environment";

export type CandidateBackend = "cpu" | "metal" | "cuda" | "vulkan";

interface Arguments {
  readonly referenceManifest?: string;
  readonly backend?: CandidateBackend;
  readonly parallel: number;
  readonly outputDirectory?: string;
  readonly dryRun: boolean;
  readonly help: boolean;
}

export interface ReferenceIdentity {
  readonly path: string;
  readonly sha256: string;
  readonly backend: CandidateBackend;
  readonly lane: "cargo-equivalent";
  readonly buildType: "Release";
  readonly selectedTargets: ReadonlyArray<string>;
  readonly native: {
    readonly path: string;
    readonly revision: string;
    readonly treeSha256: string;
    readonly excludedDirectoryNames: ReadonlyArray<string>;
  };
}

interface NativePin {
  readonly llama_cpp_rs: {
    readonly checkout_path: string;
    readonly revision: string;
  };
  readonly llama_cpp: {
    readonly checkout_path: string;
    readonly revision: string;
  };
}

interface SourceComponent {
  readonly kind: "native-source" | "bindings-source" | "icn-source";
  readonly name: string;
  readonly path: string;
  readonly revision?: string;
  readonly dirty: true;
  readonly excludedDirectoryNames: ReadonlyArray<string>;
}

interface InventoriedComponent extends SourceComponent {
  readonly inventory: SourceInventory;
}

const inferenceRoot = resolve(import.meta.dirname, "..");
export const candidatePackageName = "icn-parity-probe";
export const candidateArtifactName = "icn-probe";
const sha256Pattern = /^[0-9a-f]{64}$/;
const revisionPattern = /^[0-9a-f]{40}$/;
const candidateConfigurationInputs = [
  { name: "nativePin", path: resolve(inferenceRoot, "native-pin.toml") },
  { name: "cargoManifest", path: resolve(inferenceRoot, "Cargo.toml") },
  { name: "cargoLock", path: resolve(inferenceRoot, "Cargo.lock") },
  {
    name: "candidateBuilder",
    path: resolve(inferenceRoot, "scripts/build-candidate.ts"),
  },
  {
    name: "controlledEnvironment",
    path: resolve(inferenceRoot, "scripts/controlled-environment.ts"),
  },
  {
    name: "sourceInventory",
    path: resolve(inferenceRoot, "scripts/source-inventory.ts"),
  },
] as const;

const usage = `Build the production-owned ICN parity probe with immutable provenance.

This builds the icn-parity-probe package and preserves its icn-probe binary. A
reference manifest from the cargo-equivalent lane is required so the native
source, backend, and exact reference-build identity cannot drift.

Usage:
  bun run scripts/build-candidate.ts --reference-manifest <path> [options]

Options:
  --reference-manifest <path>       Schema-v3 reference build manifest (required)
  --backend <cpu|metal|cuda|vulkan> Assert the reference backend
  --parallel <count>                Native build parallelism (default: 8)
  --output-dir <path>               Empty directory inside inference/
  --dry-run                         Validate identity and print the Cargo command
  --help                            Show this help
`;

const requireString = (value: unknown, description: string): string => {
  if (typeof value !== "string" || value.trim().length === 0) {
    throw new Error(`${description} must be a non-empty string`);
  }
  return value;
};

const requireRecord = (
  value: unknown,
  description: string
): Record<string, unknown> => {
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    throw new Error(`${description} must be an object`);
  }
  return value as Record<string, unknown>;
};

const parseBackend = (value: string): CandidateBackend => {
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

const requireNext = (
  arguments_: readonly string[],
  index: number,
  option: string
) => {
  const value = arguments_[index + 1];
  if (value === undefined) throw new Error(`${option} requires a value`);
  return value;
};

export const parseArguments = (arguments_: readonly string[]): Arguments => {
  let referenceManifest: string | undefined;
  let backend: CandidateBackend | undefined;
  let parallel = 8;
  let outputDirectory: string | undefined;
  let dryRun = false;
  let help = false;
  for (let index = 0; index < arguments_.length; index += 1) {
    const argument = arguments_[index];
    if (argument === "--reference-manifest") {
      referenceManifest = requireNext(arguments_, index, argument);
      index += 1;
    } else if (argument === "--backend") {
      backend = parseBackend(requireNext(arguments_, index, argument));
      index += 1;
    } else if (argument === "--parallel") {
      const raw = requireNext(arguments_, index, argument);
      parallel = Number(raw);
      if (!Number.isSafeInteger(parallel) || parallel < 1) {
        throw new Error(`Invalid parallel job count: ${raw}`);
      }
      index += 1;
    } else if (argument === "--output-dir") {
      outputDirectory = requireNext(arguments_, index, argument);
      index += 1;
    } else if (argument === "--dry-run") {
      dryRun = true;
    } else if (argument === "--help") {
      help = true;
    } else {
      throw new Error(`Unsupported argument: ${argument}`);
    }
  }
  if (!help && referenceManifest === undefined) {
    throw new Error("--reference-manifest is required");
  }
  return {
    referenceManifest,
    backend,
    parallel,
    outputDirectory,
    dryRun,
    help,
  };
};

const sha256Bytes = (bytes: Uint8Array) => {
  const hasher = new Bun.CryptoHasher("sha256");
  hasher.update(bytes);
  return hasher.digest("hex");
};

const readNativePin = async (exactSource?: string): Promise<NativePin> => {
  const value = Bun.TOML.parse(
    exactSource ?? await Bun.file(resolve(inferenceRoot, "native-pin.toml")).text()
  ) as Record<string, unknown>;
  const bindings = requireRecord(value.llama_cpp_rs, "llama_cpp_rs pin");
  const native = requireRecord(value.llama_cpp, "llama_cpp pin");
  const pin: NativePin = {
    llama_cpp_rs: {
      checkout_path: requireString(
        bindings.checkout_path,
        "llama_cpp_rs.checkout_path"
      ),
      revision: requireString(bindings.revision, "llama_cpp_rs.revision"),
    },
    llama_cpp: {
      checkout_path: requireString(
        native.checkout_path,
        "llama_cpp.checkout_path"
      ),
      revision: requireString(native.revision, "llama_cpp.revision"),
    },
  };
  if (
    !revisionPattern.test(pin.llama_cpp.revision) ||
    !revisionPattern.test(pin.llama_cpp_rs.revision)
  ) {
    throw new Error("native-pin.toml contains an invalid revision");
  }
  return pin;
};

const parseSelectedTargets = (value: unknown): ReadonlyArray<string> => {
  if (!Array.isArray(value) || value.length === 0) {
    throw new Error("reference manifest selectedTargets must be non-empty");
  }
  return value.map((entry, index) => {
    if (typeof entry === "string") return requireString(entry, `selectedTargets[${index}]`);
    return requireString(
      requireRecord(entry, `selectedTargets[${index}]`).id,
      `selectedTargets[${index}].id`
    );
  });
};

export const loadReferenceIdentity = async (
  path: string
): Promise<ReferenceIdentity> => {
  const absolutePath = isAbsolute(path) ? path : resolve(process.cwd(), path);
  const bytes = await readFile(absolutePath);
  const value = requireRecord(
    JSON.parse(new TextDecoder().decode(bytes)),
    "reference manifest"
  );
  if (value.schemaVersion !== 3) {
    throw new Error("candidate builds require a schema-v3 reference manifest");
  }
  const backend = parseBackend(
    requireString(value.backend, "reference backend")
  );
  if (value.lane !== "cargo-equivalent") {
    throw new Error(
      "candidate builds require a cargo-equivalent reference build lane"
    );
  }
  if (value.buildType !== "Release") {
    throw new Error("candidate and reference builds must both use Release");
  }
  const selectedTargets = parseSelectedTargets(value.selectedTargets);
  const llamaCpp = requireRecord(value.llamaCpp, "reference llamaCpp");
  const sourceTree = requireRecord(
    llamaCpp.sourceTree,
    "reference llamaCpp.sourceTree"
  );
  const treeSha256 = requireString(
    sourceTree.sha256,
    "reference native source SHA-256"
  );
  const revision = requireString(
    llamaCpp.revision,
    "reference native revision"
  );
  const excluded = sourceTree.excludedDirectoryNames;
  if (
    !sha256Pattern.test(treeSha256) ||
    !revisionPattern.test(revision) ||
    !Array.isArray(excluded) ||
    excluded.length === 0 ||
    !excluded.every((entry) => typeof entry === "string")
  ) {
    throw new Error("reference native source identity is incomplete");
  }
  return {
    path: absolutePath,
    sha256: sha256Bytes(bytes),
    backend,
    lane: "cargo-equivalent",
    buildType: "Release",
    selectedTargets,
    native: {
      path: requireString(llamaCpp.path, "reference native source path"),
      revision,
      treeSha256,
      excludedDirectoryNames: excluded as string[],
    },
  };
};

export const candidateFeature = (
  backend: CandidateBackend
): string | undefined => (backend === "cpu" ? undefined : backend);

export const validateCandidateBackendForHost = (
  backend: CandidateBackend,
  platform: NodeJS.Platform,
  architecture: string
): void => {
  // The pinned llama-cpp-2 manifest unconditionally enables its sys crate's
  // Metal feature for Apple Silicon. Omitting our explicit `metal` feature does
  // not produce a CPU-only native binary, so accepting any other backend label
  // here would fabricate build parity with the reference manifest.
  if (platform === "darwin" && architecture === "arm64" && backend !== "metal") {
    throw new Error(
      `Apple Silicon candidate builds include Metal through the pinned binding manifest; backend ${backend} cannot match a non-Metal reference build`
    );
  }
};

export const candidateCargoArguments = (
  backend: CandidateBackend
): ReadonlyArray<string> => [
  "build",
  "--manifest-path",
  resolve(inferenceRoot, "Cargo.toml"),
  "--locked",
  "--release",
  "--package",
  candidatePackageName,
  "--no-default-features",
  ...(candidateFeature(backend) === undefined
    ? []
    : ["--features", candidateFeature(backend)!]),
];

const uniqueRunName = () => {
  const timestamp = new Date().toISOString().replaceAll(/[-:.TZ]/g, "");
  return `${timestamp}-${randomBytes(4).toString("hex")}`;
};

const isInside = (base: string, path: string) => {
  const child = relative(base, path);
  return (
    child.length > 0 &&
    child !== ".." &&
    !child.startsWith(`..${sep}`) &&
    !isAbsolute(child)
  );
};

const isWithin = (base: string, path: string) =>
  base === path || isInside(base, path);

const requireInsideInference = async (path: string, description: string) => {
  const [canonicalRoot, canonicalPath] = await Promise.all([
    realpath(inferenceRoot),
    realpath(path),
  ]);
  if (!isWithin(canonicalRoot, canonicalPath)) {
    throw new Error(`${description} must stay inside the inference directory`);
  }
  return canonicalPath;
};

const resolveOutputDirectory = (configured?: string) =>
  configured === undefined
    ? resolve(
        inferenceRoot,
        "results/parity/candidate",
        uniqueRunName()
      )
    : isAbsolute(configured)
      ? configured
      : resolve(process.cwd(), configured);

const prepareOutputDirectory = async (path: string) => {
  const lexical = resolve(path);
  if (!isInside(inferenceRoot, lexical)) {
    throw new Error("candidate output directory must stay inside inference/");
  }
  try {
    const entries = await readdir(lexical);
    if (entries.length > 0) {
      throw new Error(`candidate output directory is not empty: ${lexical}`);
    }
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code !== "ENOENT") throw error;
  }
  await mkdir(lexical, { recursive: true });
  await requireInsideInference(lexical, "candidate output directory");
  return lexical;
};

const formatCommand = (command: readonly string[]) =>
  command.map((part) => JSON.stringify(part)).join(" ");

const resolveTool = (name: string) => {
  const path = Bun.which(name);
  if (path === null) throw new Error(`required build tool is not on PATH: ${name}`);
  return path;
};

const capture = async (
  command: readonly string[],
  environment: Readonly<Record<string, string>>
) => {
  const child = Bun.spawn([...command], {
    cwd: inferenceRoot,
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
    throw new Error(stderr.trim() || `${command[0]} exited with ${exitCode}`);
  }
  return stdout.trim();
};

const buildEnvironment = (
  targetDirectory: string,
  parallel: number,
  cCompiler: string,
  cxxCompiler: string
) => {
  return controlledEnvironment({
    CARGO_INCREMENTAL: "0",
    CARGO_PROFILE_RELEASE_DEBUG_ASSERTIONS: "false",
    CARGO_TARGET_DIR: targetDirectory,
    CMAKE_BUILD_PARALLEL_LEVEL: String(parallel),
    CMAKE_C_COMPILER: cCompiler,
    CMAKE_CXX_COMPILER: cxxCompiler,
    RUSTFLAGS: "",
  });
};

const run = async (
  command: readonly string[],
  environment: Readonly<Record<string, string>>
) => {
  const child = Bun.spawn([...command], {
    cwd: inferenceRoot,
    env: environment,
    stdin: "inherit",
    stdout: "inherit",
    stderr: "inherit",
  });
  const exitCode = await child.exited;
  if (exitCode !== 0) {
    throw new Error(`${command[0]} exited with status ${exitCode}`);
  }
};

const inventoryComponents = async (
  components: ReadonlyArray<SourceComponent>
): Promise<ReadonlyArray<InventoriedComponent>> =>
  Promise.all(
    components.map(async (component) => ({
      ...component,
      inventory: await Effect.runPromise(
        sourceInventory(resolve(inferenceRoot, component.path), {
          excludedDirectoryNames: component.excludedDirectoryNames,
        })
      ),
    }))
  );

const requireUnchanged = (
  before: ReadonlyArray<InventoriedComponent>,
  after: ReadonlyArray<InventoriedComponent>
) => {
  for (const initial of before) {
    const final = after.find(
      (candidate) =>
        candidate.kind === initial.kind && candidate.name === initial.name
    );
    if (
      final === undefined ||
      final.inventory.sha256 !== initial.inventory.sha256 ||
      final.inventory.fileCount !== initial.inventory.fileCount ||
      final.inventory.totalBytes !== initial.inventory.totalBytes
    ) {
      throw new Error(
        `source component ${initial.kind}/${initial.name} changed during the candidate build`
      );
    }
  }
};

const relativeToInference = (path: string) => {
  const value = relative(inferenceRoot, path).replaceAll("\\", "/");
  if (
    value.length === 0 ||
    value === ".." ||
    value.startsWith("../") ||
    isAbsolute(value)
  ) {
    throw new Error(`path is not a child of inference/: ${path}`);
  }
  return value;
};

const fileRecordFromBytes = (path: string, bytes: Uint8Array) => ({
  path: relativeToInference(path),
  bytes: bytes.byteLength,
  sha256: sha256Bytes(bytes),
});

const artifactRecord = async (path: string) => {
  const bytes = await readFile(path);
  return {
    name: candidateArtifactName,
    ...fileRecordFromBytes(path, bytes),
  };
};

const snapshotCandidateConfiguration = async () =>
  Promise.all(
    candidateConfigurationInputs.map(async (input) => {
      const bytes = await readFile(input.path);
      return {
        ...input,
        text: new TextDecoder().decode(bytes),
        record: fileRecordFromBytes(input.path, bytes),
      };
    })
  );

const requireCandidateConfigurationUnchanged = async (
  before: Awaited<ReturnType<typeof snapshotCandidateConfiguration>>
) => {
  const after = await Promise.all(
    candidateConfigurationInputs.map(async (input) =>
      fileRecordFromBytes(input.path, await readFile(input.path))
    )
  );
  for (let index = 0; index < before.length; index += 1) {
    if (
      before[index]!.record.sha256 !== after[index]!.sha256 ||
      before[index]!.record.bytes !== after[index]!.bytes
    ) {
      throw new Error(
        `candidate build input changed during execution: ${before[index]!.record.path}`
      );
    }
  }
};

const main = async () => {
  const arguments_ = parseArguments(Bun.argv.slice(2));
  if (arguments_.help) {
    console.log(usage);
    return;
  }
  const configurationBefore = await snapshotCandidateConfiguration();
  const reference = await loadReferenceIdentity(arguments_.referenceManifest!);
  if (
    arguments_.backend !== undefined &&
    arguments_.backend !== reference.backend
  ) {
    throw new Error(
      `requested backend ${arguments_.backend} does not match reference backend ${reference.backend}`
    );
  }
  if (reference.backend === "metal" && process.platform !== "darwin") {
    throw new Error("the Metal candidate backend requires macOS");
  }
  validateCandidateBackendForHost(
    reference.backend,
    process.platform,
    process.arch
  );
  const nativePinSource = configurationBefore.find(
    (input) => input.name === "nativePin"
  )?.text;
  if (nativePinSource === undefined) {
    throw new Error("candidate build did not snapshot native-pin.toml");
  }
  const pin = await readNativePin(nativePinSource);
  if (
    reference.native.path !== pin.llama_cpp.checkout_path ||
    reference.native.revision !== pin.llama_cpp.revision
  ) {
    throw new Error(
      "reference native path/revision does not match native-pin.toml"
    );
  }

  const cargo = resolveTool("cargo");
  const command = [cargo, ...candidateCargoArguments(reference.backend)];
  console.log(`Reference manifest SHA-256: ${reference.sha256}`);
  console.log(formatCommand(command));
  if (arguments_.dryRun) return;

  const outputDirectory = await prepareOutputDirectory(
    resolveOutputDirectory(arguments_.outputDirectory)
  );
  const runName = outputDirectory.split(sep).at(-1)!;
  const targetDirectory = resolve(
    inferenceRoot,
    "target/parity-candidate",
    runName
  );
  await mkdir(resolve(inferenceRoot, "target/parity-candidate"), {
    recursive: true,
  });
  await mkdir(targetDirectory, { recursive: false });

  const workspaceExclusions = [
    ...defaultExcludedDirectoryNames,
    ".parity-models",
    "native",
    "crates",
    "parity",
    "results",
    "scripts",
    "info",
    "dist",
  ];
  const components: ReadonlyArray<SourceComponent> = [
    {
      kind: "native-source",
      name: "llama.cpp",
      path: pin.llama_cpp.checkout_path,
      revision: pin.llama_cpp.revision,
      dirty: true,
      excludedDirectoryNames: reference.native.excludedDirectoryNames,
    },
    {
      kind: "bindings-source",
      name: "llama-cpp-rs",
      path: pin.llama_cpp_rs.checkout_path,
      revision: pin.llama_cpp_rs.revision,
      dirty: true,
      excludedDirectoryNames: [
        ...defaultExcludedDirectoryNames,
        "llama.cpp",
      ],
    },
    {
      kind: "icn-source",
      name: "inference-workspace",
      path: ".",
      dirty: true,
      excludedDirectoryNames: workspaceExclusions,
    },
    {
      kind: "icn-source",
      name: "inference-crates",
      path: "crates",
      dirty: true,
      excludedDirectoryNames: defaultExcludedDirectoryNames,
    },
  ];
  for (const component of components) {
    await requireInsideInference(
      resolve(inferenceRoot, component.path),
      `source component ${component.name}`
    );
  }
  const before = await inventoryComponents(components);
  const native = before.find(
    (component) => component.kind === "native-source"
  );
  if (native?.inventory.sha256 !== reference.native.treeSha256) {
    throw new Error(
      "candidate native source tree does not match the reference build manifest"
    );
  }

  const rustc = resolveTool("rustc");
  const cCompiler = resolveTool(process.platform === "win32" ? "clang-cl" : "cc");
  const cxxCompiler = resolveTool(process.platform === "win32" ? "clang-cl" : "c++");
  const environment = buildEnvironment(
    targetDirectory,
    arguments_.parallel,
    cCompiler,
    cxxCompiler
  );
  await run(command, environment);

  const after = await inventoryComponents(components);
  requireUnchanged(before, after);
  const builtBinary = resolve(
    targetDirectory,
    "release",
    process.platform === "win32"
      ? `${candidateArtifactName}.exe`
      : candidateArtifactName
  );
  if (!(await stat(builtBinary)).isFile()) {
    throw new Error(`Cargo did not produce ${builtBinary}`);
  }
  const binaryDirectory = resolve(outputDirectory, "bin");
  await mkdir(binaryDirectory, { recursive: false });
  const preservedBinary = resolve(
    binaryDirectory,
    process.platform === "win32"
      ? `${candidateArtifactName}.exe`
      : candidateArtifactName
  );
  await copyFile(builtBinary, preservedBinary, fileConstants.COPYFILE_EXCL);
  if (process.platform !== "win32") await chmod(preservedBinary, 0o755);

  await Promise.all(
    before.map((component) =>
      writeFile(
        resolve(
          outputDirectory,
          `candidate-build-inputs-${component.kind}-${component.name}.json`
        ),
        `${JSON.stringify(component.inventory, null, 2)}\n`,
        { flag: "wx" }
      )
    )
  );
  const [rustcVersion, cVersion, cxxVersion, artifact] = await Promise.all([
    capture([rustc, "--version", "--verbose"], environment),
    capture([cCompiler, "--version"], environment),
    capture([cxxCompiler, "--version"], environment),
    artifactRecord(preservedBinary),
  ]);
  await requireCandidateConfigurationUnchanged(configurationBefore);
  const manifest = {
    schemaVersion: 1,
    createdAt: new Date().toISOString(),
    backend: reference.backend,
    lane: reference.lane,
    buildType: reference.buildType,
    referenceManifestSha256: reference.sha256,
    environment: controlledEnvironmentEvidence(environment),
    configuration: Object.fromEntries(
      configurationBefore.map((input) => [input.name, input.record])
    ),
    compiler: {
      name: `rustc + ${cCompiler} + ${cxxCompiler}`,
      version: [
        rustcVersion.split("\n")[0],
        cVersion.split("\n")[0],
        cxxVersion.split("\n")[0],
      ].join("; "),
    },
    flags: [
      ...candidateCargoArguments(reference.backend),
      "CARGO_INCREMENTAL=0",
      "CARGO_PROFILE_RELEASE_DEBUG_ASSERTIONS=false",
      `CMAKE_BUILD_PARALLEL_LEVEL=${arguments_.parallel}`,
      `CMAKE_C_COMPILER=${cCompiler}`,
      `CMAKE_CXX_COMPILER=${cxxCompiler}`,
      "RUSTFLAGS=",
      `reference-manifest-sha256=${reference.sha256}`,
    ],
    assertions: false,
    sanitizers: [],
    components: before.map((component) => ({
      kind: component.kind,
      name: component.name,
      path: component.path,
      ...(component.revision === undefined
        ? {}
        : { revision: component.revision }),
      treeSha256: component.inventory.sha256,
      dirty: component.dirty,
      excludedDirectoryNames: component.excludedDirectoryNames,
    })),
    artifacts: [artifact],
  };
  const manifestPath = resolve(
    outputDirectory,
    `candidate-build-${reference.backend}.json`
  );
  await writeFile(manifestPath, `${JSON.stringify(manifest, null, 2)}\n`, {
    flag: "wx",
  });
  console.log(`Candidate build manifest: ${manifestPath}`);
  console.log("Claim scope: production ICN parity probe through icn-llamacpp.");
};

if (import.meta.main) {
  void main().catch((error) => {
    console.error(error instanceof Error ? error.message : error);
    process.exitCode = 1;
  });
}
