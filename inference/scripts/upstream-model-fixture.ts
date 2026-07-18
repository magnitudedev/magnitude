import { Schema } from "effect";
import { createHash } from "node:crypto";
import { createReadStream } from "node:fs";
import {
  constants,
  copyFile,
  lstat,
  mkdir,
  realpath,
  stat,
} from "node:fs/promises";
import { dirname, isAbsolute, relative, resolve, sep } from "node:path";
import type { CTestModelFixture } from "./build-parity-config";

const RegistryFileSchema = Schema.Struct({
  role: Schema.String,
  path: Schema.String,
  bytes: Schema.Number,
  sha256: Schema.String,
  url: Schema.String,
});

const RegistryModelSchema = Schema.Struct({
  id: Schema.String,
  status: Schema.String,
  kind: Schema.String,
  valid_for: Schema.Array(Schema.String),
  files: Schema.Array(RegistryFileSchema),
});

const ModelRegistrySchema = Schema.Struct({
  schema_version: Schema.Literal(1),
  artifact_root_env: Schema.String,
  models: Schema.Array(RegistryModelSchema),
});

export type ModelRegistry = typeof ModelRegistrySchema.Type;

export interface AcceptedRegistryArtifact {
  readonly modelId: string;
  readonly role: string;
  readonly relativePath: string;
  readonly bytes: number;
  readonly sha256: string;
}

export type StagingMethod = "copy" | "existing";

export interface StagedFixtureArtifact {
  readonly setupName: string;
  readonly registryId: string;
  readonly role: string;
  readonly sourceCachePath: string;
  readonly stagedPath: string;
  readonly bytes: number;
  readonly sha256: string;
  readonly stagingMethod: StagingMethod;
}

export interface OfflineFixturePreparation {
  readonly registryPath: string;
  readonly modelDirectory: string | null;
  readonly modelDirectorySource: "cli" | "environment" | null;
  readonly artifacts: readonly StagedFixtureArtifact[];
}

const validSha256 = (value: string) => /^[a-f0-9]{64}$/.test(value);

const safeRelativePath = (value: string, description: string) => {
  const normalized = value.replaceAll("\\", "/");
  if (
    normalized.length === 0 ||
    normalized.startsWith("/") ||
    /^[A-Za-z]:\//.test(normalized) ||
    normalized
      .split("/")
      .some((part) => part === "" || part === "." || part === "..")
  ) {
    throw new Error(`${description} must be a safe non-empty relative path`);
  }
  return normalized;
};

const errorCode = (error: unknown) =>
  (error as NodeJS.ErrnoException | undefined)?.code;

class MissingArtifactError extends Error {}

const isDescendantOrEqual = (root: string, candidate: string) => {
  const child = relative(root, candidate);
  return (
    child.length === 0 ||
    (child !== ".." &&
      !child.startsWith(`..${sep}`) &&
      !isAbsolute(child))
  );
};

const ensureSafeDestinationParent = async (
  canonicalRoot: string,
  destination: string
) => {
  const parent = dirname(destination);
  const relativeParent = relative(canonicalRoot, parent);
  if (!isDescendantOrEqual(canonicalRoot, parent)) {
    throw new Error(`Staged fixture destination escapes its build: ${destination}`);
  }
  let current = canonicalRoot;
  for (const part of relativeParent.split(sep).filter((value) => value.length > 0)) {
    current = resolve(current, part);
    try {
      const metadata = await lstat(current);
      if (!metadata.isDirectory()) {
        throw new Error(
          `Staged fixture destination has a non-directory component: ${current}`
        );
      }
    } catch (error) {
      if (errorCode(error) !== "ENOENT") throw error;
      await mkdir(current);
    }
  }
};

const fileIdentity = async (path: string, description: string) => {
  let metadata;
  try {
    metadata = await lstat(path);
  } catch (error) {
    if (errorCode(error) === "ENOENT") {
      throw new MissingArtifactError(`${description} is missing: ${path}`);
    }
    throw error;
  }
  if (!metadata.isFile()) {
    throw new Error(`${description} is not a regular file: ${path}`);
  }
  const digest = createHash("sha256");
  let bytes = 0;
  for await (const chunk of createReadStream(path)) {
    bytes += chunk.length;
    digest.update(chunk);
  }
  return { bytes, sha256: digest.digest("hex") };
};

const verifyExactArtifact = async (
  path: string,
  artifact: AcceptedRegistryArtifact,
  description: string
) => {
  const actual = await fileIdentity(path, description);
  if (actual.bytes !== artifact.bytes || actual.sha256 !== artifact.sha256) {
    throw new Error(
      `${description} does not match registry identity at ${path}: ` +
        `expected ${artifact.bytes} bytes/${artifact.sha256}, ` +
        `got ${actual.bytes} bytes/${actual.sha256}`
    );
  }
};

const verifyIndependentInodes = async (
  sourcePath: string,
  stagedPath: string,
  description: string
) => {
  const [sourceMetadata, stagedMetadata] = await Promise.all([
    stat(sourcePath),
    stat(stagedPath),
  ]);
  if (
    sourceMetadata.ino !== 0 &&
    sourceMetadata.dev === stagedMetadata.dev &&
    sourceMetadata.ino === stagedMetadata.ino
  ) {
    throw new Error(`${description} is hard-linked to the model cache: ${stagedPath}`);
  }
};

const validateRegistry = (registry: ModelRegistry) => {
  if (registry.artifact_root_env.trim().length === 0) {
    throw new Error("Model registry artifact_root_env must be non-empty");
  }
  const modelIds = new Set<string>();
  for (const model of registry.models) {
    if (model.id.trim().length === 0 || modelIds.has(model.id)) {
      throw new Error(`Invalid or duplicate model registry id: ${model.id}`);
    }
    modelIds.add(model.id);
    if (model.files.length === 0) {
      throw new Error(`Model registry entry ${model.id} has no files`);
    }
    const roles = new Set<string>();
    for (const file of model.files) {
      safeRelativePath(file.path, `Model ${model.id} artifact path`);
      if (
        file.role.trim().length === 0 ||
        roles.has(file.role) ||
        !Number.isSafeInteger(file.bytes) ||
        file.bytes < 1 ||
        !validSha256(file.sha256)
      ) {
        throw new Error(
          `Model registry entry ${model.id} has an invalid ${file.role || "unnamed"} artifact`
        );
      }
      roles.add(file.role);
    }
  }
};

export const loadModelRegistry = async (
  registryPath: string
): Promise<ModelRegistry> => {
  const file = Bun.file(registryPath);
  if (!(await file.exists())) {
    throw new Error(`Missing model registry: ${registryPath}`);
  }
  const registry = Schema.decodeUnknownSync(ModelRegistrySchema)(
    Bun.TOML.parse(await file.text())
  );
  validateRegistry(registry);
  return registry;
};

export const acceptedRegistryArtifact = (
  registry: ModelRegistry,
  modelId: string,
  role: string
): AcceptedRegistryArtifact => {
  const model = registry.models.find((candidate) => candidate.id === modelId);
  if (model === undefined) {
    throw new Error(`Unknown model registry id: ${modelId}`);
  }
  if (model.status !== "accepted" || model.kind !== "model") {
    throw new Error(`Model registry id ${modelId} is not an accepted model`);
  }
  if (!model.valid_for.includes("C0")) {
    throw new Error(`Model registry id ${modelId} is not accepted for C0`);
  }
  const artifact = model.files.find((candidate) => candidate.role === role);
  if (artifact === undefined) {
    throw new Error(`Model registry id ${modelId} has no ${role} artifact`);
  }
  return {
    modelId,
    role,
    relativePath: safeRelativePath(
      artifact.path,
      `Model ${modelId} artifact path`
    ),
    bytes: artifact.bytes,
    sha256: artifact.sha256,
  };
};

const resolveModelDirectory = (
  registry: ModelRegistry,
  configured: string | undefined,
  environment: Readonly<Record<string, string | undefined>>,
  cwd: string
) => {
  const environmentValue = environment[registry.artifact_root_env];
  const value = configured ?? environmentValue;
  if (value === undefined || value.length === 0) {
    throw new Error(
      `Offline upstream tests require --model-dir or ${registry.artifact_root_env}; ` +
        "the test runner never downloads model fixtures"
    );
  }
  return {
    path: isAbsolute(value) ? resolve(value) : resolve(cwd, value),
    source: configured === undefined ? ("environment" as const) : ("cli" as const),
  };
};

export const stageVerifiedModelFixture = async (options: {
  readonly artifact: AcceptedRegistryArtifact;
  readonly sourceCachePath: string;
  readonly stagedPath: string;
}): Promise<StagingMethod> => {
  await verifyExactArtifact(
    options.sourceCachePath,
    options.artifact,
    "Offline source artifact"
  );

  try {
    await verifyExactArtifact(
      options.stagedPath,
      options.artifact,
      "Existing staged artifact"
    );
    await verifyIndependentInodes(
      options.sourceCachePath,
      options.stagedPath,
      "Existing staged artifact"
    );
    return "existing";
  } catch (error) {
    if (!(error instanceof MissingArtifactError)) throw error;
  }

  const parent = dirname(options.stagedPath);
  await mkdir(parent, { recursive: true });
  try {
    // Never hard-link a cache artifact into an upstream build tree: tests are
    // allowed to mutate their working files, and a hard link would silently
    // corrupt the content-addressed cache inode.
    await copyFile(
      options.sourceCachePath,
      options.stagedPath,
      constants.COPYFILE_EXCL
    );
  } catch (error) {
    if (errorCode(error) === "EEXIST") {
      await verifyExactArtifact(
        options.stagedPath,
        options.artifact,
        "Concurrently staged artifact"
      );
      await verifyIndependentInodes(
        options.sourceCachePath,
        options.stagedPath,
        "Concurrently staged artifact"
      );
      return "existing";
    }
    throw error;
  }

  await verifyExactArtifact(
    options.stagedPath,
    options.artifact,
    "Newly staged artifact"
  );
  return "copy";
};

export const prepareOfflineUpstreamFixtures = async (options: {
  readonly registryPath: string;
  readonly fixtures: readonly CTestModelFixture[];
  readonly buildDirectory: string;
  readonly configuredModelDirectory?: string;
  readonly environment?: Readonly<Record<string, string | undefined>>;
  readonly cwd?: string;
}): Promise<OfflineFixturePreparation> => {
  const registryPath = resolve(options.registryPath);
  if (options.fixtures.length === 0) {
    return {
      registryPath,
      modelDirectory: null,
      modelDirectorySource: null,
      artifacts: [],
    };
  }

  const registry = await loadModelRegistry(registryPath);
  const modelDirectory = resolveModelDirectory(
    registry,
    options.configuredModelDirectory,
    options.environment ?? process.env,
    options.cwd ?? process.cwd()
  );
  const [canonicalModelDirectory, canonicalBuildDirectory] = await Promise.all([
    realpath(modelDirectory.path),
    realpath(options.buildDirectory),
  ]);
  if (
    isDescendantOrEqual(canonicalModelDirectory, canonicalBuildDirectory) ||
    isDescendantOrEqual(canonicalBuildDirectory, canonicalModelDirectory)
  ) {
    throw new Error(
      "Offline model cache and upstream build directory must be disjoint"
    );
  }
  const fixturesByDestination = new Map<string, CTestModelFixture>();
  for (const fixture of options.fixtures) {
    const destination = safeRelativePath(
      fixture.destination,
      `CTest fixture ${fixture.setup_name} destination`
    );
    const existing = fixturesByDestination.get(destination);
    if (
      existing !== undefined &&
      (existing.setup_name !== fixture.setup_name ||
        existing.model_id !== fixture.model_id ||
        existing.artifact_role !== fixture.artifact_role)
    ) {
      throw new Error(`Conflicting offline fixtures target ${destination}`);
    }
    fixturesByDestination.set(destination, fixture);
  }

  const artifacts: StagedFixtureArtifact[] = [];
  for (const [destination, fixture] of fixturesByDestination) {
    const artifact = acceptedRegistryArtifact(
      registry,
      fixture.model_id,
      fixture.artifact_role
    );
    const configuredSourceCachePath = resolve(
      canonicalModelDirectory,
      artifact.relativePath
    );
    const sourceCachePath = await realpath(configuredSourceCachePath).catch(
      (error) => {
        if (errorCode(error) === "ENOENT") return configuredSourceCachePath;
        throw error;
      }
    );
    if (!isDescendantOrEqual(canonicalModelDirectory, sourceCachePath)) {
      throw new Error(
        `Offline source artifact escapes the model directory: ${configuredSourceCachePath}`
      );
    }
    const stagedPath = resolve(canonicalBuildDirectory, destination);
    await ensureSafeDestinationParent(canonicalBuildDirectory, stagedPath);
    const stagingMethod = await stageVerifiedModelFixture({
      artifact,
      sourceCachePath,
      stagedPath,
    });
    artifacts.push({
      setupName: fixture.setup_name,
      registryId: fixture.model_id,
      role: fixture.artifact_role,
      sourceCachePath,
      stagedPath,
      bytes: artifact.bytes,
      sha256: artifact.sha256,
      stagingMethod,
    });
  }
  return {
    registryPath,
    modelDirectory: canonicalModelDirectory,
    modelDirectorySource: modelDirectory.source,
    artifacts,
  };
};

/** Re-prove cache and staged identities after upstream CTest has run. */
export const verifyOfflineUpstreamFixtures = async (
  preparation: OfflineFixturePreparation
) => {
  for (const staged of preparation.artifacts) {
    const expected: AcceptedRegistryArtifact = {
      modelId: staged.registryId,
      role: staged.role,
      relativePath: "postflight-only",
      bytes: staged.bytes,
      sha256: staged.sha256,
    };
    await verifyExactArtifact(
      staged.sourceCachePath,
      expected,
      "Post-CTest source artifact"
    );
    await verifyExactArtifact(
      staged.stagedPath,
      expected,
      "Post-CTest staged artifact"
    );
    await verifyIndependentInodes(
      staged.sourceCachePath,
      staged.stagedPath,
      "Post-CTest staged artifact"
    );
  }
};
