import { Schema } from "effect";
import { resolve } from "node:path";

const StringRecord = Schema.Record({
  key: Schema.String,
  value: Schema.String,
});

const CTestModelFixtureSchema = Schema.Struct({
  setup_name: Schema.String,
  model_id: Schema.String,
  artifact_role: Schema.String,
  destination: Schema.String,
});

const ReferenceTargetSchema = Schema.Struct({
  id: Schema.String,
  kind: Schema.Literal("ctest-suite", "upstream-tool", "oracle"),
  description: Schema.String,
  requires_tests: Schema.Boolean,
  requires_tools: Schema.Boolean,
  requires_server: Schema.Boolean,
  cmake_targets: Schema.Array(Schema.String),
  artifacts: Schema.Array(Schema.String),
  ctest_names: Schema.Array(Schema.String),
  ctest_setup_names: Schema.Array(Schema.String),
  ctest_model_fixtures: Schema.Array(CTestModelFixtureSchema),
});

const TargetManifestSchema = Schema.Struct({
  schema_version: Schema.Literal(1),
  default_targets: Schema.Array(Schema.String),
  targets: Schema.Array(ReferenceTargetSchema),
});

const BuildProfileSchema = Schema.Struct({
  description: Schema.String,
  definitions: StringRecord,
  darwin_definitions: StringRecord,
  metal_definitions: StringRecord,
  osx_architecture_from_host: Schema.Boolean,
});

const BuildProfilesSchema = Schema.Struct({
  schema_version: Schema.Literal(1),
  profiles: Schema.Struct({
    "upstream-default": BuildProfileSchema,
    "cargo-equivalent": BuildProfileSchema,
  }),
});

export type ReferenceTarget = typeof ReferenceTargetSchema.Type;
export type CTestModelFixture = typeof CTestModelFixtureSchema.Type;
export type TargetManifest = typeof TargetManifestSchema.Type;
export type BuildProfile = typeof BuildProfileSchema.Type;
export type BuildProfiles = typeof BuildProfilesSchema.Type;
export type BuildLane = keyof BuildProfiles["profiles"];

const readToml = async (path: string, source?: string) => {
  if (source !== undefined) return Bun.TOML.parse(source);
  const file = Bun.file(path);
  if (!(await file.exists())) throw new Error(`Missing parity configuration: ${path}`);
  return Bun.TOML.parse(await file.text());
};

const ensureUnique = (values: readonly string[], description: string) => {
  const seen = new Set<string>();
  for (const value of values) {
    if (seen.has(value)) throw new Error(`Duplicate ${description}: ${value}`);
    seen.add(value);
  }
};

const ensureSafeRelativePath = (value: string, description: string) => {
  const normalized = value.replaceAll("\\", "/");
  if (
    normalized.length === 0 ||
    normalized.startsWith("/") ||
    /^[A-Za-z]:\//.test(normalized) ||
    normalized.split("/").some((part) => part === "" || part === "..")
  ) {
    throw new Error(`${description} must be a safe non-empty relative path`);
  }
};

export const loadTargetManifest = async (
  inferenceRoot: string,
  exactSource?: string
): Promise<TargetManifest> => {
  const path = resolve(inferenceRoot, "parity/upstream/targets.toml");
  const manifest = Schema.decodeUnknownSync(TargetManifestSchema)(
    await readToml(path, exactSource)
  );
  ensureUnique(
    manifest.targets.map((target) => target.id),
    "reference target id"
  );
  ensureUnique(manifest.default_targets, "default reference target id");
  for (const target of manifest.targets) {
    if (target.cmake_targets.length === 0) {
      throw new Error(`Reference target ${target.id} has no CMake targets`);
    }
    ensureUnique(target.cmake_targets, `${target.id} CMake target`);
    ensureUnique(target.artifacts, `${target.id} artifact`);
    ensureUnique(target.ctest_names, `${target.id} CTest name`);
    ensureUnique(target.ctest_setup_names, `${target.id} CTest setup name`);
    ensureUnique(
      target.ctest_model_fixtures.map(
        (fixture) =>
          `${fixture.setup_name}\0${fixture.model_id}\0${fixture.artifact_role}\0${fixture.destination}`
      ),
      `${target.id} CTest model fixture`
    );
    if (target.ctest_names.length > 0 && !target.requires_tests) {
      throw new Error(
        `Reference target ${target.id} declares CTests without requires_tests`
      );
    }
    for (const setupName of target.ctest_setup_names) {
      if (target.ctest_names.includes(setupName)) {
        throw new Error(
          `Reference target ${target.id} setup ${setupName} cannot also be a selected CTest`
        );
      }
    }
    if (target.requires_server && !target.requires_tools) {
      throw new Error(
        `Reference target ${target.id} requires the server but not tools`
      );
    }
    for (const fixture of target.ctest_model_fixtures) {
      if (!target.ctest_setup_names.includes(fixture.setup_name)) {
        throw new Error(
          `Reference target ${target.id} fixture names undeclared setup test ${fixture.setup_name}`
        );
      }
      for (const [value, name] of [
        [fixture.setup_name, "setup name"],
        [fixture.model_id, "model id"],
        [fixture.artifact_role, "artifact role"],
      ] as const) {
        if (value.trim().length === 0) {
          throw new Error(`Reference target ${target.id} fixture ${name} is empty`);
        }
      }
      ensureSafeRelativePath(
        fixture.destination,
        `Reference target ${target.id} fixture destination`
      );
    }
    for (const setupName of target.ctest_setup_names) {
      if (
        !target.ctest_model_fixtures.some(
          (fixture) => fixture.setup_name === setupName
        )
      ) {
        throw new Error(
          `Reference target ${target.id} setup ${setupName} has no offline model fixture`
        );
      }
    }
    for (const artifact of target.artifacts) {
      if (!target.cmake_targets.includes(artifact)) {
        throw new Error(
          `Reference target ${target.id} artifact ${artifact} is not a CMake target`
        );
      }
    }
  }
  const ids = new Set(manifest.targets.map((target) => target.id));
  for (const id of manifest.default_targets) {
    if (!ids.has(id)) throw new Error(`Unknown default reference target: ${id}`);
  }
  return manifest;
};

export const loadBuildProfiles = async (
  inferenceRoot: string,
  exactSource?: string
): Promise<BuildProfiles> => {
  const path = resolve(inferenceRoot, "parity/upstream/build-profiles.toml");
  return Schema.decodeUnknownSync(BuildProfilesSchema)(
    await readToml(path, exactSource)
  );
};

export const selectReferenceTargets = (
  manifest: TargetManifest,
  requested: readonly string[]
): readonly ReferenceTarget[] => {
  if (requested.includes("all") && requested.length !== 1) {
    throw new Error("Reference target 'all' cannot be combined with other targets");
  }
  const ids =
    requested.length === 0
      ? manifest.default_targets
      : requested.includes("all")
        ? manifest.targets.map((target) => target.id)
        : requested;
  const byId = new Map(manifest.targets.map((target) => [target.id, target]));
  const selected: ReferenceTarget[] = [];
  const seen = new Set<string>();
  for (const id of ids) {
    const target = byId.get(id);
    if (target === undefined) throw new Error(`Unknown reference target: ${id}`);
    if (!seen.has(id)) {
      seen.add(id);
      selected.push(target);
    }
  }
  return selected;
};

export const uniqueStrings = (
  values: Iterable<string>
): readonly string[] => [...new Set(values)];
