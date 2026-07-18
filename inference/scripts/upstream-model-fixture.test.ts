import { afterEach, describe, expect, test } from "vitest";
import { createHash } from "node:crypto";
import {
  mkdtemp,
  link,
  mkdir,
  readFile,
  rm,
  stat,
  symlink,
  writeFile,
} from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import {
  acceptedRegistryArtifact,
  loadModelRegistry,
  prepareOfflineUpstreamFixtures,
  stageVerifiedModelFixture,
  verifyOfflineUpstreamFixtures,
  type AcceptedRegistryArtifact,
} from "./upstream-model-fixture";

const temporaryDirectories: string[] = [];
const acceptedBytes = Buffer.from("accepted offline model fixture\n");
const artifact: AcceptedRegistryArtifact = {
  modelId: "fixture-model",
  role: "model",
  relativePath: "models/fixture.gguf",
  bytes: acceptedBytes.byteLength,
  sha256: createHash("sha256").update(acceptedBytes).digest("hex"),
};

const temporaryFixture = async () => {
  const root = await mkdtemp(join(tmpdir(), "icn-upstream-fixture-"));
  temporaryDirectories.push(root);
  const sourceCachePath = join(root, "cache/models/fixture.gguf");
  const stagedPath = join(root, "build/models/fixture.gguf");
  return { root, sourceCachePath, stagedPath };
};

const write = async (path: string, bytes: Uint8Array) => {
  await mkdir(resolve(path, ".."), { recursive: true });
  await writeFile(path, bytes);
};

afterEach(async () => {
  await Promise.all(
    temporaryDirectories.splice(0).map((directory) =>
      rm(directory, { recursive: true, force: true })
    )
  );
});

describe("offline upstream model fixture", () => {
  test("fails when the registry artifact is missing", async () => {
    const fixture = await temporaryFixture();

    await expect(
      stageVerifiedModelFixture({ artifact, ...fixture })
    ).rejects.toThrow("Offline source artifact is missing");
  });

  test("fails when the registry artifact identity mismatches", async () => {
    const fixture = await temporaryFixture();
    await write(fixture.sourceCachePath, Buffer.from("unverified"));

    await expect(
      stageVerifiedModelFixture({ artifact, ...fixture })
    ).rejects.toThrow("does not match registry identity");
  });

  test("stages a verified artifact create-only", async () => {
    const fixture = await temporaryFixture();
    await write(fixture.sourceCachePath, acceptedBytes);

    const method = await stageVerifiedModelFixture({ artifact, ...fixture });

    expect(method).toBe("copy");
    expect(await readFile(fixture.stagedPath)).toEqual(acceptedBytes);
    expect((await stat(fixture.stagedPath)).ino).not.toBe(
      (await stat(fixture.sourceCachePath)).ino
    );
  });

  test("accepts an existing matching staged artifact", async () => {
    const fixture = await temporaryFixture();
    await write(fixture.sourceCachePath, acceptedBytes);
    await write(fixture.stagedPath, acceptedBytes);

    await expect(
      stageVerifiedModelFixture({ artifact, ...fixture })
    ).resolves.toBe("existing");
  });

  test("rejects an existing hard link into the immutable model cache", async () => {
    const fixture = await temporaryFixture();
    await write(fixture.sourceCachePath, acceptedBytes);
    await mkdir(resolve(fixture.stagedPath, ".."), { recursive: true });
    await link(fixture.sourceCachePath, fixture.stagedPath);

    await expect(
      stageVerifiedModelFixture({ artifact, ...fixture })
    ).rejects.toThrow("hard-linked to the model cache");
  });

  test("rejects and preserves an existing mismatched staged artifact", async () => {
    const fixture = await temporaryFixture();
    const mismatched = Buffer.from("do not overwrite\n");
    await write(fixture.sourceCachePath, acceptedBytes);
    await write(fixture.stagedPath, mismatched);

    await expect(
      stageVerifiedModelFixture({ artifact, ...fixture })
    ).rejects.toThrow("Existing staged artifact does not match registry identity");
    expect(await readFile(fixture.stagedPath)).toEqual(mismatched);
  });

  test("parses the accepted C0 artifact from the checked-in registry", async () => {
    const registry = await loadModelRegistry(
      resolve(import.meta.dirname, "../parity/models/registry.toml")
    );

    expect(acceptedRegistryArtifact(registry, "stories15m-q4-0", "model")).toEqual({
      modelId: "stories15m-q4-0",
      role: "model",
      relativePath: "tinyllamas/stories15M-q4_0.gguf",
      bytes: 19_077_344,
      sha256: "66967fbece6dbe97886593fdbb73589584927e29119ec31f08090732d1861739",
    });
  });

  test("rejects a staged parent symlink that escapes the build directory", async () => {
    const fixture = await temporaryFixture();
    const registryPath = join(fixture.root, "registry.toml");
    const externalDirectory = join(fixture.root, "external");
    await write(fixture.sourceCachePath, acceptedBytes);
    await mkdir(join(fixture.root, "build"), { recursive: true });
    await mkdir(externalDirectory, { recursive: true });
    await symlink(externalDirectory, join(fixture.root, "build", "models"));
    await writeFile(
      registryPath,
      `schema_version = 1\nartifact_root_env = "TEST_MODELS"\n\n[[models]]\nid = "fixture-model"\nstatus = "accepted"\nkind = "model"\nvalid_for = ["C0"]\n\n[[models.files]]\nrole = "model"\npath = "models/fixture.gguf"\nbytes = ${artifact.bytes}\nsha256 = "${artifact.sha256}"\nurl = "https://example.invalid/fixture.gguf"\n`
    );

    await expect(
      prepareOfflineUpstreamFixtures({
        registryPath,
        fixtures: [
          {
            setup_name: "download-model",
            model_id: "fixture-model",
            artifact_role: "model",
            destination: "models/fixture.gguf",
          },
        ],
        buildDirectory: join(fixture.root, "build"),
        configuredModelDirectory: join(fixture.root, "cache"),
      })
    ).rejects.toThrow("non-directory component");
  });

  test("rejects dot path aliases before staging", async () => {
    const fixture = await temporaryFixture();
    const registryPath = join(fixture.root, "registry.toml");
    await write(fixture.sourceCachePath, acceptedBytes);
    await mkdir(join(fixture.root, "build"), { recursive: true });
    await writeFile(
      registryPath,
      `schema_version = 1\nartifact_root_env = "TEST_MODELS"\n\n[[models]]\nid = "fixture-model"\nstatus = "accepted"\nkind = "model"\nvalid_for = ["C0"]\n\n[[models.files]]\nrole = "model"\npath = "models/fixture.gguf"\nbytes = ${artifact.bytes}\nsha256 = "${artifact.sha256}"\nurl = "https://example.invalid/fixture.gguf"\n`
    );

    await expect(
      prepareOfflineUpstreamFixtures({
        registryPath,
        fixtures: [
          {
            setup_name: "download-model",
            model_id: "fixture-model",
            artifact_role: "model",
            destination: "models/./fixture.gguf",
          },
        ],
        buildDirectory: join(fixture.root, "build"),
        configuredModelDirectory: join(fixture.root, "cache"),
      })
    ).rejects.toThrow("safe non-empty relative path");
  });

  test("postflight detects a staged artifact mutated by a test", async () => {
    const fixture = await temporaryFixture();
    await write(fixture.sourceCachePath, acceptedBytes);
    await stageVerifiedModelFixture({ artifact, ...fixture });
    await writeFile(fixture.stagedPath, Buffer.from("mutated by test\n"));

    await expect(
      verifyOfflineUpstreamFixtures({
        registryPath: join(fixture.root, "registry.toml"),
        modelDirectory: join(fixture.root, "cache"),
        modelDirectorySource: "cli",
        artifacts: [
          {
            setupName: "download-model",
            registryId: artifact.modelId,
            role: artifact.role,
            sourceCachePath: fixture.sourceCachePath,
            stagedPath: fixture.stagedPath,
            bytes: artifact.bytes,
            sha256: artifact.sha256,
            stagingMethod: "copy",
          },
        ],
      })
    ).rejects.toThrow("Post-CTest staged artifact does not match registry identity");
  });
});
