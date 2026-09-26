import { afterEach, describe, expect, test } from "vitest";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
  candidateArtifactName,
  candidateCargoArguments,
  candidateFeature,
  candidatePackageName,
  loadReferenceIdentity,
  parseArguments,
  validateCandidateBackendForHost,
} from "./build-candidate";

const temporaryDirectories: string[] = [];

afterEach(async () => {
  await Promise.all(
    temporaryDirectories.splice(0).map((directory) =>
      rm(directory, { force: true, recursive: true })
    )
  );
});

const referenceManifest = (overrides: Record<string, unknown> = {}) => ({
  schemaVersion: 3,
  backend: "cpu",
  lane: "cargo-equivalent",
  buildType: "Release",
  selectedTargets: [{ id: "oracle" }],
  llamaCpp: {
    path: "native/llama-cpp-rs/llama-cpp-sys-2/llama.cpp",
    revision: "9e3b928fd8c9d14dbf15a8768b9fdd7e5c721d66",
    sourceTree: {
      sha256: "a".repeat(64),
      excludedDirectoryNames: [".git", "target"],
    },
  },
  ...overrides,
});

describe("build-candidate configuration", () => {
  test("maps each backend to an explicit Cargo feature policy", () => {
    expect(candidatePackageName).toBe("icn-parity-probe");
    expect(candidateArtifactName).toBe("icn-probe");
    expect(candidateFeature("cpu")).toBeUndefined();
    expect(candidateFeature("metal")).toBe("metal");
    expect(candidateFeature("cuda")).toBe("cuda");
    expect(candidateFeature("vulkan")).toBe("vulkan");
    expect(candidateCargoArguments("cpu")).not.toContain("--features");
    expect(candidateCargoArguments("metal")).toContain("metal");
    expect(candidateCargoArguments("cpu")).toContain("--locked");
    expect(candidateCargoArguments("cpu")).toContain("--release");
    expect(candidateCargoArguments("cpu")).toContain("icn-parity-probe");
    expect(candidateCargoArguments("cpu")).not.toContain("icn-bench");
  });

  test("refuses to mislabel Apple Silicon's unconditional Metal build", () => {
    expect(() =>
      validateCandidateBackendForHost("cpu", "darwin", "arm64")
    ).toThrow("include Metal");
    expect(() =>
      validateCandidateBackendForHost("vulkan", "darwin", "arm64")
    ).toThrow("non-Metal reference");
    expect(() =>
      validateCandidateBackendForHost("metal", "darwin", "arm64")
    ).not.toThrow();
    expect(() =>
      validateCandidateBackendForHost("cpu", "linux", "x64")
    ).not.toThrow();
  });

  test("requires a reference and rejects ambiguous CLI values", () => {
    expect(() => parseArguments([])).toThrow("--reference-manifest is required");
    expect(() =>
      parseArguments(["--reference-manifest", "ref.json", "--backend", "rocm"])
    ).toThrow("Unsupported backend");
    expect(
      parseArguments([
        "--reference-manifest",
        "ref.json",
        "--parallel",
        "4",
        "--dry-run",
      ])
    ).toMatchObject({ parallel: 4, dryRun: true });
  });

  test("accepts any non-empty target set from a release cargo-equivalent reference", async () => {
    const directory = await mkdtemp(join(tmpdir(), "icn-candidate-reference-"));
    temporaryDirectories.push(directory);
    const accepted = join(directory, "accepted.json");
    await writeFile(accepted, JSON.stringify(referenceManifest()));
    const identity = await loadReferenceIdentity(accepted);
    expect(identity.backend).toBe("cpu");
    expect(identity.lane).toBe("cargo-equivalent");
    expect(identity.sha256).toMatch(/^[0-9a-f]{64}$/);

    const wrongLane = join(directory, "wrong-lane.json");
    await writeFile(
      wrongLane,
      JSON.stringify(referenceManifest({ lane: "upstream-default" }))
    );
    await expect(loadReferenceIdentity(wrongLane)).rejects.toThrow(
      "cargo-equivalent"
    );

    const emptyTargets = join(directory, "empty-targets.json");
    await writeFile(
      emptyTargets,
      JSON.stringify(referenceManifest({ selectedTargets: [] }))
    );
    await expect(loadReferenceIdentity(emptyTargets)).rejects.toThrow(
      "selectedTargets must be non-empty"
    );
  });
});
