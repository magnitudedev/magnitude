import { describe, expect, it } from "vitest"
import { Option, Schema } from "effect"
import releasePlan from "../../release-plan.json"
import { ReleaseManifestSchema } from "@magnitudedev/release"
import type { ReleaseArtifact, ReleaseManifest } from "@magnitudedev/release"
import { selectCudaArtifact } from "./cuda-compatibility"

const sha = "a".repeat(64)
const pack = (
  id: string,
  toolkitVersion: string,
  minimumDriverApi: number,
  target: number,
  architectureSpecific = false,
): ReleaseArtifact => ({
  id,
  kind: "icn-backend",
  host: Option.some("linux-x64-gnu"),
  backend: Option.some("cuda"),
  filename: `${id}.tar.gz`,
  bytes: 1,
  sha256: sha,
  requiredBaseId: Option.some("base"),
  nativeBuild: Option.some("native"),
  backendModuleAbi: Option.some("abi"),
  compatibility: Option.some({
    kind: "cuda",
    toolkitVersion,
    compiler: "test compiler",
    images: [{
      ptxVersion: toolkitVersion === "12.9" ? "8.8" : "7.8",
      target,
      architectureSpecific,
      minimumDriverApi,
    }],
  }),
})

const manifest = (artifacts: readonly ReleaseArtifact[]): ReleaseManifest => ({
  schemaVersion: 2,
  version: "1.0.0",
  acnRevision: 0,
  rpc: releasePlan.rpc,
  plugins: Schema.decodeUnknownSync(ReleaseManifestSchema.fields.plugins)(releasePlan.plugins.map(plugin => plugin.artifact)),
  tag: "@magnitudedev/cli@1.0.0",
  sourceCommit: "b".repeat(40),
  artifacts: artifacts as [ReleaseArtifact, ...ReleaseArtifact[]],
})

describe("CUDA pack selection", () => {
  const cuda118 = pack("cuda-11.8", "11.8", 11080, 80)
  const cuda129 = pack("cuda-12.9", "12.9", 12090, 80)

  it("selects the newest eligible compiler output using PTX driver floors", () => {
    const release = manifest([cuda118, cuda129])
    expect(Option.getOrThrow(selectCudaArtifact(
      release, "linux-x64-gnu", 12000, ["86"],
    )).id).toBe("cuda-11.8")
    expect(Option.getOrThrow(selectCudaArtifact(
      release, "linux-x64-gnu", 12090, ["86"],
    )).id).toBe("cuda-12.9")
  })

  it("allows ordinary older PTX on newer hardware", () => {
    expect(Option.getOrThrow(selectCudaArtifact(
      manifest([cuda118]), "linux-x64-gnu", 11080, ["120"],
    )).id).toBe("cuda-11.8")
  })

  it("does not treat architecture-specific PTX as a numeric minimum", () => {
    const exact = pack("blackwell", "12.9", 12090, 120, true)
    expect(Option.isNone(selectCudaArtifact(
      manifest([exact]), "linux-x64-gnu", 12090, ["130"],
    ))).toBe(true)
    expect(Option.getOrThrow(selectCudaArtifact(
      manifest([exact]), "linux-x64-gnu", 12090, ["120"],
    )).id).toBe("blackwell")
  })

  it("requires an applicable image for every detected CUDA architecture", () => {
    expect(Option.isNone(selectCudaArtifact(
      manifest([cuda118]), "linux-x64-gnu", 11080, ["75", "120"],
    ))).toBe(true)
  })
})
