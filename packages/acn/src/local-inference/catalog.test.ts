import { describe, expect, it } from "vitest"
import {
  LOCAL_MODEL_CATALOG_OVERLAY,
  resolveCatalogArtifact,
  validateCanonicalModelCatalog,
} from "./catalog"
import type { CanonicalModelCatalogOverlay } from "./types"

describe("canonical local model catalog overlay", () => {
  it("is internally valid and versioned", () => {
    expect(LOCAL_MODEL_CATALOG_OVERLAY).toMatchObject({
      schemaVersion: 2,
      catalogVersion: "2026-07-20",
      reviewedAt: "2026-07-20",
    })
    expect(validateCanonicalModelCatalog(LOCAL_MODEL_CATALOG_OVERLAY)).toEqual([])
  })

  it("contains stable identities and selectors, never Hub snapshots", () => {
    const encoded = JSON.stringify(LOCAL_MODEL_CATALOG_OVERLAY)
    expect(encoded).not.toContain('"revision"')
    expect(encoded).not.toContain('"sha256"')
    expect(encoded).not.toContain('"sizeBytes"')
    expect(encoded).not.toContain('"primaryFile"')
    expect(LOCAL_MODEL_CATALOG_OVERLAY.models).toHaveLength(13)
    expect(LOCAL_MODEL_CATALOG_OVERLAY.models.flatMap(({ artifacts }) => artifacts)).toHaveLength(25)
  })

  it("groups fidelity choices under one canonical model", () => {
    const model = LOCAL_MODEL_CATALOG_OVERLAY.models.find(({ id }) => id === "qwen3.6-35b-a3b")!
    expect(model.artifacts.map(({ quantization }) => quantization.format)).toEqual([
      "UD-Q4_K_XL",
      "UD-Q5_K_XL",
      "UD-Q6_K_XL",
      "UD-Q8_K_XL",
    ])
  })

  it("resolves an unsharded selector against a live snapshot", () => {
    const model = LOCAL_MODEL_CATALOG_OVERLAY.models[0]!
    const candidate = model.artifacts[0]!
    expect(resolveCatalogArtifact(model, candidate, {
      repository: candidate.repository,
      commit: "a".repeat(40),
      license: "apache-2.0",
      license_url: null,
      gguf_files: [
        { path: "model-UD-Q4_K_XL.gguf", size_bytes: 100 },
        { path: "model-UD-Q5_K_XL.gguf", size_bytes: 120 },
        { path: "mmproj-model-UD-Q4_K_XL.gguf", size_bytes: 5 },
      ],
    })).toMatchObject({
      revision: "a".repeat(40),
      primaryGguf: "model-UD-Q4_K_XL.gguf",
      publishedWeightBytes: 100,
      quantTag: "UD-Q4_K_XL",
    })
  })

  it("selects only the first file of a complete shard family as preview primary", () => {
    const model = LOCAL_MODEL_CATALOG_OVERLAY.models.find(({ id }) => id === "glm-5.2")!
    const candidate = model.artifacts[0]!
    expect(resolveCatalogArtifact(model, candidate, {
      repository: candidate.repository,
      commit: "b".repeat(40),
      gguf_files: [
        { path: "UD-Q4_K_XL/model-UD-Q4_K_XL-00002-of-00003.gguf", size_bytes: 20 },
        { path: "UD-Q4_K_XL/model-UD-Q4_K_XL-00001-of-00003.gguf", size_bytes: 10 },
        { path: "UD-Q4_K_XL/model-UD-Q4_K_XL-00003-of-00003.gguf", size_bytes: 30 },
      ],
    })).toMatchObject({
      primaryGguf: "UD-Q4_K_XL/model-UD-Q4_K_XL-00001-of-00003.gguf",
      publishedWeightBytes: 60,
    })
  })

  it("refuses ambiguous or missing selectors", () => {
    const model = LOCAL_MODEL_CATALOG_OVERLAY.models[0]!
    const candidate = model.artifacts[0]!
    const snapshot = { repository: candidate.repository, commit: "c".repeat(40), gguf_files: [] }
    expect(resolveCatalogArtifact(model, candidate, snapshot)).toBeUndefined()
    expect(resolveCatalogArtifact(model, candidate, {
      ...snapshot,
      gguf_files: [{ path: "a-UD-Q4_K_XL.gguf", size_bytes: 10 }, { path: "b-UD-Q4_K_XL.gguf", size_bytes: 10 }],
    })).toBeUndefined()
  })

  it("rejects duplicate IDs and pinned-path selectors", () => {
    const original = LOCAL_MODEL_CATALOG_OVERLAY.models[0]!
    const broken: CanonicalModelCatalogOverlay = {
      ...LOCAL_MODEL_CATALOG_OVERLAY,
      models: [original, {
        ...original,
        artifacts: [{ ...original.artifacts[0]!, filenameIncludes: "model.gguf" }],
      }],
    }
    const issues = validateCanonicalModelCatalog(broken)
    expect(issues).toContain(`duplicate model id ${original.id}`)
    expect(issues).toContain(`duplicate artifact id ${original.artifacts[0]!.id}`)
    expect(issues.some((issue) => issue.includes("pinned path"))).toBe(true)
  })
})
