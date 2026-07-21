import {
  LOCAL_MODEL_CATALOG_OVERLAY,
  resolveCatalogArtifact,
  validateCanonicalModelCatalog,
} from "../src/local-inference/catalog"

interface HubSibling {
  readonly rfilename: string
  readonly size?: number
  readonly blobId?: string
  readonly lfs?: { readonly size: number; readonly sha256: string }
}

interface HubModel {
  readonly id: string
  readonly sha?: string
  readonly tags?: readonly string[]
  readonly cardData?: { readonly license?: string; readonly license_link?: string }
  readonly siblings?: readonly HubSibling[]
}

const cache = new Map<string, Promise<HubModel>>()
const fetchLatest = (repository: string, blobs = false): Promise<HubModel> => {
  const key = `${repository}:${blobs}`
  const existing = cache.get(key)
  if (existing) return existing
  const request = (async () => {
    const response = await fetch(`https://huggingface.co/api/models/${repository}${blobs ? "?blobs=true" : ""}`)
    if (!response.ok) throw new Error(`${repository}: Hugging Face returned HTTP ${response.status}`)
    return await response.json() as HubModel
  })()
  cache.set(key, request)
  return request
}

const structuralIssues = validateCanonicalModelCatalog(LOCAL_MODEL_CATALOG_OVERLAY)
if (structuralIssues.length > 0) throw new Error(structuralIssues.join("\n"))

const failures: string[] = []
const snapshots = new Map<string, HubModel>()
const modelRepositories = [...new Set(LOCAL_MODEL_CATALOG_OVERLAY.models.map(({ modelRepository }) => modelRepository))]
const artifactRepositories = [...new Set(LOCAL_MODEL_CATALOG_OVERLAY.models.flatMap(({ artifacts }) =>
  artifacts.map(({ repository }) => repository)))]

await Promise.all(modelRepositories.map(async (repository) => {
  try {
    const remote = await fetchLatest(repository)
    if (!remote.sha?.match(/^[a-f0-9]{40,64}$/)) failures.push(`${repository}: current revision is missing or invalid`)
  } catch (cause) {
    failures.push(cause instanceof Error ? cause.message : String(cause))
  }
}))

await Promise.all(artifactRepositories.map(async (repository) => {
  try {
    const remote = await fetchLatest(repository, true)
    snapshots.set(repository, remote)
    if (!remote.sha?.match(/^[a-f0-9]{40,64}$/)) failures.push(`${repository}: current revision is missing or invalid`)
    if (!(remote.siblings ?? []).some(({ rfilename }) => rfilename.toLowerCase().endsWith(".gguf"))) {
      failures.push(`${repository}: repository has no GGUF files`)
    }
  } catch (cause) {
    failures.push(cause instanceof Error ? cause.message : String(cause))
  }
}))

for (const model of LOCAL_MODEL_CATALOG_OVERLAY.models) {
  for (const candidate of model.artifacts) {
    const remote = snapshots.get(candidate.repository)
    if (!remote?.sha) continue
    const license = remote.cardData?.license
      ?? remote.tags?.find((tag) => tag.startsWith("license:"))?.slice("license:".length)
    if (license && license !== "other" && license !== model.licenseReview.expectedId) {
      failures.push(`${candidate.id}: Hub license ${license} differs from reviewed ${model.licenseReview.expectedId}`)
    }
    const entry = resolveCatalogArtifact(model, candidate, {
      repository: remote.id,
      commit: remote.sha,
      license,
      license_url: remote.cardData?.license_link,
      gguf_files: (remote.siblings ?? []).map(({ rfilename, size, lfs }) => ({
        path: rfilename,
        size_bytes: lfs?.size ?? size,
      })),
    })
    if (!entry) {
      failures.push(`${candidate.id}: selector ${candidate.filenameIncludes} did not resolve uniquely`)
      continue
    }
    const primary = remote.siblings?.find(({ rfilename }) => rfilename === entry.primaryGguf)
    const size = primary?.lfs?.size ?? primary?.size
    const identity = primary?.lfs?.sha256 ?? primary?.blobId
    if (!size || !identity) failures.push(`${candidate.id}: selected primary lacks live size or content identity`)
  }
}

if (failures.length > 0) {
  console.error(`Local model catalog validation failed:\n${failures.map((failure) => `- ${failure}`).join("\n")}`)
  process.exit(1)
}

const artifactCount = LOCAL_MODEL_CATALOG_OVERLAY.models.reduce((count, model) => count + model.artifacts.length, 0)
console.log(`Validated ${LOCAL_MODEL_CATALOG_OVERLAY.models.length} model overlays and ${artifactCount} artifact selectors against current Hugging Face repository state.`)
