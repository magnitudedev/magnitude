import { Array as Arr, Option, Schema } from "effect"
import {
  MODEL_RECIPE_REGISTRY,
  resolveModelRecipeArtifact,
  validateModelRecipeRegistry,
} from "@magnitudedev/icn/recipes"

const HubLfs = Schema.Struct({
  size: Schema.Number,
  sha256: Schema.String,
})
const HubSibling = Schema.Struct({
  rfilename: Schema.String,
  size: Schema.optionalWith(Schema.Number, { as: "Option", exact: true }),
  blobId: Schema.optionalWith(Schema.String, { as: "Option", exact: true }),
  lfs: Schema.optionalWith(HubLfs, { as: "Option", exact: true }),
})
const HubModel = Schema.Struct({
  id: Schema.String,
  sha: Schema.optionalWith(Schema.String, { as: "Option", exact: true }),
  tags: Schema.optionalWith(Schema.Array(Schema.String), { as: "Option", exact: true }),
  cardData: Schema.optionalWith(Schema.Struct({
    license: Schema.optionalWith(Schema.String, { as: "Option", exact: true }),
    license_link: Schema.optionalWith(Schema.String, { as: "Option", exact: true }),
  }), { as: "Option", exact: true }),
  siblings: Schema.optionalWith(Schema.Array(HubSibling), { as: "Option", exact: true }),
})
type HubModel = typeof HubModel.Type
const decodeHubModel = Schema.decodeUnknownPromise(HubModel)

const cache = new Map<string, Promise<HubModel>>()
const fetchLatest = (repository: string, blobs = false): Promise<HubModel> => {
  const key = `${repository}:${blobs}`
  const existing = Option.fromNullable(cache.get(key))
  if (Option.isSome(existing)) return existing.value
  const request = (async () => {
    const response = await fetch(`https://huggingface.co/api/models/${repository}${blobs ? "?blobs=true" : ""}`)
    if (!response.ok) throw new Error(`${repository}: Hugging Face returned HTTP ${response.status}`)
    return decodeHubModel(await response.json())
  })()
  cache.set(key, request)
  return request
}

const structuralIssues = validateModelRecipeRegistry(MODEL_RECIPE_REGISTRY)
if (structuralIssues.length > 0) throw new Error(structuralIssues.join("\n"))

const failures: string[] = []
const snapshots = new Map<string, HubModel>()
const modelRepositories = [...new Set(MODEL_RECIPE_REGISTRY.models.map(({ modelRepository }) => modelRepository))]
const artifactRepositories = [...new Set(MODEL_RECIPE_REGISTRY.models.flatMap(({ artifacts }) =>
  artifacts.map(({ repository }) => repository)))]

await Promise.all(modelRepositories.map(async (repository) => {
  try {
    const remote = await fetchLatest(repository)
    if (!Option.exists(remote.sha, (sha) => /^[a-f0-9]{40,64}$/.test(sha))) {
      failures.push(`${repository}: current revision is missing or invalid`)
    }
  } catch (cause) {
    failures.push(cause instanceof Error ? cause.message : String(cause))
  }
}))

await Promise.all(artifactRepositories.map(async (repository) => {
  try {
    const remote = await fetchLatest(repository, true)
    snapshots.set(repository, remote)
    if (!Option.exists(remote.sha, (sha) => /^[a-f0-9]{40,64}$/.test(sha))) {
      failures.push(`${repository}: current revision is missing or invalid`)
    }
    if (!Option.exists(remote.siblings, (siblings) =>
      siblings.some(({ rfilename }) => rfilename.toLowerCase().endsWith(".gguf")))) {
      failures.push(`${repository}: repository has no GGUF files`)
    }
  } catch (cause) {
    failures.push(cause instanceof Error ? cause.message : String(cause))
  }
}))

for (const model of MODEL_RECIPE_REGISTRY.models) {
  for (const candidate of model.artifacts) {
    const remote = Option.fromNullable(snapshots.get(candidate.repository))
    if (Option.isNone(remote) || Option.isNone(remote.value.sha)) continue
    const license = Option.orElse(
      Option.flatMap(remote.value.cardData, (card) => card.license),
      () => Option.flatMap(
        remote.value.tags,
        (tags) => Option.map(
          Arr.findFirst(tags, (tag) => tag.startsWith("license:")),
          (tag) => tag.slice("license:".length),
        ),
      ),
    )
    const mismatchedLicense = Option.filter(
      license,
      (value) => value !== "other" && value !== model.licenseReview.expectedId,
    )
    if (Option.isSome(mismatchedLicense)) {
      failures.push(`${candidate.id}: Hub license ${mismatchedLicense.value} differs from reviewed ${model.licenseReview.expectedId}`)
    }
    const entry = resolveModelRecipeArtifact(model, candidate, {
      repository: remote.value.id,
      commit: remote.value.sha.value,
      license,
      licenseUrl: Option.flatMap(remote.value.cardData, (card) => card.license_link),
      ggufFiles: Option.getOrElse(remote.value.siblings, () => []).map(({ rfilename, size, lfs }) => ({
        path: rfilename,
        sizeBytes: Option.orElse(Option.map(lfs, (value) => value.size), () => size),
      })),
    })
    if (Option.isNone(entry)) {
      failures.push(`${candidate.id}: selector ${candidate.filenameIncludes} did not resolve uniquely`)
      continue
    }
    const primary = Option.flatMap(
      remote.value.siblings,
      (siblings) => Arr.findFirst(siblings, ({ rfilename }) => rfilename === entry.value.primaryGguf),
    )
    const size = Option.flatMap(
      primary,
      (value) => Option.orElse(Option.map(value.lfs, (lfs) => lfs.size), () => value.size),
    )
    const identity = Option.flatMap(
      primary,
      (value) => Option.orElse(Option.map(value.lfs, (lfs) => lfs.sha256), () => value.blobId),
    )
    if (Option.isNone(size) || Option.isNone(identity)) {
      failures.push(`${candidate.id}: selected primary lacks live size or content identity`)
    }
  }
}

if (failures.length > 0) {
  console.error(`Model recipe validation failed:\n${failures.map((failure) => `- ${failure}`).join("\n")}`)
  process.exit(1)
}

const artifactCount = MODEL_RECIPE_REGISTRY.models.reduce((count, model) => count + model.artifacts.length, 0)
console.log(`Validated ${MODEL_RECIPE_REGISTRY.models.length} model overlays and ${artifactCount} artifact selectors against current Hugging Face repository state.`)
