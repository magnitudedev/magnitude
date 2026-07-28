const version = process.env.MAGNITUDE_RELEASE_VERSION?.trim()
const expectedIntegrity = process.env.MAGNITUDE_EXPECTED_NPM_INTEGRITY?.trim()
if (!version || !expectedIntegrity) {
  throw new Error("release version and expected npm integrity are required")
}

const url =
  `https://registry.npmjs.org/%40magnitudedev%2Fcli/${encodeURIComponent(version)}`
let lastFailure = "npm registry did not expose the published version"
let verified = false
for (let attempt = 0; attempt < 20; attempt++) {
  try {
    const response = await fetch(url, { signal: AbortSignal.timeout(30_000) })
    if (response.ok) {
      const metadata = await response.json() as {
        readonly dist?: { readonly integrity?: string }
      }
      if (metadata.dist?.integrity !== expectedIntegrity) {
        throw new Error("published npm package differs from the accepted candidate")
      }
      verified = true
      break
    }
    if (response.status !== 404 && response.status < 500) {
      throw new Error(`npm registry returned HTTP ${response.status} for ${version}`)
    }
    lastFailure = `npm registry returned HTTP ${response.status} for ${version}`
  } catch (cause) {
    if (
      cause instanceof Error &&
      (
        cause.message.includes("differs from the accepted candidate") ||
        cause.message.includes("returned HTTP 4")
      )
    ) throw cause
    lastFailure = cause instanceof Error ? cause.message : lastFailure
  }
  if (attempt < 19) await Bun.sleep(3_000)
}

if (!verified) throw new Error(lastFailure)

export {}
