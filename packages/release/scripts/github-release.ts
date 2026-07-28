export interface GithubReleaseAsset {
  readonly id: number
  readonly name: string
  readonly size: number
  readonly digest?: string | null
  readonly url: string
}

export interface GithubRelease {
  readonly id: number
  readonly draft: boolean
  readonly tag_name: string
  readonly target_commitish: string
  readonly upload_url: string
  readonly assets: readonly GithubReleaseAsset[]
}

const request = async <A>(
  repository: string,
  token: string,
  path: string,
): Promise<A | undefined> => {
  const response = await fetch(`https://api.github.com${path}`, {
    headers: {
      accept: "application/vnd.github+json",
      authorization: `Bearer ${token}`,
      "x-github-api-version": "2022-11-28",
    },
    signal: AbortSignal.timeout(30_000),
  })
  if (response.status === 404) return undefined
  if (!response.ok) {
    throw new Error(`GitHub release lookup returned HTTP ${response.status}`)
  }
  return await response.json() as A
}

export const findGithubRelease = async (
  repository: string,
  token: string,
  tag: string,
  sourceCommit: string,
): Promise<GithubRelease | undefined> => {
  const published = await request<GithubRelease>(
    repository,
    token,
    `/repos/${repository}/releases/tags/${encodeURIComponent(tag)}`,
  )
  if (published) return published

  const listed = await request<readonly GithubRelease[]>(
    repository,
    token,
    `/repos/${repository}/releases?per_page=100`,
  )
  const drafts = (listed ?? []).filter(
    (release) => release.draft && release.tag_name === tag,
  )
  if (drafts.length > 1) {
    throw new Error(`GitHub has multiple drafts for ${tag}`)
  }
  const draft = drafts[0]
  if (draft && draft.target_commitish !== sourceCommit) {
    throw new Error(`GitHub draft for ${tag} targets a different commit`)
  }
  return draft
}
