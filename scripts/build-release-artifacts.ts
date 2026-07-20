import { mkdir, rm, writeFile } from 'node:fs/promises'
import { basename, resolve } from 'node:path'
import { $ } from 'bun'
import { buildCliBinary } from '../cli/scripts/build-binary'
import { buildAcnBinary } from '../packages/acn/scripts/build-binary'
import { getDefaultBunTarget, getTargetInfo } from './release-target'

const PROJECT_ROOT = resolve(import.meta.dir, '..')

interface ArtifactManifestEntry {
  readonly name: string
  readonly kind: 'cli' | 'acn'
  readonly platform: string
  readonly arch: string
  readonly sha256: string
  readonly bytes: number
}

async function sha256(path: string): Promise<string> {
  const bytes = await Bun.file(path).arrayBuffer()
  const hash = await crypto.subtle.digest('SHA-256', bytes)
  return Buffer.from(hash).toString('hex')
}

async function verifyTarball(path: string, expectedEntries: ReadonlyArray<string>): Promise<void> {
  const listing = (await $`tar -tzf ${path}`.text())
    .split('\n')
    .map((line) => line.trim())
    .filter(Boolean)
  if (JSON.stringify(listing.sort()) !== JSON.stringify([...expectedEntries].sort())) {
    throw new Error(
      `[release] ${basename(path)} contains ${JSON.stringify(listing)}, expected ${JSON.stringify(expectedEntries)}`
    )
  }
}

async function manifestEntry(
  path: string,
  kind: 'cli' | 'acn',
  info: ReturnType<typeof getTargetInfo>,
): Promise<ArtifactManifestEntry> {
  const file = Bun.file(path)
  return {
    name: basename(path),
    kind,
    platform: info.platform,
    arch: info.arch,
    sha256: await sha256(path),
    bytes: file.size,
  }
}

export async function buildReleaseArtifacts(
  target: string,
  outDir = resolve(PROJECT_ROOT, 'release'),
): Promise<void> {
  const info = getTargetInfo(target)
  await rm(outDir, { recursive: true, force: true })
  await mkdir(outDir, { recursive: true })

  const acnTarball = await buildAcnBinary(target, outDir)
  const cliTarball = await buildCliBinary(target, outDir)

  await verifyTarball(acnTarball, [
    'magnitude-acn' + info.executableExt,
    'magnitude-icn' + info.executableExt,
    'magnitude-icn-manifest.json',
  ])
  await verifyTarball(cliTarball, ['magnitude' + info.executableExt])

  const artifacts = [
    await manifestEntry(cliTarball, 'cli', info),
    await manifestEntry(acnTarball, 'acn', info),
  ]

  const checksums = artifacts
    .map((artifact) => `${artifact.sha256}  ${artifact.name}`)
    .join('\n') + '\n'
  await writeFile(resolve(outDir, 'SHA256SUMS'), checksums)

  const manifest = {
    schemaVersion: 1,
    target,
    platformKey: info.platformKey,
    generatedAt: new Date().toISOString(),
    artifacts,
  }
  await writeFile(
    resolve(outDir, 'magnitude-release-manifest.json'),
    JSON.stringify(manifest, null, 2) + '\n',
  )

  console.log('[release] Wrote artifacts to ' + outDir)
}

if (import.meta.main) {
  const target = process.argv[2] ?? getDefaultBunTarget()
  const outDir = process.argv[3] ? resolve(process.argv[3]) : undefined
  await buildReleaseArtifacts(target, outDir)
}
