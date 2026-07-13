import { writeFile, mkdir, copyFile, readFile, unlink } from 'fs/promises'
import { resolve } from 'path'
import { existsSync } from 'fs'
import { $ } from 'bun'
import { getDefaultBunTarget, getTargetInfo } from '../../scripts/release-target'

// =============================================================================
// Native binding patching
// =============================================================================

/**
 * NAPI-RS native binding loaders use `createRequire(import.meta.url)` which
 * breaks inside compiled Bun binaries (virtual filesystem /$bunfs/root/).
 *
 * We patch these loaders before building to use direct `require()` calls
 * with the platform-specific package name, which Bun can resolve and embed.
 */

const PROJECT_ROOT = resolve(import.meta.dir, '../..')

/**
 * A build-time patch that modifies a file before compilation and restores it after.
 * - `file`: path relative to PROJECT_ROOT, or a function that resolves one dynamically.
 * - `patch`: receives (originalContent, platform, arch) and returns the patched content.
 */
interface BuildPatch {
  file: string | (() => string)
  patch: (content: string, platform: string, arch: string) => string
}

function getOpentuiNativePackage(platform: string, arch: string): string {
  const p = platform === 'windows' ? 'win32' : platform
  return `@opentui/core-${p}-${arch}`
}

function findOpentuiIndexFile(): string {
  const dir = resolve(PROJECT_ROOT, 'node_modules/@opentui/core')
  const files = require('fs').readdirSync(dir) as string[]
  const match = files.find((f: string) => f.startsWith('index-') && f.endsWith('.js'))
  if (!match) throw new Error('[patch] Could not find @opentui/core/index-*.js')
  return 'node_modules/@opentui/core/' + match
}

const BUILD_PATCHES: BuildPatch[] = [
  {
    file: findOpentuiIndexFile,
    patch: (content, platform, arch) =>
      content.replace(
        'var module = await import(`@opentui/core-${process.platform}-${process.arch}/index.ts`);',
        'var module = await import("' + getOpentuiNativePackage(platform, arch) + '/index.ts");',
      ),
  },
]

async function applyBuildPatches(platform: string, arch: string): Promise<void> {
  for (const p of BUILD_PATCHES) {
    const file = typeof p.file === 'function' ? p.file() : p.file
    const resolved = resolve(PROJECT_ROOT, file)
    await copyFile(resolved, resolved + '.bak')
    const original = await readFile(resolved, 'utf-8')
    const patched = p.patch(original, platform, arch)
    await writeFile(resolved, patched)
    console.log('  [patch] ' + file)
  }
}

async function restoreBuildPatches(): Promise<void> {
  for (const p of BUILD_PATCHES) {
    const file = typeof p.file === 'function' ? p.file() : p.file
    const resolved = resolve(PROJECT_ROOT, file)
    if (existsSync(resolved + '.bak')) {
      await copyFile(resolved + '.bak', resolved)
      await unlink(resolved + '.bak').catch(() => {})
    }
  }
}

// =============================================================================
// Build
// =============================================================================

export async function buildCliBinary(target: string, outDir = resolve(PROJECT_ROOT, 'release')): Promise<string> {
  const info = getTargetInfo(target)
  const ext = info.executableExt
  const binaryFile = resolve(PROJECT_ROOT, 'bin', 'magnitude' + ext)
  const tarballName = `magnitude-${info.platformKey}.tar.gz`
  const tarballPath = resolve(outDir, tarballName)

  console.log('[build:cli] Building ' + target)

  console.log('Generating Magnitude runtime version...')
  await $`bun run ${resolve(PROJECT_ROOT, 'packages/version/scripts/generate-version.ts')}`

  const binDir = resolve(PROJECT_ROOT, 'bin')
  if (!existsSync(binDir)) await mkdir(binDir)
  await mkdir(outDir, { recursive: true })

  // Patch NAPI-RS loaders for compiled binary compatibility (CLI only)
  await applyBuildPatches(info.platform, info.arch)

  try {
    const entrypoint = resolve(import.meta.dir, '..', 'src', 'index.tsx')

    await $`bun build ${entrypoint} --compile --target=${target} --outfile=${binaryFile} --external electron --external chromium-bidi`
  } finally {
    // Always restore original files
    await restoreBuildPatches()
  }

  // Ad-hoc codesign macOS binaries to prevent Gatekeeper "damaged" warnings
  if (info.platform === 'darwin') {
    await $`codesign --force --deep --sign - ${binaryFile}`
    console.log('  [codesign] ' + binaryFile)
  }

  const entryName = 'magnitude' + ext
  await $`tar -czf ${tarballPath} -C ${binDir} ${entryName}`

  console.log('[build:cli] Built ' + binaryFile)
  console.log('[build:cli] Created ' + tarballPath)
  return tarballPath
}

if (import.meta.main) {
  const targetArg = process.argv[2] ?? getDefaultBunTarget()
  const outDirArg = process.argv[3] ? resolve(process.argv[3]) : undefined
  await buildCliBinary(targetArg, outDirArg)
}
