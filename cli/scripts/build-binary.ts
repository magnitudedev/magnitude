import { writeFile, mkdir, copyFile, readFile, unlink } from 'fs/promises'
import { resolve } from 'path'
import { existsSync } from 'fs'
import { $ } from 'bun'
import { downloadRg } from '@magnitudedev/ripgrep'

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

function getBamlNativePackage(platform: string, arch: string): string {
  if (platform === 'darwin') return `@boundaryml/baml-darwin-${arch}`
  if (platform === 'windows') return `@boundaryml/baml-win32-${arch}-msvc`
  return `@boundaryml/baml-linux-${arch}-gnu`
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
    file: 'node_modules/@boundaryml/baml/native.js',
    patch: (_content, platform, arch) =>
      'const nativeBinding = require("' + getBamlNativePackage(platform, arch) + '");\n' +
      'module.exports = nativeBinding;\n',
  },
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
// WASM embedding
// =============================================================================

/**
 * Modules that load WASM via fs.readFileSync or similar patterns break inside
 * compiled Bun binaries (virtual filesystem /$bunfs/root/). We patch the JS
 * loaders to embed WASM binaries inline as base64 before building.
 */

interface WasmPatchTarget {
  /** Path to the .wasm file to embed */
  wasmFile: string
  /** JS files to patch, each with a find/replace pattern */
  modules: {
    file: string
    pattern: string
    getReplacement: (wasmBase64: string) => string
  }[]
}

const WASM_PATCH_TARGETS: WasmPatchTarget[] = [
  {
    wasmFile: 'packages/image/pkg/magnitude_image_bg.wasm',
    modules: [
      {
        file: 'packages/image/pkg/magnitude_image.js',
        pattern: "const wasmBytes = require('fs').readFileSync(wasmPath);",
        getReplacement: (b64) => 'const wasmBytes = Buffer.from("' + b64 + '","base64");',
      },
    ],
  },
]

async function patchWasmModules(): Promise<void> {
  for (const target of WASM_PATCH_TARGETS) {
    const wasmPath = resolve(PROJECT_ROOT, target.wasmFile)
    if (!existsSync(wasmPath)) continue

    const wasmBinary = await readFile(wasmPath)
    const wasmBase64 = wasmBinary.toString('base64')

    for (const mod of target.modules) {
      const modPath = resolve(PROJECT_ROOT, mod.file)
      if (!existsSync(modPath)) continue

      await copyFile(modPath, modPath + '.bak')

      let content = await readFile(modPath, 'utf-8')
      if (!content.includes(mod.pattern)) {
        console.warn('  [patch] WARNING: Could not find "' + mod.pattern + '" in ' + mod.file + ', skipping')
        continue
      }

      content = content.replace(mod.pattern, mod.getReplacement(wasmBase64))
      await writeFile(modPath, content)
      console.log('  [patch] ' + mod.file + ' (embedded ' + (wasmBinary.length / 1024).toFixed(0) + 'KB WASM)')
    }
  }
}

async function restoreWasmModules(): Promise<void> {
  for (const target of WASM_PATCH_TARGETS) {
    for (const mod of target.modules) {
      const modPath = resolve(PROJECT_ROOT, mod.file)
      if (existsSync(modPath + '.bak')) {
        await copyFile(modPath + '.bak', modPath)
        await unlink(modPath + '.bak').catch(() => {})
      }
    }
  }
}

// =============================================================================
// Build
// =============================================================================

function getTargetPlatformArch(target: string): { platform: string; arch: string } {
  const parts = target.replace('bun-', '').split('-')
  return { platform: parts[0], arch: parts[1] }
}

function bunTargetToRipgrepTarget(bunTarget: string): string {
  const { platform, arch } = getTargetPlatformArch(bunTarget)
  const map: Record<string, Record<string, string>> = {
    darwin: { arm64: 'aarch64-apple-darwin', x64: 'x86_64-apple-darwin' },
    linux: { x64: 'x86_64-unknown-linux-musl', arm64: 'aarch64-unknown-linux-musl' },
    windows: { x64: 'x86_64-pc-windows-msvc' },
  }
  const target = map[platform]?.[arch]
  if (!target) throw new Error(`No ripgrep target for ${bunTarget}`)
  return target
}

const targets = [
  'bun-darwin-arm64',
  'bun-darwin-x64',
  'bun-linux-x64',
  'bun-linux-arm64',
  'bun-windows-x64',
] as const

const targetArg = process.argv[2]

async function build(target: string) {
  const isWindows = target.includes('windows')
  const ext = isWindows ? '.exe' : ''
  const binaryFile = resolve(PROJECT_ROOT, 'bin', 'magnitude-' + target.replace('bun-', '') + ext)
  const { platform, arch } = getTargetPlatformArch(target)

  console.log('Building ' + target + '...')

  // Build WASM dependencies
  console.log('Building @magnitudedev/image WASM...')
  await $`cd ${resolve(PROJECT_ROOT, 'packages/image')} && wasm-pack build --target nodejs --out-dir pkg --release`.quiet()

  // Download ripgrep binary for embedding
  const rgBinDir = resolve(PROJECT_ROOT, 'packages/ripgrep/bin')
  const rgTarget = bunTargetToRipgrepTarget(target)
  console.log('Downloading ripgrep (' + rgTarget + ') for ' + target + '...')
  const rgBinPath = await downloadRg(rgBinDir, rgTarget)

  const binDir = resolve(PROJECT_ROOT, 'bin')
  if (!existsSync(binDir)) await mkdir(binDir)

  // Patch NAPI-RS loaders and WASM modules for compiled binary compatibility
  await applyBuildPatches(platform, arch)
  await patchWasmModules()

  try {
    const entrypoint = resolve(import.meta.dir, '..', 'src', 'index.tsx')
    await $`bun build ${entrypoint} ${rgBinPath} --compile --target=${target} --outfile=${binaryFile} --external electron --external chromium-bidi`
  } finally {
    // Always restore original files
    await restoreBuildPatches()
    await restoreWasmModules()
  }

  // Ad-hoc codesign macOS binaries to prevent Gatekeeper "damaged" warnings
  if (target.includes('darwin')) {
    await $`codesign --force --deep --sign - ${binaryFile}`
    console.log('  [codesign] ' + binaryFile)
  }

  console.log('Built ' + binaryFile)
}

if (targetArg === '--all') {
  for (const target of targets) {
    await build(target)
  }
} else if (targetArg) {
  await build(targetArg)
} else {
  const platform = process.platform === 'darwin' ? 'darwin' : process.platform === 'win32' ? 'windows' : 'linux'
  const arch = process.arch === 'arm64' ? 'arm64' : 'x64'
  await build('bun-' + platform + '-' + arch)
}
