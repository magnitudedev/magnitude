import { writeFile, chmod, mkdir, copyFile, readFile } from 'fs/promises'
import { dirname, join, resolve } from 'path'
import { existsSync } from 'fs'
import { $ } from 'bun'

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

interface PatchTarget {
  file: string
  getContent: (platform: string, arch: string) => string
}

const PROJECT_ROOT = resolve(import.meta.dir, '../..')

const PATCH_TARGETS: PatchTarget[] = [
  {
    file: 'node_modules/oxc-parser/src-js/bindings.js',
    getContent: (platform, arch) => [
      'const nativeBinding = require("@oxc-parser/binding-' + platform + '-' + arch + '");',
      'export const { Severity, ParseResult, ExportExportNameKind, ExportImportNameKind, ExportLocalNameKind, ImportNameKind, parse, parseSync, rawTransferSupported, getBufferOffset, parseRaw, parseRawSync } = nativeBinding;',
    ].join('\n'),
  },
  {
    file: 'node_modules/@boundaryml/baml/native.js',
    getContent: (platform, arch) => [
      'const nativeBinding = require("@boundaryml/baml-' + platform + '-' + arch + '");',
      'module.exports = nativeBinding;',
    ].join('\n'),
  },
]

async function patchNativeBindings(platform: string, arch: string): Promise<void> {
  for (const target of PATCH_TARGETS) {
    const resolved = resolve(PROJECT_ROOT, target.file)
    await copyFile(resolved, resolved + '.bak')
    await writeFile(resolved, target.getContent(platform, arch))
    console.log('  [patch] ' + target.file)
  }
}

async function restoreNativeBindings(): Promise<void> {
  for (const target of PATCH_TARGETS) {
    const resolved = resolve(PROJECT_ROOT, target.file)
    if (existsSync(resolved + '.bak')) {
      await copyFile(resolved + '.bak', resolved)
      const { unlink } = await import('fs/promises')
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

const QUICKJS_WASM_DIR = 'node_modules/@jitl/quickjs-wasmfile-release-asyncify/dist'

const WASM_PATCH_TARGETS: WasmPatchTarget[] = [
  {
    wasmFile: QUICKJS_WASM_DIR + '/emscripten-module.wasm',
    modules: [
      {
        file: QUICKJS_WASM_DIR + '/emscripten-module.cjs',
        pattern: 'var z=c.wasmBinary',
        getReplacement: (b64) => 'var z=c.wasmBinary||Buffer.from("' + b64 + '","base64")',
      },
      {
        file: QUICKJS_WASM_DIR + '/emscripten-module.mjs',
        pattern: 'var D=c.wasmBinary',
        getReplacement: (b64) => 'var D=c.wasmBinary||Buffer.from("' + b64 + '","base64")',
      },
    ],
  },
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
        const { unlink } = await import('fs/promises')
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

  const binDir = resolve(PROJECT_ROOT, 'bin')
  if (!existsSync(binDir)) await mkdir(binDir)

  // Patch NAPI-RS loaders and QuickJS WASM for compiled binary compatibility
  await patchNativeBindings(platform, arch)
  await patchWasmModules()

  try {
    const entrypoint = resolve(import.meta.dir, '..', 'src', 'index.tsx')
    await $`bun build ${entrypoint} --compile --target=${target} --outfile=${binaryFile} --external electron --external chromium-bidi`
  } finally {
    // Always restore original files
    await restoreNativeBindings()
    await restoreWasmModules()
  }

  console.log('Built ' + binaryFile)

  // Create launcher wrapper that sets BUN_JSC_useOMGJIT=0
  // This prevents pathological WASM JIT compilation of the Asyncified QuickJS binary
  // See: bugs/26-02-14/cpu-spin-investigation.md
  if (isWindows) {
    const wrapperFile = binaryFile.replace('.exe', '-launcher.cmd')
    const wrapperContent = [
      '@echo off',
      'set BUN_JSC_useOMGJIT=0',
      '"%~dp0\\' + binaryFile.split('/').pop() + '" %*',
    ].join('\r\n')
    await writeFile(wrapperFile, wrapperContent)
    console.log('Built ' + wrapperFile + ' (launcher)')
  } else {
    const wrapperFile = binaryFile + '-launcher'
    const wrapperContent = [
      '#!/bin/sh',
      '# Disable JSC OMG JIT to prevent pathological WASM compilation',
      'export BUN_JSC_useOMGJIT=0',
      'exec "$(dirname "$0")/' + binaryFile.split('/').pop() + '" "$@"',
    ].join('\n')
    await writeFile(wrapperFile, wrapperContent)
    await chmod(wrapperFile, 0o755)
    console.log('Built ' + wrapperFile + ' (launcher)')
  }
}

if (targetArg) {
  await build(targetArg)
} else {
  const platform = process.platform === 'darwin' ? 'darwin' : process.platform === 'win32' ? 'windows' : 'linux'
  const arch = process.arch === 'arm64' ? 'arm64' : 'x64'
  await build('bun-' + platform + '-' + arch)
}
