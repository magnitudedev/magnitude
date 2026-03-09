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
// QuickJS WASM patching
// =============================================================================

/**
 * The QuickJS emscripten module loads its WASM file via fs.readFileSync(__dirname + "/...").
 * Inside a compiled Bun binary, __dirname resolves to the virtual filesystem (/$bunfs/root/)
 * where the WASM file doesn't exist (it's loaded dynamically, not via require/import).
 *
 * We patch the emscripten module to embed the WASM binary inline as base64,
 * using the `wasmBinary` option that emscripten checks before filesystem loading.
 */

const QUICKJS_WASM_DIR = 'node_modules/@jitl/quickjs-wasmfile-release-asyncify/dist'
const QUICKJS_WASM_FILE = QUICKJS_WASM_DIR + '/emscripten-module.wasm'

// Both CJS and MJS variants need patching — Bun may resolve either depending on import context.
// The variable name for wasmBinary differs: `z` in CJS, `D` in MJS.
const QUICKJS_EMSCRIPTEN_MODULES = [
  { file: QUICKJS_WASM_DIR + '/emscripten-module.cjs', pattern: 'var z=c.wasmBinary', varName: 'z' },
  { file: QUICKJS_WASM_DIR + '/emscripten-module.mjs', pattern: 'var D=c.wasmBinary', varName: 'D' },
]

async function patchQuickJSWasm(): Promise<void> {
  const wasmPath = resolve(PROJECT_ROOT, QUICKJS_WASM_FILE)

  // Read WASM and base64 encode
  const wasmBinary = await readFile(wasmPath)
  const wasmBase64 = wasmBinary.toString('base64')

  for (const mod of QUICKJS_EMSCRIPTEN_MODULES) {
    const modPath = resolve(PROJECT_ROOT, mod.file)
    if (!existsSync(modPath)) continue

    // Backup original
    await copyFile(modPath, modPath + '.bak')

    // Read and patch
    let content = await readFile(modPath, 'utf-8')
    if (!content.includes(mod.pattern)) {
      console.warn('  [patch] WARNING: Could not find "' + mod.pattern + '" in ' + mod.file + ', skipping')
      continue
    }

    const patched = 'var ' + mod.varName + '=c.wasmBinary||Buffer.from("' + wasmBase64 + '","base64")'
    content = content.replace(mod.pattern, patched)
    await writeFile(modPath, content)
    console.log('  [patch] ' + mod.file + ' (embedded ' + (wasmBinary.length / 1024).toFixed(0) + 'KB WASM)')
  }
}

async function restoreQuickJSWasm(): Promise<void> {
  for (const mod of QUICKJS_EMSCRIPTEN_MODULES) {
    const modPath = resolve(PROJECT_ROOT, mod.file)
    if (existsSync(modPath + '.bak')) {
      await copyFile(modPath + '.bak', modPath)
      const { unlink } = await import('fs/promises')
      await unlink(modPath + '.bak').catch(() => {})
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

  const binDir = resolve(PROJECT_ROOT, 'bin')
  if (!existsSync(binDir)) await mkdir(binDir)

  // Patch NAPI-RS loaders and QuickJS WASM for compiled binary compatibility
  await patchNativeBindings(platform, arch)
  await patchQuickJSWasm()

  try {
    const entrypoint = resolve(import.meta.dir, '..', 'src', 'index.tsx')
    await $`bun build ${entrypoint} --compile --target=${target} --outfile=${binaryFile} --external electron --external chromium-bidi`
  } finally {
    // Always restore original files
    await restoreNativeBindings()
    await restoreQuickJSWasm()
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
