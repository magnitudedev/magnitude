import { mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join, resolve } from 'node:path'
import { $ } from 'bun'
import { downloadRg } from '@magnitudedev/ripgrep'
import { bunTargetToRipgrepTarget, getDefaultBunTarget, getTargetInfo } from '../../../scripts/release-target'

const PROJECT_ROOT = resolve(import.meta.dir, '../../..')
const RG_EMBED_PATH = resolve(PROJECT_ROOT, 'packages/ripgrep/src/rg-embed.ts')

async function generateRgEmbed(isWindows: boolean): Promise<() => Promise<void>> {
  const original = await readFile(RG_EMBED_PATH, 'utf8')
  const binName = isWindows ? 'rg.exe' : 'rg'
  const content = `export { default as rgPath } from "../bin/${binName}" with { type: "file" };\n`
  await writeFile(RG_EMBED_PATH, content)
  console.log('  [rg-embed] generated for ' + binName)
  return async () => {
    await writeFile(RG_EMBED_PATH, original)
  }
}

async function smokeVersion(binaryFile: string): Promise<string> {
  const proc = Bun.spawn([binaryFile, 'version'], { stdout: 'pipe', stderr: 'pipe' })
  const stdout = (await new Response(proc.stdout).text()).trim()
  const stderr = await new Response(proc.stderr).text()
  const exit = await proc.exited
  if (exit !== 0 || stdout.length === 0) {
    throw new Error(`[build:acn] version smoke failed (${exit}): ${stderr}`)
  }
  return stdout
}

async function smokeDoctor(binaryFile: string): Promise<void> {
  const home = await mkdtemp(join(tmpdir(), 'magnitude-acn-doctor-'))
  try {
    const proc = Bun.spawn([binaryFile, 'doctor'], {
      stdout: 'pipe',
      stderr: 'pipe',
      env: { ...process.env, HOME: home, USERPROFILE: home },
    })
    const stdout = await new Response(proc.stdout).text()
    const stderr = await new Response(proc.stderr).text()
    const exit = await proc.exited
    if (exit !== 0 || !stdout.includes('ripgrep')) {
      throw new Error(`[build:acn] doctor smoke failed (${exit}): ${stderr || stdout}`)
    }
  } finally {
    await rm(home, { recursive: true, force: true }).catch(() => {})
  }
}

export async function buildAcnBinary(target: string, outDir = resolve(PROJECT_ROOT, 'release')): Promise<string> {
  const info = getTargetInfo(target)
  const isWindows = info.platform === 'windows'
  const binDir = resolve(PROJECT_ROOT, 'bin')
  const binaryFile = resolve(binDir, 'magnitude-acn' + info.executableExt)
  const tarballName = `magnitude-acn-${info.platformKey}.tar.gz`
  const tarballPath = resolve(outDir, tarballName)

  console.log('[build:acn] Building ' + target)
  await mkdir(binDir, { recursive: true })
  await mkdir(outDir, { recursive: true })

  console.log('[build:acn] Generating Magnitude runtime version...')
  await $`bun run ${resolve(PROJECT_ROOT, 'packages/version/scripts/generate-version.ts')}`

  const rgBinDir = resolve(PROJECT_ROOT, 'packages/ripgrep/bin')
  const rgTarget = bunTargetToRipgrepTarget(target)
  console.log('[build:acn] Downloading ripgrep (' + rgTarget + ')...')
  await downloadRg(rgBinDir, rgTarget)

  const restoreRgEmbed = await generateRgEmbed(isWindows)
  try {
    const entrypoint = resolve(PROJECT_ROOT, 'packages/acn/src/binary.ts')
    await $`bun build ${entrypoint} --compile --target=${target} --outfile=${binaryFile}`
  } finally {
    await restoreRgEmbed()
  }

  const version = await smokeVersion(binaryFile)
  console.log('  [version] ' + version)
  await smokeDoctor(binaryFile)
  console.log('  [doctor] embedded ripgrep ok')

  if (info.platform === 'darwin') {
    await $`codesign --force --deep --sign - ${binaryFile}`
    console.log('  [codesign] ' + binaryFile)
  }

  const entryName = 'magnitude-acn' + info.executableExt
  await $`tar -czf ${tarballPath} -C ${binDir} ${entryName}`
  console.log('[build:acn] Created ' + tarballPath)
  return tarballPath
}

if (import.meta.main) {
  const targetArg = process.argv[2] ?? getDefaultBunTarget()
  const outDirArg = process.argv[3] ? resolve(process.argv[3]) : undefined
  await buildAcnBinary(targetArg, outDirArg)
}
