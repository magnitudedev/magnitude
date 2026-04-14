import { join, dirname, resolve } from 'node:path'
import { homedir } from 'node:os'

const SKILLSET_NAME = 'magnitude'
const DEV_SKILLSETS_ROOT = resolve(import.meta.dir, '../../..', 'skillsets', SKILLSET_NAME)

function getEmbeddedSkillsetFiles(): Array<Blob & { name: string }> {
  const files = (Bun as any).embeddedFiles as Array<Blob & { name?: string }> | undefined
  if (!files?.length) return []
  const prefix = `skillsets/${SKILLSET_NAME}/`
  return files.filter((f): f is Blob & { name: string } =>
    typeof f.name === 'string' && f.name.includes(prefix)
  )
}

/**
 * Provisions the magnitude skillset to ~/.magnitude/skillsets/magnitude/ on first run.
 * Compiled mode: reads from Bun's embedded files.
 * Dev mode: reads from skillsets/magnitude/ in the project directory.
 * Only runs if ~/.magnitude/skillsets/magnitude/ doesn't already exist.
 */
export async function provisionMagnitudeSkillset(): Promise<void> {
  const skillsetDir = join(homedir(), '.magnitude', 'skillsets', SKILLSET_NAME)
  const markerFile = join(skillsetDir, 'SKILLSET.md')

  // Already provisioned — user owns it
  if (await Bun.file(markerFile).exists()) return

  // Try embedded files first (compiled binary)
  const embedded = getEmbeddedSkillsetFiles()
  if (embedded.length > 0) {
    for (const blob of embedded) {
      const idx = blob.name.indexOf(`skillsets/${SKILLSET_NAME}/`)
      if (idx === -1) continue
      const relPath = blob.name.slice(idx + `skillsets/${SKILLSET_NAME}/`.length)
      const destPath = join(skillsetDir, relPath)
      await Bun.write(destPath, blob)
    }
  } else {
    // Dev mode — read from project source
    const glob = new Bun.Glob('**/*.md')
    for await (const relPath of glob.scan({ cwd: DEV_SKILLSETS_ROOT })) {
      const content = await Bun.file(join(DEV_SKILLSETS_ROOT, relPath)).text()
      const destPath = join(skillsetDir, relPath)
      await Bun.write(destPath, content)
    }
  }

  // Set selectedSkillset in global config if not already set
  const configPath = join(homedir(), '.magnitude', 'config.json')
  try {
    let config: Record<string, unknown> = {}
    try {
      config = JSON.parse(await Bun.file(configPath).text())
    } catch {
      // Config doesn't exist yet
    }

    if (config.selectedSkillset === undefined || config.selectedSkillset === null) {
      config.selectedSkillset = SKILLSET_NAME
      await Bun.write(configPath, JSON.stringify(config, null, 2))
    }
  } catch (err) {
    console.warn('[magnitude] Warning: could not update config with selectedSkillset:', err)
  }
}
