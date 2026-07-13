import { downloadRg } from './download'
import { getTarget } from './platform'
import { resolve, dirname } from 'path'
import { fileURLToPath } from 'url'

const __dirname = dirname(fileURLToPath(import.meta.url))
const BIN_DIR = resolve(__dirname, '../bin')

async function main() {
  const target = getTarget()
  const isWin = target.includes('windows')
  const binName = isWin ? 'rg.exe' : 'rg'
  const binPath = resolve(BIN_DIR, binName)

  if (await Bun.file(binPath).exists()) {
    console.log(`[ripgrep/prepare] Binary already exists at ${binPath}, skipping download`)
    return
  }

  console.log(`[ripgrep/prepare] Downloading ripgrep for ${target}...`)
  const path = await downloadRg(BIN_DIR, target)
  console.log(`[ripgrep/prepare] Downloaded to ${path}`)
}

main().catch((err) => {
  console.error('[ripgrep/prepare] Failed:', err)
  process.exit(1)
})
