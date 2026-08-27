import { chmod } from "node:fs/promises"
import { dirname, resolve } from "node:path"
import { fileURLToPath } from "node:url"

const scriptDirectory = dirname(fileURLToPath(import.meta.url))

/**
 * The published bin file must be runnable by whatever the user has: sh execs
 * into node or bun, then the same bytes parse as CommonJS.
 */
const POLYGLOT_HEADER = `#!/bin/sh
':' //; if command -v node >/dev/null 2>&1; then exec node "$0" "$@"; fi
':' //; if command -v bun >/dev/null 2>&1; then exec bun "$0" "$@"; fi
':' //; echo "Magnitude requires Node.js or Bun to start." >&2; exit 127
`

export const buildLauncher = async (outdir: string): Promise<string> => {
  const result = await Bun.build({
    entrypoints: [resolve(scriptDirectory, "../src/main.ts")],
    outdir,
    naming: "magnitude.js",
    format: "cjs",
    target: "node",
    minify: true,
    banner: POLYGLOT_HEADER,
  })
  if (!result.success) {
    throw new AggregateError(result.logs, "failed to build the Magnitude launcher")
  }
  const outfile = resolve(outdir, "magnitude.js")
  await chmod(outfile, 0o755)
  return outfile
}

if (import.meta.main) {
  await buildLauncher(resolve(scriptDirectory, "../bin"))
}
