import { chmod } from "node:fs/promises"
import { dirname, resolve } from "node:path"
import { fileURLToPath } from "node:url"

const scriptDirectory = dirname(fileURLToPath(import.meta.url))

// npm generates native Windows command shims from this Node entry point.
export const buildLauncher = async (outdir: string): Promise<string> => {
  const result = await Bun.build({
    entrypoints: [resolve(scriptDirectory, "../src/main.ts")],
    outdir,
    naming: "magnitude.js",
    format: "cjs",
    target: "node",
    minify: true,
    banner: "#!/usr/bin/env node",
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
