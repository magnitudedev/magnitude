import { resolve } from "node:path"

const result = await Bun.build({
  entrypoints: [resolve(import.meta.dir, "../../release/src/launcher.ts")],
  outdir: resolve(import.meta.dir, "../lib"),
  naming: "release-runtime.cjs",
  format: "cjs",
  target: "node",
  minify: true,
})

if (!result.success) {
  throw new AggregateError(result.logs, "failed to bundle the release runtime")
}
