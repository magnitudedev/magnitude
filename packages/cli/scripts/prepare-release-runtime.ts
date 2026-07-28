import { resolve } from "node:path"

const trustedKeys = process.env.MAGNITUDE_RELEASE_TRUSTED_KEYS_JSON?.trim()
if (!trustedKeys) {
  throw new Error("MAGNITUDE_RELEASE_TRUSTED_KEYS_JSON is required when packing the CLI")
}
JSON.parse(trustedKeys)

const result = await Bun.build({
  entrypoints: [resolve(import.meta.dir, "../../release/src/launcher.ts")],
  outdir: resolve(import.meta.dir, "../lib"),
  naming: "release-runtime.cjs",
  format: "cjs",
  target: "node",
  minify: true,
  define: {
    MAGNITUDE_RELEASE_TRUSTED_KEYS_JSON: JSON.stringify(trustedKeys),
  },
})

if (!result.success) {
  throw new AggregateError(result.logs, "failed to bundle the authenticated release runtime")
}
