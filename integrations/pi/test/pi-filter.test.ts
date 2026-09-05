import { describe, expect, it } from "vitest"
import { resolve } from "node:path"
import { DefaultPackageManager } from "@earendil-works/pi-coding-agent"
import { Schema } from "effect"
import { PI_COMPANION_EXTENSION_PATH, piPackageExtensionEnabled } from "../../../cli/src/harness-connections/connectors/pi-package"
import { PiPackageEntrySchema } from "../../../cli/src/harness-connections/connectors/pi-settings"

// Differential check against the pinned host, without its filesystem discovery.
// Only discovery and collection are substituted; Pi executes its actual filters.
describe("supported Pi package filter conformance", () => {
  const root = "/tmp/pi-filter-package"
  const file = resolve(root, PI_COMPANION_EXTENSION_PATH)
  const cases = [undefined, [], ["*.js"], ["!magnitude.js"], ["dist/*.js"], ["!dist/**"], [`-${file}`], [`+${file}`], ["!*.js", `+${PI_COMPANION_EXTENSION_PATH}`], [`+${PI_COMPANION_EXTENSION_PATH}`, "!*.js"], [`+${PI_COMPANION_EXTENSION_PATH}`, `-${PI_COMPANION_EXTENSION_PATH}`], ["other.js"], ["!other.js"]]
  for (const autoload of [true, false]) for (const extensions of cases) it(`matches autoload=${autoload}, filters=${JSON.stringify(extensions)}`, () => {
    const entry = { source: "local", autoload, ...(extensions === undefined ? {} : { extensions }) }
    let enabled = false
    const receiver = { collectManifestFiles: () => ({ allFiles: [file] }), addResource: (_target: unknown, _file: string, _metadata: unknown, value: boolean) => { enabled = value } }
    const native = DefaultPackageManager.prototype as unknown as Record<string, (...args: any[]) => void>
    if (extensions === undefined) enabled = autoload
    else native[autoload ? "applyPackageFilter" : "applyPackageDeltaFilter"]!.call(receiver, root, extensions, "extensions", new Map(), {})
    expect(piPackageExtensionEnabled(Schema.decodeUnknownSync(PiPackageEntrySchema)(entry), root)).toBe(enabled)
  })
})
