import { createRequire } from "node:module"
import { mkdtemp, readFile, rm, symlink, writeFile } from "node:fs/promises"
import { tmpdir } from "node:os"
import { join } from "node:path"
import { Effect } from "effect"
import { describe, expect, it } from "vitest"
import { authorizeMacCliLink } from "./mac-cli-authorization"

const addon = new URL(`../../dist/native/darwin-${process.arch}/desktop-host.node`, import.meta.url).pathname
// These native checks intentionally never request administrator authorization.
describe.skipIf(process.platform !== "darwin")("native Mac CLI authorization boundary", () => {
  it("rejects malformed operations before requesting authorization", () => {
    const native = createRequire(import.meta.url)(addon)
    expect(() => native.configureCliLink("relative", "/Applications/Magnitude.app/Contents/Resources/magnitude", false)).toThrow()
    expect(() => native.configureCliLink("/usr/local/bin/other", "/Applications/Magnitude.app/Contents/Resources/magnitude", false)).toThrow()
  })
  it("preserves another owner's link without prompting", async () => {
    const root = await mkdtemp(join(tmpdir(), "magnitude-native-cli-"))
    try {
      const other = join(root, "other")
      const link = join(root, "magnitude")
      await writeFile(other, "preserve")
      await symlink(other, link)
      await Effect.runPromise(authorizeMacCliLink(addon, link, "/Applications/Magnitude.app/Contents/Resources/magnitude", true))
      expect(await readFile(link, "utf8")).toBe("preserve")
    } finally { await rm(root, { recursive: true, force: true }) }
  })
})
