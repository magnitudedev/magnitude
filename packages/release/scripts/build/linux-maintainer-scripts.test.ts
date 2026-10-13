import { execFileSync } from "node:child_process"
import { readFileSync } from "node:fs"
import { join } from "node:path"
import { describe, expect, it } from "vitest"
import { renderLinuxMaintainerScripts } from "./linux-maintainer-scripts"

const resources = join(import.meta.dirname, "../../resources/linux")
const scripts = renderLinuxMaintainerScripts(
  readFileSync(join(resources, "installation-begin.sh"), "utf8"),
  readFileSync(join(resources, "installation-end.sh"), "utf8"),
)

describe("Debian maintainer scripts", () => {
  it.each(Object.entries(scripts))("%s is valid POSIX shell", (_, body) => {
    expect(() => execFileSync("sh", ["-n"], { input: `#!/bin/sh\nset -eu\n${body}` })).not.toThrow()
  })

  it("keeps the installation gate on remove and deletes it on purge", () => {
    const postrm = scripts.postrm!
    expect(postrm).toMatch(/^case "\$1" in remove\|purge\)/)
    expect(postrm.trimEnd().split("\n").at(-1)).toBe('if [ "$1" = purge ]; then rm -rf /var/lib/magnitude-desktop; fi')
  })
})
