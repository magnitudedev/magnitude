import { execFileSync } from "node:child_process"
import { readFileSync } from "node:fs"
import { join } from "node:path"
import { describe, expect, it } from "vitest"
import { renderLinuxMaintainerScripts, renderPacmanInstallScript } from "./linux-maintainer-scripts"

const resources = join(import.meta.dirname, "../../resources/linux")
const scripts = renderLinuxMaintainerScripts(
  readFileSync(join(resources, "installation-begin.sh"), "utf8"),
  readFileSync(join(resources, "installation-end.sh"), "utf8"),
)

describe("Debian maintainer scripts", () => {
  it.each(Object.entries(scripts))("%s is valid POSIX shell", (_, body) => {
    expect(() => execFileSync("sh", ["-n"], { input: `#!/bin/sh\nset -eu\n${body}` })).not.toThrow()
  })

  it("deletes the server's sudoers rule on remove and purge, before the gate check can exit, and never on upgrade", () => {
    expect(scripts.postrm!.split("\n").slice(0, 2)).toEqual(['case "$1" in remove|purge) ;; *) exit 0 ;; esac', "rm -f /etc/sudoers.d/magnitude"])
    expect(Object.entries(scripts).filter(([name]) => name !== "postrm").some(([, body]) => body.includes("sudoers"))).toBe(false)
  })

  it("keeps the installation gate on remove and deletes it on purge", () => {
    const postrm = scripts.postrm!
    expect(postrm).toMatch(/^case "\$1" in remove\|purge\)/)
    expect(postrm.trimEnd().split("\n").at(-1)).toBe('if [ "$1" = purge ]; then rm -rf /var/lib/magnitude-desktop; fi')
  })
})

describe("pacman install script", () => {
  const script = renderPacmanInstallScript(
    readFileSync(join(resources, "installation-begin.sh"), "utf8"),
    readFileSync(join(resources, "installation-end.sh"), "utf8"),
  )
  it("deletes the server's sudoers rule only when the package is removed", () => {
    const remove = script.slice(script.indexOf("post_remove()"))
    expect(remove.split("\n").slice(0, 3)).toEqual(["post_remove() (", "set -eu", "rm -f /etc/sudoers.d/magnitude"])
    expect(script.slice(0, script.indexOf("post_remove()"))).not.toContain("sudoers")
  })
})
