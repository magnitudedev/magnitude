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

  it("restores the server's sudoers rule on configure only, before the gate check can exit", () => {
    const postinst = scripts.postinst!.split("\n")
    expect(postinst[1]).toBe('if [ "$1" = configure ]; then /usr/lib/magnitude-desktop/resources/magnitude _server-restore-grant || :; fi')
    expect(postinst.findIndex(line => line.includes("exit 0") && line.includes("installing"))).toBeGreaterThan(1)
    expect(Object.entries(scripts).filter(([name]) => name !== "postinst").some(([, body]) => body.includes("_server-restore-grant"))).toBe(false)
  })

  it.each(["postinst", "postrm"])("%s reloads systemd before the gate check can exit", name => {
    const lines = scripts[name]!.split("\n")
    const reload = lines.indexOf("if [ -d /run/systemd/system ]; then systemctl daemon-reload || :; fi")
    expect(reload).toBeGreaterThan(0)
    expect(reload).toBeLessThan(lines.findIndex(line => line.includes("|| exit 0") && line.includes("/var/lib/magnitude-desktop")))
  })

  it("forgets the server only on purge, keeping its account and data", () => {
    const postrm = scripts.postrm!
    expect(postrm).toContain('if [ "$1" = purge ]; then\nrm -f /etc/magnitude/server /etc/systemd/system/multi-user.target.wants/magnitude.service\nrmdir /etc/magnitude 2>/dev/null || :\nfi\n')
    expect(postrm).not.toMatch(/userdel|groupdel|\/var\/lib\/magnitude\b(?!-desktop)/)
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
  it("restores the server's sudoers rule on install and upgrade, and reloads systemd in every scriptlet", () => {
    const [install, upgrade, remove] = ["post_install()", "post_upgrade()", "post_remove()"].map((name, index, names) =>
      script.slice(script.indexOf(name), index + 1 < names.length ? script.indexOf(names[index + 1]!) : undefined))
    for (const body of [install!, upgrade!]) expect(body).toContain("/usr/lib/magnitude-desktop/resources/magnitude _server-restore-grant || :\n")
    expect(remove).not.toContain("_server-restore-grant")
    for (const body of [install!, upgrade!, remove!]) expect(body).toContain("systemctl daemon-reload || :")
    expect(upgrade!.indexOf("daemon-reload")).toBeLessThan(upgrade!.indexOf("|| exit 0"))
  })
  it("is valid POSIX shell", () => {
    expect(() => execFileSync("sh", ["-n"], { input: script })).not.toThrow()
  })
})
