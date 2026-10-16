import { LINUX_DESKTOP_PACKAGE_NAME } from "../../src/executables"
import { linuxSudoersPath } from "./linux-system-files"

/**
 * Removing the package removes the update helper the server's sudoers rule names, so removal (not
 * upgrade) deletes the rule too; `magnitude server remove` still cleans up the rest.
 */
export const removeServerGrant = `rm -f ${linuxSudoersPath}\n`

/**
 * Installing or upgrading over a configured server writes the current rule again, through the CLI's
 * own account check and visudo validation. A failure leaves the package installed; the CLI says why.
 */
export const restoreServerGrant = `/usr/lib/${LINUX_DESKTOP_PACKAGE_NAME}/resources/magnitude _server-restore-grant || :\n`

/** The packaged unit may have changed, so systemd reloads it before anything starts it again. */
export const reloadSystemd = "if [ -d /run/systemd/system ]; then systemctl daemon-reload || :; fi\n"

/**
 * The `.desktop` entry handles `magnitude://` links. Distributions refresh the MIME cache through
 * package-manager triggers when desktop-file-utils is installed; this also covers systems without them.
 */
export const refreshDesktopDatabase = "if command -v update-desktop-database >/dev/null 2>&1; then update-desktop-database -q /usr/share/applications || :; fi\n"

/**
 * Purging also forgets the server, so a later installation asks again instead of bringing it back.
 * The `magnitude` account and the data directory stay, as `server remove` keeps the data, and setup
 * reuses both.
 */
export const forgetServer = "rm -f /etc/magnitude/server /etc/systemd/system/multi-user.target.wants/magnitude.service\n"
  + "rmdir /etc/magnitude 2>/dev/null || :\n"

/**
 * Dpkg calls the installed postinst after restoring an aborted upgrade or removal. Removal keeps
 * the installation gate so a reinstall reuses its lock inode; purge deletes it with the package.
 */
export const renderLinuxMaintainerScripts = (begin: string, end: string): Readonly<Record<string, string>> => ({
  preinst: `case "$1" in install|upgrade) ;; *) exit 0 ;; esac\n${begin}`,
  postinst: `case "$1" in configure|abort-upgrade|abort-remove) ;; *) exit 0 ;; esac\nif [ "$1" = configure ]; then ${restoreServerGrant.trimEnd()}; fi\n${reloadSystemd}${refreshDesktopDatabase}[ -f /var/lib/magnitude-desktop/installing ] || exit 0\n${end}`,
  prerm: `case "$1" in remove) ;; *) exit 0 ;; esac\n${begin}`,
  postrm: `case "$1" in remove|purge) ;; *) exit 0 ;; esac\n${removeServerGrant}${reloadSystemd}${refreshDesktopDatabase}if [ "$1" = purge ]; then\n${forgetServer}fi\n[ -f /var/lib/magnitude-desktop/installation.lock ] || exit 0\n${end}\nif [ "$1" = purge ]; then rm -rf /var/lib/magnitude-desktop; fi\n`,
})

/**
 * Pacman ignores scriptlet failures, so the admission check runs in a PreTransaction hook instead.
 * A first installation has no hook on disk yet and acquires and completes admission here. Removal
 * deletes the hook's own files, so completion is a scriptlet, which pacman keeps for post_remove.
 */
export const renderPacmanInstallScript = (begin: string, end: string): string => {
  const completion = `[ -f /var/lib/magnitude-desktop/installing ] || exit 0\n${end}`
  const scriptlet = (name: string, body: string) => `${name}() (\nset -eu\n${body.trimEnd()}\n)\n`
  return [
    scriptlet("post_install", `${begin}\n${end}\n${restoreServerGrant}${reloadSystemd}${refreshDesktopDatabase}`),
    scriptlet("post_upgrade", `${restoreServerGrant}${reloadSystemd}${refreshDesktopDatabase}${completion}`),
    scriptlet("post_remove", `${removeServerGrant}${reloadSystemd}${refreshDesktopDatabase}${completion}`),
  ].join("\n")
}
