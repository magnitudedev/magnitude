import { LINUX_DESKTOP_PACKAGE_NAME } from "../../src/executables"
import { linuxSudoersPath } from "./linux-system-files"

/**
 * Removing the package removes the update helper an unconverted server's sudoers rule names, so removal
 * (not upgrade) deletes the rule too.
 */
export const removeServerGrant = `rm -f ${linuxSudoersPath}\n`

/**
 * Installing over a server set up before the CLI-only install only writes its converted unit: this script
 * may run inside that server's own update, so it stops and moves nothing. The unit's next start converts it.
 */
export const writeConvertedServerUnit = `/usr/lib/${LINUX_DESKTOP_PACKAGE_NAME}/resources/magnitude _convert-server --unit || :\n`

/** The packaged unit may have changed, so systemd reloads it before anything starts it again. */
export const reloadSystemd = "if [ -d /run/systemd/system ]; then systemctl daemon-reload || :; fi\n"

/**
 * The `.desktop` entry handles `magnitude://` links. Distributions refresh the MIME cache through
 * package-manager triggers when desktop-file-utils is installed; this also covers systems without them.
 */
export const refreshDesktopDatabase = "if command -v update-desktop-database >/dev/null 2>&1; then update-desktop-database -q /usr/share/applications || :; fi\n"

/**
 * Dpkg calls the installed postinst after restoring an aborted upgrade or removal. Removal keeps
 * the installation gate so a reinstall reuses its lock inode; purge deletes it with the package.
 */
export const renderLinuxMaintainerScripts = (begin: string, end: string): Readonly<Record<string, string>> => ({
  preinst: `case "$1" in install|upgrade) ;; *) exit 0 ;; esac\n${begin}`,
  postinst: `case "$1" in configure|abort-upgrade|abort-remove) ;; *) exit 0 ;; esac\nif [ "$1" = configure ]; then ${writeConvertedServerUnit.trimEnd()}; fi\n${reloadSystemd}${refreshDesktopDatabase}[ -f /var/lib/magnitude-desktop/installing ] || exit 0\n${end}`,
  prerm: `case "$1" in remove) ;; *) exit 0 ;; esac\n${begin}`,
  postrm: `case "$1" in remove|purge) ;; *) exit 0 ;; esac\n${removeServerGrant}${reloadSystemd}${refreshDesktopDatabase}[ -f /var/lib/magnitude-desktop/installation.lock ] || exit 0\n${end}\nif [ "$1" = purge ]; then rm -rf /var/lib/magnitude-desktop; fi\n`,
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
    scriptlet("post_install", `${begin}\n${end}\n${writeConvertedServerUnit}${reloadSystemd}${refreshDesktopDatabase}`),
    scriptlet("post_upgrade", `${writeConvertedServerUnit}${reloadSystemd}${refreshDesktopDatabase}${completion}`),
    scriptlet("post_remove", `${removeServerGrant}${reloadSystemd}${refreshDesktopDatabase}${completion}`),
  ].join("\n")
}
