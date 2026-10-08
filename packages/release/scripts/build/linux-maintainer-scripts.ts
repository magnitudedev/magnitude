/** Dpkg calls the installed postinst after restoring an aborted upgrade or removal. */
export const renderLinuxMaintainerScripts = (begin: string, end: string): Readonly<Record<string, string>> => ({
  preinst: `case "$1" in install|upgrade) ;; *) exit 0 ;; esac\n${begin}`,
  postinst: `case "$1" in configure|abort-upgrade|abort-remove) ;; *) exit 0 ;; esac\n[ -f /var/lib/magnitude-desktop/installing ] || exit 0\n${end}`,
  prerm: `case "$1" in remove) ;; *) exit 0 ;; esac\n${begin}`,
  postrm: `case "$1" in remove|purge) ;; *) exit 0 ;; esac\n[ -f /var/lib/magnitude-desktop/installation.lock ] || exit 0\n${end}`,
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
    scriptlet("post_install", `${begin}\n${end}`),
    scriptlet("post_upgrade", completion),
    scriptlet("post_remove", completion),
  ].join("\n")
}
