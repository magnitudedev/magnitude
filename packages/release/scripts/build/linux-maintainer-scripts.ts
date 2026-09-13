/** Dpkg calls the installed postinst after restoring an aborted upgrade or removal. */
export const renderLinuxMaintainerScripts = (begin: string, end: string): Readonly<Record<string, string>> => ({
  preinst: `case "$1" in install|upgrade) ;; *) exit 0 ;; esac\n${begin}`,
  postinst: `case "$1" in configure|abort-upgrade|abort-remove) ;; *) exit 0 ;; esac\n[ -f /var/lib/magnitude-desktop/installing ] || exit 0\n${end}`,
  prerm: `case "$1" in remove) ;; *) exit 0 ;; esac\n${begin}`,
  postrm: `case "$1" in remove|purge) ;; *) exit 0 ;; esac\n[ -f /var/lib/magnitude-desktop/installation.lock ] || exit 0\n${end}`,
})
