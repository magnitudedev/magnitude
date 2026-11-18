import { FileSystem } from "@effect/platform"
import { Effect } from "effect"

/**
 * The Linux server that `magnitude server setup` created before the CLI-only install: a `magnitude`
 * system account serving `/var/lib/magnitude`, marked by `/etc/magnitude/server`. The desktop package
 * converts it into a CLI-only install under that account the next time its service starts.
 */
export const LEGACY_SERVER_MARKER = "/etc/magnitude/server"
export const LEGACY_SERVER_HOME = "/var/lib/magnitude"
export const LEGACY_SERVER_USER = "magnitude"
export const LEGACY_SERVER_SUDOERS = "/etc/sudoers.d/magnitude"
export const LEGACY_SERVER_UNIT = "/etc/systemd/system/magnitude.service"
export const LEGACY_SERVER_WRAPPER = "/usr/local/bin/magnitude"

/** True while a server from before the CLI-only install waits for its conversion. */
export const isLegacyServerMarked = (platform: NodeJS.Platform) => platform !== "linux" ? Effect.succeed(false)
  : Effect.flatMap(FileSystem.FileSystem, fs => fs.exists(LEGACY_SERVER_MARKER)).pipe(Effect.orElseSucceed(() => false))

/**
 * The converted service: it serves the account's own profile through the CLI-only command, and first runs
 * the package's conversion as root (`+`), tolerating its absence once the package is removed (`-`).
 * `/usr/bin/env` keeps SELinux (Fedora) running it as an unconfined service.
 */
export const legacyServerUnit = `# Written by the Magnitude desktop package to convert the Magnitude server into a CLI-only install.
[Unit]
Description=Magnitude server
Documentation=https://docs.magnitude.dev/remote-server
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=${LEGACY_SERVER_USER}
Group=${LEGACY_SERVER_USER}
Environment=HOME=${LEGACY_SERVER_HOME}
ExecStartPre=+-/usr/bin/env /usr/lib/magnitude-desktop/resources/magnitude _convert-server
ExecStart=/usr/bin/env ${LEGACY_SERVER_HOME}/.magnitude/bin/magnitude serve
Restart=always
RestartSec=10
KillMode=control-group
UMask=0007

[Install]
WantedBy=multi-user.target
`
