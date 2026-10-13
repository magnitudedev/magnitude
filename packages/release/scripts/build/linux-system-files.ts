import { dirname, join, resolve } from "node:path"
import { fileURLToPath } from "node:url"
import { LINUX_UPDATE_INSTALLER_PATH } from "../../src/executables"

const resources = resolve(dirname(fileURLToPath(import.meta.url)), "../../resources/linux/system")

/**
 * System integration shipped by every Linux desktop package. The unit and sudoers rule are inert
 * until `magnitude server setup` creates the magnitude user and enables the service; the Polkit
 * action lets the person at the desktop run the update installer wrapper, and only it, without a password.
 */
export const linuxSystemFiles = [
  { source: join(resources, "dev.magnitude.update.policy"), path: "usr/share/polkit-1/actions/dev.magnitude.update.policy", mode: 0o644 },
  { source: join(resources, "magnitude-install-update"), path: LINUX_UPDATE_INSTALLER_PATH.slice(1), mode: 0o755 },
  { source: join(resources, "magnitude.service"), path: "usr/lib/systemd/system/magnitude.service", mode: 0o644 },
  { source: join(resources, "sudoers"), path: "etc/sudoers.d/magnitude", mode: 0o440 },
] as const

/** Sudo ignores a rule that is writable or readable beyond root, so its mode is part of the contract. */
export const linuxSudoersPath = "etc/sudoers.d/magnitude"
