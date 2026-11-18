import { dirname, join, resolve } from "node:path"
import { fileURLToPath } from "node:url"
import { LINUX_UPDATE_INSTALLER_PATH } from "../../src/executables"

const resources = resolve(dirname(fileURLToPath(import.meta.url)), "../../resources/linux/system")

/**
 * System integration shipped by every Linux desktop package: the Polkit action lets the person at the
 * desktop run the update installer wrapper, and only it, without a password. No package ships a sudoers rule.
 */
export const linuxSystemFiles = [
  { source: join(resources, "dev.magnitude.update.policy"), path: "usr/share/polkit-1/actions/dev.magnitude.update.policy", mode: 0o644 },
  { source: join(resources, "magnitude-install-update"), path: LINUX_UPDATE_INSTALLER_PATH.slice(1), mode: 0o755 },
] as const

/** Each password-free Polkit action and the one wrapper it may name. */
export const linuxPolkitActions = [
  { policy: join(resources, "dev.magnitude.update.policy"), wrapper: LINUX_UPDATE_INSTALLER_PATH },
] as const

/** Written by `magnitude server setup` before the CLI-only install; package removal deletes it, since the helper it names goes with the package. */
export const linuxSudoersPath = "/etc/sudoers.d/magnitude"
