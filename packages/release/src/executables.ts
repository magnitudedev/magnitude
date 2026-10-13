/** Canonical executable names for the ACN and ICN services. */
export const ACN_EXECUTABLE_NAME = "magnitude-service"
export const ICN_EXECUTABLE_NAME = "magnitude-inference"
export const LINUX_DESKTOP_PACKAGE_NAME = "magnitude-desktop"
export const LINUX_DESKTOP_EXECUTABLE_PATH = `/usr/bin/${LINUX_DESKTOP_PACKAGE_NAME}`
/**
 * The only programs Polkit admits without a password for the person at the machine: each runs one
 * entry of the installed CLI, the update installer or server removal, and nothing else.
 */
export const LINUX_UPDATE_INSTALLER_PATH = `/usr/lib/${LINUX_DESKTOP_PACKAGE_NAME}/resources/magnitude-install-update`
export const LINUX_SERVER_REMOVER_PATH = `/usr/lib/${LINUX_DESKTOP_PACKAGE_NAME}/resources/magnitude-server-remove`
