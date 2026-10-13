/**
 * The only commands the CLI runs as root under pkexec: the update installer, which Polkit admits
 * through its wrapper, and server removal, which the desktop runs after administrator
 * authentication. Every other command is refused there, whatever a Polkit policy allowed.
 */
export const PKEXEC_COMMANDS: ReadonlySet<string> = new Set(["_install-application-update", "_server-remove"])

export const admitsPkexecCommand = (argv: readonly string[], uid: number | undefined, environment: Readonly<Record<string, string | undefined>>) =>
  uid !== 0 || environment.PKEXEC_UID === undefined || PKEXEC_COMMANDS.has(argv[0] ?? "")
