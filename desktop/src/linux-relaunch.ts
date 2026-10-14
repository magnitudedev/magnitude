import { spawn } from "node:child_process"

/**
 * Electron's relauncher starts the new application with PR_SET_NO_NEW_PRIVS, which stops the setuid
 * `chrome-sandbox` from gaining root. Where unprivileged user namespaces are also restricted, as on
 * Ubuntu 24.04, Chromium then has no usable sandbox and aborts at startup. The packaged Linux app
 * instead hands its relaunch to a detached shell that waits for this process to exit and then starts
 * the installed launcher, exactly as a menu launch does.
 */
export const linuxRelaunchCommand = (pid: number, launcher: string, args: readonly string[]) => ({
  command: "/bin/sh",
  args: ["-c", 'while kill -0 "$0" 2>/dev/null; do sleep 0.2; done; exec "$@"', String(pid), launcher, ...args],
})

export const relaunchLinuxPackage = (launcher: string, args: readonly string[]) => {
  const { command, args: commandArgs } = linuxRelaunchCommand(process.pid, launcher, args)
  spawn(command, commandArgs, { detached: true, stdio: "ignore" }).unref()
}
