/**
 * Whether a `launchctl print` against `gui/<uid>` failed only because the user has no GUI
 * domain, as on a Mac reached over SSH with nobody logged in at the console. launchd reports
 * this as 112 before any login since boot and as 125 once the domain exists without a session.
 */
export const launchdGuiDomainIsAbsent = (result: { readonly code: number; readonly stderr: string }, uid: number) => {
  const stderr = result.stderr.trim()
  if (result.code === 112) return stderr === `Bad request.\nCould not find domain for user gui: ${uid}`
  if (result.code === 125) return stderr === "Could not print domain: 125: Domain does not support specified action"
  return false
}
