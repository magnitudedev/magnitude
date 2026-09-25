import { FileSystem } from "@effect/platform"
import { Effect } from "effect"
import { dirname, join } from "node:path"
import { CliLinkFailed } from "./mac-cli-link"

const begin = "# >>> Magnitude CLI PATH >>>"
const end = "# <<< Magnitude CLI PATH <<<"
export const MAC_CLI_DIRECTORY = ".magnitude/bin"
const posixBlock = `\n${begin}
case "$PATH" in
  "$HOME/${MAC_CLI_DIRECTORY}"|"$HOME/${MAC_CLI_DIRECTORY}":*) ;;
  *) export PATH="$HOME/${MAC_CLI_DIRECTORY}:$PATH" ;;
esac
${end}\n`
const fishBlock = `\n${begin}
fish_add_path --path --move --prepend "$HOME/${MAC_CLI_DIRECTORY}"
${end}\n`

export const makeMacCliPath = (home: string, environment: Readonly<Record<string, string>>) => Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const read = (path: string) => fs.readFileString(path).pipe(
    Effect.catchAll(error => error._tag === "SystemError" && error.reason === "NotFound"
      ? Effect.succeed("") : Effect.fail(error)),
  )
  const zsh = environment.ZDOTDIR || home
  const files = [join(zsh, ".zprofile"), join(zsh, ".zshrc")]
  // Do not create a higher-priority login profile that hides existing user configuration.
  let bashProfile = join(home, ".bash_profile")
  for (const name of [".bash_profile", ".bash_login", ".profile"]) {
    const path = join(home, name)
    if (yield* fs.exists(path)) { bashProfile = path; break }
  }
  files.push(bashProfile, join(home, ".bashrc"))
  const profiles = files.map(path => ({ path, block: posixBlock }))
  const fish = join(environment.XDG_CONFIG_HOME || join(home, ".config"), "fish")
  if (environment.SHELL?.endsWith("/fish") || (yield* fs.exists(fish))) {
    profiles.push({ path: join(fish, "config.fish"), block: fishBlock })
  }
  const change = (remove: boolean) => Effect.gen(function* () {
    const selected = remove
      ? [...profiles, ...[".bash_profile", ".bash_login", ".profile"].map(name => ({ path: join(home, name), block: posixBlock }))]
      : profiles
    const unique = [...new Map(selected.map(profile => [profile.path, profile])).values()]
    const edits = yield* Effect.forEach(unique, ({ path, block }) => Effect.gen(function* () {
      const current = yield* read(path)
      // Edited or incomplete blocks belong to the user; never guess what to delete.
      if (current.includes(begin) && !current.includes(block)) {
        return yield* new CliLinkFailed({ message: `Magnitude's PATH entry in ${path} was edited; it was left unchanged.` })
      }
      const next = remove ? current.replace(block, "") : current.includes(block) ? current : current + block
      return { path, current, next }
    }))
    // Validate every managed block before changing any profile.
    yield* Effect.forEach(edits, ({ path, current, next }) => Effect.gen(function* () {
      if (next === current) return
      yield* fs.makeDirectory(dirname(path), { recursive: true })
      yield* fs.writeFileString(path, next)
    }), { discard: true })
  }).pipe(Effect.mapError(error => error._tag === "CliLinkFailed" ? error
    : new CliLinkFailed({ message: `Could not update shell PATH: ${error.message}` })))
  return { install: change(false), remove: change(true) }
})
