import { FileSystem } from "@effect/platform"
import { Array, Effect, Option, Schema } from "effect"
import { dirname, join } from "node:path"

/** Shell profiles that could not be changed; every other profile was. */
export class CliPathFailed extends Schema.TaggedError<CliPathFailed>()("CliPathFailed", {
  message: Schema.String,
  paths: Schema.NonEmptyArray(Schema.String),
}) {}

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
    // Each profile is independent, so an unwritable or edited one leaves the others correct.
    const outcomes = yield* Effect.forEach(unique, ({ path, block }) => Effect.gen(function* () {
      const current = yield* read(path)
      // Edited or incomplete blocks belong to the user; never guess what to delete.
      if (current.includes(begin) && !current.includes(block)) return Option.some(`${path}: its Magnitude PATH entry was edited, so it was left unchanged`)
      const next = remove ? current.replace(block, "") : current.includes(block) ? current : current + block
      if (next !== current) {
        yield* fs.makeDirectory(dirname(path), { recursive: true })
        yield* fs.writeFileString(path, next)
      }
      return Option.none<string>()
    }).pipe(
      Effect.catchAll(error => Effect.succeed(Option.some(error._tag === "SystemError" && error.reason === "PermissionDenied"
        ? `${path} isn't writable` : `${path}: ${error.message}`))),
      Effect.map(reason => Option.map(reason, reason => ({ path, reason }))),
    ))
    const failed = Array.getSomes(outcomes)
    if (!Array.isNonEmptyArray(failed)) return
    return yield* new CliPathFailed({
      message: `Could not ${remove ? "remove Magnitude's PATH entry from" : "add Magnitude's PATH entry to"} ${failed.length === 1 ? "a shell profile" : "some shell profiles"}. ${failed.map(failure => failure.reason).join(". ")}.`,
      paths: Array.map(failed, failure => failure.path),
    })
  })
  return { install: change(false), remove: change(true) }
})
