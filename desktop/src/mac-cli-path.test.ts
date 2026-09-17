import { FileSystem } from "@effect/platform"
import { NodeContext } from "@effect/platform-node"
import { Effect } from "effect"
import { execFileSync } from "node:child_process"
import { join } from "node:path"
import { describe, expect, it } from "vitest"
import { makeMacCliLink } from "./mac-cli-link"
import { makeMacCliPath } from "./mac-cli-path"

const fixture = (test: (home: string, fs: FileSystem.FileSystem) => Effect.Effect<void, unknown, FileSystem.FileSystem>) =>
  Effect.runPromise(Effect.scoped(Effect.gen(function* () {
    const fs = yield* FileSystem.FileSystem
    const home = yield* fs.makeTempDirectoryScoped({ prefix: "magnitude fresh home ' " })
    yield* test(home, fs)
  })).pipe(Effect.provide(NodeContext.layer)))

describe("user-owned Mac command PATH", () => {
  it("preserves login profiles, is idempotent, and removes only its own text", () => fixture((home, fs) => Effect.gen(function* () {
    const profile = join(home, ".profile")
    const original = 'export MY_EXISTING_SETTING=yes'
    yield* fs.writeFileString(profile, original)
    const path = yield* makeMacCliPath(home, {})
    yield* path.install
    const installed = yield* fs.readFileString(profile)
    yield* path.install
    expect(yield* fs.readFileString(profile)).toBe(installed)
    expect(yield* fs.exists(join(home, ".bash_profile"))).toBe(false)
    yield* fs.writeFileString(profile, installed + '# added later\n')
    yield* path.remove
    expect(yield* fs.readFileString(profile)).toBe(original + '# added later\n')
    yield* path.remove
    expect(yield* fs.readFileString(profile)).toBe(original + '# added later\n')
  })))

  it("preserves an edited registration instead of deleting user content", () => fixture((home, fs) => Effect.gen(function* () {
    const path = yield* makeMacCliPath(home, {})
    yield* path.install
    const file = join(home, ".zshrc")
    const edited = (yield* fs.readFileString(file)).replace('export PATH=', 'export CUSTOM_PATH=')
    yield* fs.writeFileString(file, edited)
    expect((yield* Effect.either(path.remove))._tag).toBe("Left")
    expect(yield* fs.readFileString(file)).toBe(edited)
  })))

  it("removes its original login entry after a higher-priority profile is added", () => fixture((home, fs) => Effect.gen(function* () {
    const original = "# original profile\n"
    yield* fs.writeFileString(join(home, ".profile"), original)
    yield* (yield* makeMacCliPath(home, {})).install
    yield* fs.writeFileString(join(home, ".bash_profile"), "# added by user\n")
    yield* (yield* makeMacCliPath(home, {})).remove
    expect(yield* fs.readFileString(join(home, ".profile"))).toBe(original)
    expect(yield* fs.readFileString(join(home, ".bash_profile"))).toBe("# added by user\n")
  })))

  it("respects custom zsh and fish configuration directories", () => fixture((home, fs) => Effect.gen(function* () {
    const zdot = join(home, "zsh")
    const config = join(home, "config")
    const path = yield* makeMacCliPath(home, { ZDOTDIR: zdot, XDG_CONFIG_HOME: config, SHELL: "/opt/homebrew/bin/fish" })
    yield* path.install
    expect(yield* fs.exists(join(zdot, ".zshrc"))).toBe(true)
    expect(yield* fs.exists(join(home, ".zshrc"))).toBe(false)
    expect(yield* fs.readFileString(join(config, "fish/config.fish"))).toContain('fish_add_path --path --move --prepend')
  })))

  it.skipIf(process.platform !== "darwin")("real fresh login and interactive shells prefer the bundled CLI", () => fixture((home, fs) => Effect.gen(function* () {
    const bin = join(home, ".magnitude/bin")
    const target = join(home, "Magnitude.app/Contents/Resources/magnitude")
    yield* fs.makeDirectory(join(target, ".."), { recursive: true })
    yield* fs.writeFileString(target, '#!/bin/sh\nprintf "bundled-cli\\n"\n')
    yield* fs.chmod(target, 0o755)
    const link = yield* makeMacCliLink({ link: join(bin, "magnitude"), target })
    const path = yield* makeMacCliPath(home, {})
    yield* link.install
    yield* path.install
    for (const shell of ["/bin/zsh", "/bin/bash"]) {
      for (const mode of ["-lc", "-ic"]) {
        const output = execFileSync(shell, [mode, 'command -v magnitude; magnitude'], {
          cwd: home, env: { HOME: home, PATH: "/usr/bin:/bin:/usr/sbin:/sbin", ZDOTDIR: home, TERM: "dumb" },
          encoding: "utf8", stdio: ["ignore", "pipe", "pipe"],
        })
        expect(output.trim().split("\n")).toEqual([join(bin, "magnitude"), "bundled-cli"])
      }
    }
    yield* link.remove
    yield* path.remove
    expect(yield* fs.exists(join(bin, "magnitude"))).toBe(false)
  })))
})
