import { spawn } from "node:child_process"
import { resolve } from "node:path"
import { fileURLToPath } from "node:url"
import { Effect, Option } from "effect"
import { developmentRelaunchFromExitCode, type DevelopmentRelaunch } from "../src/development-relaunch"

const root = fileURLToPath(new URL("../", import.meta.url))
const electronVite = resolve(root, "../node_modules/.bin/electron-vite")

const electronArguments = (relaunch: DevelopmentRelaunch): string => {
  const inherited: ReadonlyArray<string> = process.env.ELECTRON_CLI_ARGS ? JSON.parse(process.env.ELECTRON_CLI_ARGS) : []
  return JSON.stringify([...inherited.filter(argument => argument !== "--background"), ...(relaunch.showWindow ? [] : ["--background"])])
}

const runElectronVite = (relaunch: Option.Option<DevelopmentRelaunch>) => Effect.async<number>(resume => {
  const environment = Option.match(relaunch, {
    onNone: () => process.env,
    onSome: value => ({ ...process.env, ELECTRON_CLI_ARGS: electronArguments(value) }),
  })
  const child = spawn(electronVite, ["dev", ...process.argv.slice(2)], { cwd: root, stdio: "inherit", env: environment })
  child.once("error", error => { console.error(error.message); resume(Effect.succeed(1)) })
  child.once("exit", (code, signal) => resume(Effect.succeed(code ?? (signal ? 1 : 0))))
})

/** Restarts `electron-vite dev` when the app exits with a development relaunch code, so in-app Restart keeps a live renderer dev server. */
const supervise = Effect.gen(function* () {
  let relaunch = Option.none<DevelopmentRelaunch>()
  for (;;) {
    const code = yield* runElectronVite(relaunch)
    relaunch = developmentRelaunchFromExitCode(code)
    if (Option.isNone(relaunch)) return code
    console.log("[desktop] Restart requested; starting Electron with a fresh renderer dev server")
  }
})

process.exit(await Effect.runPromise(supervise))
