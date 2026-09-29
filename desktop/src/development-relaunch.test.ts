import { Option } from "effect"
import { expect, it } from "vitest"
import { developmentRelaunchExitCode, developmentRelaunchFromExitCode, rendererServedByDevServer } from "./development-relaunch"

it("round-trips window visibility through the exit code and rejects ordinary exits", () => {
  for (const showWindow of [true, false]) {
    expect(developmentRelaunchFromExitCode(developmentRelaunchExitCode({ showWindow }))).toEqual(Option.some({ showWindow }))
  }
  expect(developmentRelaunchFromExitCode(0)).toEqual(Option.none())
  expect(developmentRelaunchFromExitCode(1)).toEqual(Option.none())
  expect(developmentRelaunchFromExitCode(null)).toEqual(Option.none())
})

it("detects the renderer dev server only from its URL variable", () => {
  expect(rendererServedByDevServer({ ELECTRON_RENDERER_URL: "http://localhost:5173" })).toBe(true)
  expect(rendererServedByDevServer({ NODE_ENV: "development" })).toBe(false)
})
