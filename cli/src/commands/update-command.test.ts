import { Effect } from "effect"
import { expect, it } from "vitest"
import { ApplicationUpdateState, type ApplicationUpdateAction } from "@magnitudedev/sdk/desktop-host"
import { performUpdate, renderApplicationUpdate } from "./update-runtime"

const state = (check: ApplicationUpdateState["check"], transfer: ApplicationUpdateState["transfer"] = { _tag: "Idle" }) =>
  ApplicationUpdateState.make({ transfer, check, preference: { _tag: "Known", autoDownload: true } })

const owner = (states: Partial<Record<ApplicationUpdateAction, ApplicationUpdateState>>) => {
  const calls: ApplicationUpdateAction[] = []
  const update = (action: ApplicationUpdateAction) => Effect.sync(() => {
    calls.push(action)
    return { owner: "Headless" as const, state: states[action] ?? states.status! }
  })
  return { calls, update }
}

it("fails the command when the requested update check fails", async () => {
  const { update } = owner({ check: state({ _tag: "Failed", message: "Could not check for application updates." }) })
  const error = await Effect.runPromise(performUpdate("check", update).pipe(Effect.flip))
  expect(error.message).toBe("Could not check for application updates.")
})

it("reports a successful update check", async () => {
  const { update } = owner({ check: state({ _tag: "Succeeded", at: 1 }) })
  const result = await Effect.runPromise(performUpdate("check", update))
  expect(renderApplicationUpdate(result.state)).toBe("Magnitude is up to date.\n")
})

it("says Magnitude is up to date when install has nothing downloaded, without asking the owner to install", async () => {
  const { calls, update } = owner({ status: state({ _tag: "Succeeded", at: 1 }) })
  const result = await Effect.runPromise(performUpdate("install", update))
  expect(calls).toEqual(["status"])
  expect(result.installed).toBe(false)
  expect(renderApplicationUpdate(result.state, "Headless", true)).toBe("Magnitude is up to date.\n")
})

it("asks the owner to install a downloaded update", async () => {
  const { calls, update } = owner({ status: state({ _tag: "Succeeded", at: 1 }, { _tag: "Ready", version: "2.0.0" }) })
  const result = await Effect.runPromise(performUpdate("install", update))
  expect(calls).toEqual(["status", "install"])
  expect(result.installed).toBe(true)
})
