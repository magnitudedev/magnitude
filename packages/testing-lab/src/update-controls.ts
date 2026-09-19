import { Effect, Schema } from "effect"
import type { Page } from "playwright"
import { desktopAutomation as automation } from "../../../desktop/src/automation"
import { AssertionFailure } from "./domain"

export const UpdateState = Schema.Literal("Idle", "Available", "Downloading", "Cancelling", "Staging", "Ready", "InstallationFailed", "Failed", "Unavailable", "Closed")
export const UpdateObservation = Schema.Struct({ state: UpdateState, version: Schema.optionalWith(Schema.String, { as: "Option", exact: true }) })
export interface DesktopUpdates {
  readonly action: (action: "check" | "download" | "restart" | "discard") => Effect.Effect<void, AssertionFailure>
  readonly automatic: (enabled: boolean) => Effect.Effect<void, AssertionFailure>
  readonly wait: (state: typeof UpdateState.Type) => Effect.Effect<typeof UpdateObservation.Type, AssertionFailure>
}
const failure = (message: string) => new AssertionFailure({ message })
const action = <A>(label: string, run: () => Promise<A>) => Effect.tryPromise({ try: run,
  catch: error => failure(`${label}: ${error instanceof Error ? error.message.slice(0, 1800) : "Playwright failed"}`) })

/** Only the visible Settings controls perform mutations; attributes expose rendered domain state. */
export const playwrightUpdates = (page: Page, openSettings: Effect.Effect<void, AssertionFailure>): DesktopUpdates => ({
  action: name => openSettings.pipe(Effect.zipRight(action(`Update ${name}`, () => page.getByTestId(automation.updateAction(name)).click()))),
  automatic: enabled => openSettings.pipe(Effect.zipRight(action("Set automatic update downloads", async () => {
    const control = page.getByTestId(automation.updateAutomatic)
    // Controlled inputs may restore their previous value until the owner's async acknowledgement.
    // Click once, then observe the rendered state; setChecked requires an immediate DOM toggle.
    if (await control.isChecked() !== enabled) await control.click()
    await control.and(page.locator(enabled ? ":checked" : ":not(:checked)")).waitFor()
  }))),
  wait: expected => openSettings.pipe(Effect.zipRight(action("Wait for application update state", async () => {
    await page.waitForFunction(({ id, expected }) => {
      const element = document.querySelector(`[data-testid="${id}"]`)
      const state = element?.getAttribute("data-update-state")
      return state === expected || state === "Failed" || state === "InstallationFailed" || state === "Unavailable" || element?.getAttribute("data-update-check") === "Failed"
    }, { id: automation.updates, expected }, { timeout: 600_000 })
    const region = page.getByTestId(automation.updates)
    const state = await region.getAttribute("data-update-state")
    if (state !== expected) throw new Error(`Expected ${expected}, observed ${state}: ${(await region.innerText()).slice(0, 1500)}`)
    const version = await region.getAttribute("data-update-version")
    return { state, ...(version === null ? {} : { version }) }
  })), Effect.flatMap(value => Schema.decodeUnknown(UpdateObservation)(value).pipe(Effect.mapError(() => failure("Rendered update state is invalid"))))),
})
