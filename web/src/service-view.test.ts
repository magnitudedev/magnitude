import { expect, it } from "vitest"
import type { ConnectionState } from "@magnitudedev/sdk"
import { browserServiceView } from "./service-view"

const connecting = (reason: "initial" | "recovery" | "retry") => ({ _tag: "Connecting", reason }) as unknown as ConnectionState

it("shows a dropped connection as reconnecting rather than starting", () => {
  expect(browserServiceView(connecting("initial"))._tag).toBe("Starting")
  expect(browserServiceView(connecting("recovery"))._tag).toBe("Reconnecting")
  expect(browserServiceView(connecting("retry"))._tag).toBe("Reconnecting")
  expect(browserServiceView({ _tag: "Idle" } as unknown as ConnectionState)._tag).toBe("Starting")
})
