import { Effect, Option, Schema } from "effect"
import { AppearancePreference } from "@magnitudedev/sdk/desktop-host"
import { ApplicationHostFailed, type ApplicationHost } from "@magnitudedev/client-common"
import { renderApplication } from "../run"

const APPEARANCE_KEY = "magnitude:appearance"

/** A browser keeps only the viewer's own appearance; everything about the server comes from ACN. */
const host: ApplicationHost = {
  window: Option.none(),
  appearance: {
    read: Effect.try({ try: () => window.localStorage.getItem(APPEARANCE_KEY), catch: () => new ApplicationHostFailed({ message: "The saved appearance could not be read." }) }).pipe(
      Effect.map(value => Schema.is(AppearancePreference)(value) ? value : "system" as const),
    ),
    save: preference => Effect.try({ try: () => window.localStorage.setItem(APPEARANCE_KEY, preference), catch: () => new ApplicationHostFailed({ message: "The appearance could not be saved in this browser." }) }),
  },
  shell: Option.none(),
}

renderApplication({ host, origin: Effect.succeed(window.location.origin), navigation: "location" })
