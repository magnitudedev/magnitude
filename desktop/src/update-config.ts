import { decodeUpdateConfiguration } from "@magnitudedev/release/hosted-update"

declare const __MAGNITUDE_UPDATE_CONFIGURATION__: unknown
declare const __MAGNITUDE_UPDATE_ACCEPTANCE__: boolean
export const isUpdateAcceptanceBuild = __MAGNITUDE_UPDATE_ACCEPTANCE__

/** Acceptance trust is an explicit build input, never a runtime environment override. */
export const readUpdateConfiguration = decodeUpdateConfiguration(__MAGNITUDE_UPDATE_CONFIGURATION__, isUpdateAcceptanceBuild)
