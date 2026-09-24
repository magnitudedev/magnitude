import { decodeApplicationUpdateConfiguration } from "@magnitudedev/daemon-management/application-update"

declare const __MAGNITUDE_UPDATE_CONFIGURATION__: unknown
declare const __MAGNITUDE_UPDATE_ACCEPTANCE__: boolean
export const isUpdateAcceptanceBuild = __MAGNITUDE_UPDATE_ACCEPTANCE__

/** Acceptance trust is an explicit build input, never a runtime environment override. */
export const readUpdateConfiguration = decodeApplicationUpdateConfiguration(__MAGNITUDE_UPDATE_CONFIGURATION__)
