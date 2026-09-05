import { type MagnitudeServiceStarter, ServiceStartFailed } from "@magnitudedev/sdk"
import { Option, Stream } from "effect"
import type { AcnInstanceManager } from "./acn-jit/acn-instance-manager"
import { formatAcnEnsuranceError } from "./acn-jit/format-error"
import { DAEMON_TARGET } from "./version"

/** The host owns daemon selection; the SDK sees only startup progress and failure. */
export const makeServiceStarter = (manager: AcnInstanceManager): MagnitudeServiceStarter => ({
  start: manager.ensure({ target: DAEMON_TARGET }).pipe(
    Stream.filterMap(event => event._tag === "Observation" ? Option.some(event.observation) : Option.none()),
    Stream.mapError(error => new ServiceStartFailed({ message: formatAcnEnsuranceError(error) })),
  ),
})
