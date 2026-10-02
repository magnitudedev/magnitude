import { Rpc } from "@effect/rpc"
import { Schema } from "effect"
import { atMostOnce } from "../transport/recovery"
import {
  HarnessConnectOutcome,
  HarnessConnectRequest,
  HarnessConnectionFailed,
  HarnessConnectionsSnapshot,
  HarnessIdSchema,
} from "../schemas/harness-connections"

/** Inspects on subscribe, after any connect or disconnect, and every two seconds while observed. */
const WatchHarnessConnections = Rpc.make("WatchHarnessConnections", {
  payload: Schema.Struct({}),
  success: HarnessConnectionsSnapshot,
  stream: true,
})

const ConnectHarness = Rpc.make("ConnectHarness", {
  payload: HarnessConnectRequest,
  success: HarnessConnectOutcome,
  error: HarnessConnectionFailed,
}).pipe(atMostOnce)

const SyncHarnessConnections = Rpc.make("SyncHarnessConnections", {
  payload: Schema.Struct({ harness: Schema.optionalWith(HarnessIdSchema, { as: "Option", exact: true }) }),
  success: Schema.Struct({}),
  error: HarnessConnectionFailed,
}).pipe(atMostOnce)

const DisconnectHarness = Rpc.make("DisconnectHarness", {
  payload: Schema.Struct({ harness: HarnessIdSchema }),
  success: Schema.Struct({}),
  error: HarnessConnectionFailed,
}).pipe(atMostOnce)

export const Connections = {
  watchHarnessConnections: WatchHarnessConnections,
  connectHarness: ConnectHarness,
  syncHarnessConnections: SyncHarnessConnections,
  disconnectHarness: DisconnectHarness,
}
