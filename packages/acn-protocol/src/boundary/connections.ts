import { Rpc } from "@effect/rpc"
import { Schema } from "effect"
import { atMostOnce, replaySafe } from "../transport/recovery"
import {
  HarnessConnectOutcome,
  HarnessConnectRequest,
  HarnessConnectionFailed,
  HarnessConnectionsSnapshot,
  HarnessIdSchema,
  HarnessSetup,
  HarnessSetupRequest,
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

/** Renders the setup prompt and run command for a harness on the viewer's computer; changes nothing. */
const DescribeHarnessSetup = Rpc.make("DescribeHarnessSetup", {
  payload: HarnessSetupRequest,
  success: HarnessSetup,
  error: HarnessConnectionFailed,
}).pipe(replaySafe)

export const Connections = {
  watchHarnessConnections: WatchHarnessConnections,
  connectHarness: ConnectHarness,
  syncHarnessConnections: SyncHarnessConnections,
  disconnectHarness: DisconnectHarness,
  describeHarnessSetup: DescribeHarnessSetup,
}
