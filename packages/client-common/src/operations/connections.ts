import { Connections as Rpcs, type HarnessConnectionsSnapshot } from "@magnitudedev/sdk";
import { Group } from "@magnitudedev/effect-query";
import { mutation, streamQuery } from "./bind";

/** The latest inspection of every supported harness on the machine running Magnitude. */
const WatchHarnessConnections = streamQuery(
  Rpcs.watchHarnessConnections,
  (client) => client.connections.watchHarnessConnections,
  { reduce: (_, snapshot): HarnessConnectionsSnapshot => snapshot },
);

// The watch reinspects after every change, so mutations need no query synchronization.
const ConnectHarness = mutation(Rpcs.connectHarness, (client) => client.connections.connectHarness);
const SyncHarnessConnections = mutation(Rpcs.syncHarnessConnections, (client) => client.connections.syncHarnessConnections);
const DisconnectHarness = mutation(Rpcs.disconnectHarness, (client) => client.connections.disconnectHarness);

export const Connections = Group.make({
  WatchHarnessConnections,
  ConnectHarness,
  SyncHarnessConnections,
  DisconnectHarness,
});
