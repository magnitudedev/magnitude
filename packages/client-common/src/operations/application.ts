import { Application as Rpcs, type ApplicationOwnerState } from "@magnitudedev/sdk";
import { Group } from "@magnitudedev/effect-query";
import { mutation, streamQuery } from "./bind";

/** The application that owns the service: its kind, supported requests, updates, and launch at login. */
const WatchApplicationOwner = streamQuery(
  Rpcs.watchApplicationOwner,
  (client) => client.application.watchApplicationOwner,
  { reduce: (_, state): ApplicationOwnerState => state },
);

// The owner reports every resulting change through the watch, so no mutation synchronizes a query.
const CheckApplicationUpdate = mutation(Rpcs.checkApplicationUpdate, (client) => client.application.checkApplicationUpdate);
const DownloadApplicationUpdate = mutation(Rpcs.downloadApplicationUpdate, (client) => client.application.downloadApplicationUpdate);
const DiscardApplicationUpdate = mutation(Rpcs.discardApplicationUpdate, (client) => client.application.discardApplicationUpdate);
const InstallApplicationUpdate = mutation(Rpcs.installApplicationUpdate, (client) => client.application.installApplicationUpdate);
const SetApplicationAutoDownload = mutation(Rpcs.setApplicationAutoDownload, (client) => client.application.setApplicationAutoDownload);
const SetLaunchAtLogin = mutation(Rpcs.setLaunchAtLogin, (client) => client.application.setLaunchAtLogin);
const RestartApplication = mutation(Rpcs.restartApplication, (client) => client.application.restartApplication);
const QuitApplication = mutation(Rpcs.quitApplication, (client) => client.application.quitApplication);

export const Application = Group.make({
  WatchApplicationOwner,
  CheckApplicationUpdate,
  DownloadApplicationUpdate,
  DiscardApplicationUpdate,
  InstallApplicationUpdate,
  SetApplicationAutoDownload,
  SetLaunchAtLogin,
  RestartApplication,
  QuitApplication,
});
