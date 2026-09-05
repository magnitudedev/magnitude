import { Changes as Rpcs } from "@magnitudedev/sdk";
import { Group } from "@magnitudedev/effect-query";
import { subscription } from "./bind";

const StreamChanges = subscription(
  Rpcs.streamChanges,
  (client) => client.changes.streamChanges
);

export const Changes = Group.make({ StreamChanges });
