import { Display as Rpcs } from "@magnitudedev/sdk";
import { Group } from "@magnitudedev/effect-query";
import { subscription } from "./bind";

const StreamDisplayView = subscription(
  Rpcs.streamDisplayView,
  (client) => client.display.streamDisplayView,
  {
    // A superseded shape's stream closes shortly after its last observer leaves.
    gcTime: "5 seconds",
  }
);

export const Display = Group.make({ StreamDisplayView });
