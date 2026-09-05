import { Shell as Rpcs } from "@magnitudedev/sdk";
import { Group } from "@magnitudedev/effect-query";
import { mutation } from "./bind";

const RunBash = mutation(Rpcs.runBash, (client) => client.shell.runBash);

export const Shell = Group.make({ RunBash });
