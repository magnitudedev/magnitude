import { Connection as Rpcs } from "@magnitudedev/sdk";
import { Group } from "@magnitudedev/effect-query";
import { query } from "./bind";

const Health = query(Rpcs.health, (client) => client.connection.health);

export const Connection = Group.make({ Health });
