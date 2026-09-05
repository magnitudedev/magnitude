import { Agent as Rpcs } from "@magnitudedev/sdk";
import { Group } from "@magnitudedev/effect-query";
import { turnAdmissionScope } from "./configuration";
import { mutation } from "./bind";

const SendMessage = mutation(
  Rpcs.sendMessage,
  (client) => client.agent.sendMessage,
  { scope: () => turnAdmissionScope }
);

const StartGoal = mutation(Rpcs.startGoal, (client) => client.agent.startGoal, {
  scope: () => turnAdmissionScope,
});

const Interrupt = mutation(Rpcs.interrupt, (client) => client.agent.interrupt);

export const Agent = Group.make({ SendMessage, StartGoal, Interrupt });
