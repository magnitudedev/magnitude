import type { ConnectionState } from "@magnitudedev/sdk";
import { FSM } from "@magnitudedev/utils";
import { Schema } from "effect";
import {
  ServiceChecking,
  LifecycleModelSchema,
  reduceLifecycle,
} from "./lifecycle";

class Bootstrapping extends Schema.TaggedClass<Bootstrapping>()(
  "Bootstrapping",
  {
    lifecycle: LifecycleModelSchema,
  }
) {}
class Online extends Schema.TaggedClass<Online>()("Online", {
  occurrence: Schema.Number.pipe(Schema.int(), Schema.nonNegative()),
}) {}
class Recovering extends Schema.TaggedClass<Recovering>()("Recovering", {
  occurrence: Schema.Number.pipe(Schema.int(), Schema.positive()),
  lifecycle: LifecycleModelSchema,
}) {}
const states = { Bootstrapping, Online, Recovering };
export const PresentationModelSchema = Schema.Union(...Object.values(states));
export type PresentationModel = typeof PresentationModelSchema.Type;
const PresentationFsm = FSM.defineFSM(states, {
  Bootstrapping: ["Bootstrapping", "Online", "Recovering"],
  Online: ["Recovering"],
  Recovering: ["Recovering", "Online"],
} as const);

export const initialPresentation = new Bootstrapping({
  lifecycle: new ServiceChecking({}),
});

/** Records only presentation history: latched bootstrap and numbered recovery notices. */
export const reducePresentation = (
  current: PresentationModel,
  state: ConnectionState,
  now: number
): PresentationModel => {
  if (state._tag === "Ready") {
    return current._tag === "Online"
      ? current
      : PresentationFsm.transition(current, "Online", {
          occurrence: current._tag === "Recovering" ? current.occurrence : 0,
        });
  }
  if (
    state._tag === "Connecting" &&
    state.reason === "recovery" &&
    current._tag !== "Recovering"
  ) {
    return PresentationFsm.transition(current, "Recovering", {
      occurrence: current._tag === "Online" ? current.occurrence + 1 : 1,
      lifecycle: reduceLifecycle(new ServiceChecking({}), state, now),
    });
  }
  switch (current._tag) {
    case "Online":
      return current;
    case "Bootstrapping":
      return PresentationFsm.transition(current, "Bootstrapping", {
        lifecycle: reduceLifecycle(current.lifecycle, state, now),
      });
    case "Recovering":
      return PresentationFsm.transition(current, "Recovering", {
        occurrence: current.occurrence,
        lifecycle: reduceLifecycle(current.lifecycle, state, now),
      });
  }
};
