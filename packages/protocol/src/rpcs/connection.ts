import { Rpc } from "@effect/rpc";
import { Schema } from "effect";

export const Health = Rpc.make("Health", {
  payload: Schema.Struct({}),
  success: Schema.Struct({
    service: Schema.Literal("magnitude-acn"),
    version: Schema.String,
    id: Schema.String,
    pid: Schema.Number,
  }),
});
