import { Rpc } from "@effect/rpc";
import { Schema } from "effect";
import { AcnHealthResponseSchema } from "../schemas/acn-health";

export const Health = Rpc.make("Health", {
  payload: Schema.Struct({}),
  success: AcnHealthResponseSchema,
});
