import { createId } from "@magnitudedev/generate-id";
import {
  AcnInstanceIdSchema,
  AcnIdentitySchema,
  type AcnHealthResponse,
  type AcnHealthState,
} from "@magnitudedev/acn-protocol";

/** Stable for the lifetime of this ACN process and unique across candidates. */
export const ACN_INSTANCE_ID = AcnInstanceIdSchema.make(createId());

export const makeHealthResponse = (
  version: string,
  state: AcnHealthState,
  id: string = ACN_INSTANCE_ID,
  pid: number = process.pid
): AcnHealthResponse => ({
  service: "magnitude-acn",
  version: AcnIdentitySchema.make(version),
  id: AcnInstanceIdSchema.make(id),
  pid,
  state,
});
