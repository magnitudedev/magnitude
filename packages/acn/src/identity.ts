import { createId } from "@magnitudedev/generate-id";
import {
  AcnInstanceIdSchema,
  AcnIdentitySchema,
  AcnRevisionSchema,
  type AcnHealthResponse,
  type AcnHealthState,
} from "@magnitudedev/acn-protocol";
import { ACN_REVISION } from "./version";

/** Stable for the lifetime of this ACN process and unique across candidates. */
export const ACN_INSTANCE_ID = AcnInstanceIdSchema.make(createId());

export const makeHealthResponse = (
  version: string,
  state: AcnHealthState,
  id: string = ACN_INSTANCE_ID,
  pid: number = process.pid,
  revision: number = ACN_REVISION,
): AcnHealthResponse => ({
  service: "magnitude-acn",
  version: AcnIdentitySchema.make(version),
  revision: AcnRevisionSchema.make(revision),
  id: AcnInstanceIdSchema.make(id),
  pid,
  state,
});
