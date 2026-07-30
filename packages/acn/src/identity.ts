import { createId } from "@magnitudedev/generate-id";
import {
  AcnOwnerIdSchema,
  type AcnHealthResponse,
  type AcnHealthState,
} from "@magnitudedev/acn-protocol";

/** Stable for the lifetime of this ACN process and unique across candidates. */
export const ACN_OWNER_ID = AcnOwnerIdSchema.make(createId());

export const makeHealthResponse = (
  version: string,
  state: AcnHealthState,
  id: string = ACN_OWNER_ID,
  pid: number = process.pid
): AcnHealthResponse => ({
  service: "magnitude-acn",
  version,
  id: AcnOwnerIdSchema.make(id),
  pid,
  state,
});
