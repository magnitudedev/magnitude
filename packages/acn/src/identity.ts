import { createId } from "@magnitudedev/generate-id";
import {
  AcnInstanceIdSchema,
  AcnIdentitySchema,
  AcnRevisionSchema,
  type MagnitudeHealthResponse,
  MAGNITUDE_RPC_VERSION,
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
): MagnitudeHealthResponse => ({
  service: "magnitude-acn",
  version: AcnIdentitySchema.make(version),
  revision: AcnRevisionSchema.make(revision),
  rpcVersion: MAGNITUDE_RPC_VERSION,
  id: AcnInstanceIdSchema.make(id),
  pid,
  state,
});
