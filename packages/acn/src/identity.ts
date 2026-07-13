import { createId } from "@magnitudedev/generate-id";

/** Stable for the lifetime of this ACN process and unique across candidates. */
export const ACN_OWNER_ID = createId();

export interface HealthResponse {
  readonly service: "magnitude-acn";
  readonly version: string;
  readonly id: string;
  readonly pid: number;
}

export const makeHealthResponse = (
  version: string,
  id: string = ACN_OWNER_ID,
  pid: number = process.pid
): HealthResponse => ({
  service: "magnitude-acn",
  version,
  id,
  pid,
});
