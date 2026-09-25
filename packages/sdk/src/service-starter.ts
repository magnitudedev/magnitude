import { Context, type Stream } from "effect";
import type { ServiceStartProgress } from "@magnitudedev/acn-protocol";
import type { ServiceStartError } from "./connection-errors";

export interface MagnitudeServiceStarter {
  readonly start: Stream.Stream<ServiceStartProgress, ServiceStartError>;
}
export const MagnitudeServiceStarter = Context.GenericTag<MagnitudeServiceStarter>(
  "@magnitudedev/sdk/MagnitudeServiceStarter"
);
