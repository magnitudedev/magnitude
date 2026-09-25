import { Schema } from "effect";

export class ServiceUnavailable extends Schema.TaggedError<ServiceUnavailable>()(
  "ServiceUnavailable",
  {
    origin: Schema.String,
    message: Schema.String,
  }
) {}
export class ServiceStartFailed extends Schema.TaggedError<ServiceStartFailed>()(
  "ServiceStartFailed",
  {
    message: Schema.String,
  }
) {}
export const ServiceStartErrorSchema = ServiceStartFailed;
export type ServiceStartError = typeof ServiceStartErrorSchema.Type;
export class InvalidServiceResponse extends Schema.TaggedError<InvalidServiceResponse>()(
  "InvalidServiceResponse",
  {
    origin: Schema.String,
    message: Schema.String,
  }
) {}
export class ProtocolMismatch extends Schema.TaggedError<ProtocolMismatch>()(
  "ProtocolMismatch",
  {
    expected: Schema.Number,
    actual: Schema.Number,
    daemonVersion: Schema.String,
  }
) {}
export class ConnectionClosed extends Schema.TaggedError<ConnectionClosed>()(
  "ConnectionClosed",
  {}
) {}
export const ConnectionErrorSchema = Schema.Union(
  ServiceUnavailable,
  ServiceStartErrorSchema,
  InvalidServiceResponse,
  ProtocolMismatch,
  ConnectionClosed
);
export type ConnectionError = typeof ConnectionErrorSchema.Type;

export const formatConnectionError = (error: ConnectionError): string => {
  switch (error._tag) {
    case "ProtocolMismatch":
      return `Magnitude RPC protocol mismatch: client requires ${error.expected}, service ${error.daemonVersion} provides ${error.actual}. Install matching Magnitude and plugin versions.`;
    case "ConnectionClosed":
      return "Magnitude client is closed";
    default:
      return error.message;
  }
};
