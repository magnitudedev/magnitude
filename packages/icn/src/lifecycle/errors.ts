import type { PlatformError } from "@effect/platform/Error";
import type { IcnStartupRecord } from "@magnitudedev/icn-protocol";
import type { GeneratedClientError } from "@magnitudedev/openapi-effect/client-runtime";
import { Data, Duration, type ParseResult } from "effect";
import type { ReleaseIcnInstallationError } from "./release-installation.js";

export class IcnBinaryNotFound extends Data.TaggedError("IcnBinaryNotFound")<{
  readonly path: string;
}> {
  override get message(): string {
    return `The inference server binary was not found at ${this.path}`;
  }
}

export class IcnBinaryNotExecutable extends Data.TaggedError(
  "IcnBinaryNotExecutable",
)<{ readonly path: string }> {
  override get message(): string {
    return `The inference server binary is not executable: ${this.path}`;
  }
}

export class IcnIdentityProbeTimedOut extends Data.TaggedError(
  "IcnIdentityProbeTimedOut",
)<{ readonly path: string; readonly timeout: Duration.Duration }> {
  override get message(): string {
    return `The inference server identity probe timed out after ${Duration.format(this.timeout)}`;
  }
}

export class IcnApiIncompatible extends Data.TaggedError("IcnApiIncompatible")<{
  readonly path: string;
  readonly expected: number;
  readonly actual: number;
}> {
  override get message(): string {
    return `Inference server API ${this.actual} is incompatible with ${this.expected}`;
  }
}

export class IcnNativeBuildIncompatible extends Data.TaggedError(
  "IcnNativeBuildIncompatible",
)<{ readonly path: string; readonly expected: string; readonly actual: string }> {
  override get message(): string {
    return `Inference server native build ${this.actual} does not match ${this.expected}`;
  }
}

export class IcnTargetIncompatible extends Data.TaggedError(
  "IcnTargetIncompatible",
)<{ readonly path: string; readonly expected: string; readonly actual: string }> {
  override get message(): string {
    return `Inference server target ${this.actual} does not match ${this.expected}`;
  }
}

export class IcnCapabilityMissing extends Data.TaggedError("IcnCapabilityMissing")<{
  readonly path: string;
  readonly capability: string;
}> {
  override get message(): string {
    return `The inference server binary does not provide required capability ${this.capability}`;
  }
}

export class IcnShutdownTimedOut extends Data.TaggedError("IcnShutdownTimedOut")<{
  readonly pid: number;
  readonly timeout: Duration.Duration;
}> {
  override get message(): string {
    return `Inference server process ${this.pid} did not exit within ${Duration.format(this.timeout)} after SIGKILL`;
  }
}

export class IcnExitedBeforeReady extends Data.TaggedError("IcnExitedBeforeReady")<{
  readonly pid: number;
  readonly code: number;
  readonly output: string;
}> {
  override get message(): string {
    return `Inference server process ${this.pid} exited with code ${this.code} before readiness`;
  }
}

export class IcnStartupRecordTimedOut extends Data.TaggedError(
  "IcnStartupRecordTimedOut",
)<{ readonly pid: number; readonly timeout: Duration.Duration; readonly output: string }> {
  override get message(): string {
    return `Inference server process ${this.pid} did not emit a startup record within ${Duration.format(this.timeout)}`;
  }
}

export class IcnStartupIdentityMismatch extends Data.TaggedError(
  "IcnStartupIdentityMismatch",
)<{
  readonly pid: number;
  readonly expectedInstanceId: string;
  readonly expectedApiVersion: number;
  readonly expectedNativeBuild: string;
  readonly actual: IcnStartupRecord;
  readonly output: string;
}> {
  override get message(): string {
    return `Inference server process ${this.pid} emitted startup identity for a different process or binary`;
  }
}

export class IcnStartupOriginInvalid extends Data.TaggedError(
  "IcnStartupOriginInvalid",
)<{ readonly origin: string; readonly output: string }> {
  override get message(): string {
    return `Inference server emitted an invalid startup origin: ${this.origin}`;
  }
}

export class IcnStartupOriginNotLoopback extends Data.TaggedError(
  "IcnStartupOriginNotLoopback",
)<{ readonly origin: string; readonly output: string }> {
  override get message(): string {
    return `Inference server bound a non-loopback startup origin: ${this.origin}`;
  }
}

export class IcnHealthIdentityMismatch extends Data.TaggedError(
  "IcnHealthIdentityMismatch",
)<{
  readonly expectedInstanceId: string;
  readonly expectedApiVersion: number;
  readonly expectedNativeBuild: string;
  readonly actualReady: boolean;
  readonly actualInstanceId: string;
  readonly actualApiVersion: number;
  readonly actualNativeBuild: string;
  readonly output: string;
}> {
  override get message(): string {
    return "Inference server health identity does not match its startup identity";
  }
}

export class IcnReadinessTimedOut extends Data.TaggedError("IcnReadinessTimedOut")<{
  readonly pid: number;
  readonly timeout: Duration.Duration;
  readonly output: string;
}> {
  override get message(): string {
    return `Inference server process ${this.pid} did not become ready within ${Duration.format(this.timeout)}`;
  }
}

export class IcnReadinessCommitRejected extends Data.TaggedError(
  "IcnReadinessCommitRejected",
)<{ readonly output: string }> {
  override get message(): string {
    return "Inference server stopped while readiness was being committed";
  }
}

export class IcnUnexpectedExit extends Data.TaggedError("IcnUnexpectedExit")<{
  readonly pid: number;
  readonly code: number;
  readonly output: string;
}> {
  override get message(): string {
    return `Inference server process ${this.pid} exited unexpectedly with code ${this.code}`;
  }
}

export type IcnBinaryResolutionError =
  | ReleaseIcnInstallationError
  | PlatformError
  | ParseResult.ParseError
  | IcnBinaryNotFound
  | IcnBinaryNotExecutable
  | IcnIdentityProbeTimedOut
  | IcnApiIncompatible
  | IcnNativeBuildIncompatible
  | IcnTargetIncompatible
  | IcnCapabilityMissing;

export type IcnLifecycleError =
  | IcnBinaryResolutionError
  | GeneratedClientError<never>
  | IcnShutdownTimedOut
  | IcnExitedBeforeReady
  | IcnStartupRecordTimedOut
  | IcnStartupIdentityMismatch
  | IcnStartupOriginInvalid
  | IcnStartupOriginNotLoopback
  | IcnHealthIdentityMismatch
  | IcnReadinessTimedOut
  | IcnReadinessCommitRejected
  | IcnUnexpectedExit;
