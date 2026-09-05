import { ProcessStartIdentitySchema } from "@magnitudedev/acn-protocol";
import { describe, expect, it } from "vitest";
import {
  AcnCandidateExitedAfterAdmission,
  AcnHealthRequestFailed,
  AcnHealthResponseInvalid,
  AcnHealthUnavailable,
} from "./errors";
import { formatAcnEnsuranceError } from "./format-error";

describe("daemon startup diagnostics", () => {
  it("formats candidate exits as diagnostics without duplicating the CLI heading", () => {
    expect(formatAcnEnsuranceError(new AcnCandidateExitedAfterAdmission({
      pid: 42,
      code: 1,
      stderr: "  local inference failed  ",
    }))).toBe("local inference failed");
    expect(formatAcnEnsuranceError(new AcnCandidateExitedAfterAdmission({
      pid: 42,
      code: 1,
      stderr: "",
    }))).toBe("The service process exited before becoming ready");
  });

  it("formats unavailable health with both attempt diagnostics and no internal error tags", () => {
    const message = formatAcnEnsuranceError(new AcnHealthUnavailable({
      owner: {
        pid: 42,
        processStartIdentity: ProcessStartIdentitySchema.make("start-1"),
        port: 14_000,
      },
      attempts: [
        new AcnHealthRequestFailed({ message: "connection refused" }),
        new AcnHealthResponseInvalid({ message: "expected magnitude-acn health" }),
      ],
    }));

    expect(message).toBe([
      "A live Magnitude service process was observed at http://127.0.0.1:14000/health (PID 42), but neither health check produced a valid response.",
      "Health check 1: request failed: connection refused",
      "Health check 2: response was invalid: expected magnitude-acn health",
      "Run `magnitude service start`.",
    ].join("\n"));
    expect(message).not.toContain("AcnHealthUnavailable");
    expect(message).not.toContain("AcnEnsuranceFailed");
  });

})
