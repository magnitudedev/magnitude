import { describe, expect, it } from "vitest";
import { makeHealthResponse } from "./identity";

describe("ACN identity", () => {
  it("exposes the exact registry owner identity through health", () => {
    expect(makeHealthResponse("1.2.3", "owner-1", 1234)).toEqual({
      service: "magnitude-acn",
      version: "1.2.3",
      id: "owner-1",
      pid: 1234,
    });
  });
});
