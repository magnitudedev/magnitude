import { describe, expect, it } from "vitest";
import { namespaceMetalInferenceEnvironment } from "./namespace-metal-environment.js";

const receipt = {
  LAB_NAMESPACE_METAL_SHIM: "/Users/runner/lab-runtime/metal-compatibility/LumeMetalCapabilities-arm64.dylib",
  LAB_NAMESPACE_METAL_FAMILY_MAX: "1007",
  LAB_NAMESPACE_METAL_MAX_THREADGROUP_MEMORY: "32768",
};

describe("Namespace Metal inference launch", () => {
  it("loads the qualified shim only at the macOS inference boundary", () => {
    expect(namespaceMetalInferenceEnvironment(receipt, "darwin")).toEqual({
      DYLD_INSERT_LIBRARIES: receipt.LAB_NAMESPACE_METAL_SHIM,
      LUME_METAL_PROCESS_NAME: "magnitude-inference",
      LUME_METAL_APPLE_FAMILY_MAX: "1007",
      LUME_METAL_MAX_THREADGROUP_MEMORY: "32768",
    });
    expect(namespaceMetalInferenceEnvironment(receipt, "linux")).toEqual({});
  });

  it("does not load unqualified or conflicting profiles", () => {
    expect(namespaceMetalInferenceEnvironment({}, "darwin")).toEqual({});
    expect(namespaceMetalInferenceEnvironment({ ...receipt, LAB_NAMESPACE_METAL_FAMILY_MAX: "1009" }, "darwin")).toEqual({});
    expect(namespaceMetalInferenceEnvironment({ ...receipt, DYLD_INSERT_LIBRARIES: "/tmp/other.dylib" }, "darwin")).toEqual({});
  });
});
