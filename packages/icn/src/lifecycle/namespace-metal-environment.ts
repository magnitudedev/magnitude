import { isAbsolute, normalize } from "node:path";

/** The lab's qualified Namespace receipt applies only at the native inference process boundary. */
export const namespaceMetalInferenceEnvironment = (
  inherited: NodeJS.ProcessEnv = process.env,
  platform: NodeJS.Platform = process.platform,
): Readonly<Record<string, string>> => {
  if (platform !== "darwin") return {};
  const shim = inherited.LAB_NAMESPACE_METAL_SHIM;
  if (!shim || !isAbsolute(shim) ||
      !normalize(shim).startsWith("/Users/runner/lab-runtime/metal-compatibility/") ||
      inherited.LAB_NAMESPACE_METAL_FAMILY_MAX !== "1007" ||
      inherited.LAB_NAMESPACE_METAL_MAX_THREADGROUP_MEMORY !== "32768" ||
      inherited.DYLD_INSERT_LIBRARIES !== undefined) return {};
  return {
    DYLD_INSERT_LIBRARIES: shim,
    LUME_METAL_PROCESS_NAME: "magnitude-inference",
    LUME_METAL_APPLE_FAMILY_MAX: "1007",
    LUME_METAL_MAX_THREADGROUP_MEMORY: "32768",
  };
};
