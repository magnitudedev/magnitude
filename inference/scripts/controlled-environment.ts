const inheritedNames = [
  "PATH",
  "HOME",
  "CARGO_HOME",
  "RUSTUP_HOME",
  "TMPDIR",
  "TEMP",
  "TMP",
  "SDKROOT",
  "DEVELOPER_DIR",
  "MACOSX_DEPLOYMENT_TARGET",
  "CUDA_HOME",
  "CUDA_PATH",
  "CUDACXX",
  "CUDAToolkit_ROOT",
  "VULKAN_SDK",
  "PKG_CONFIG_PATH",
  "CMAKE_PREFIX_PATH",
  "LIBCLANG_PATH",
  "LD_LIBRARY_PATH",
  "DYLD_LIBRARY_PATH",
  "SystemRoot",
  "CC",
  "CXX",
  "AR",
  "RANLIB",
  "CMAKE_GENERATOR",
  "CMAKE_MAKE_PROGRAM",
] as const;

export const controlledEnvironment = (
  overrides: Readonly<Record<string, string>>,
  source: Readonly<Record<string, string | undefined>> = process.env
): Readonly<Record<string, string>> => {
  const environment: Record<string, string> = {};
  for (const name of inheritedNames) {
    const value = source[name];
    if (value !== undefined && value.length > 0) environment[name] = value;
  }
  return {
    ...environment,
    LC_ALL: "C",
    LANG: "C",
    TZ: "UTC",
    ...overrides,
  };
};

export const controlledEnvironmentEvidence = (
  environment: Readonly<Record<string, string>>
) => {
  const entries = Object.entries(environment).sort(([left], [right]) =>
    left.localeCompare(right)
  );
  const hasher = new Bun.CryptoHasher("sha256");
  for (const [name, value] of entries) {
    hasher.update(name);
    hasher.update("\0");
    hasher.update(value);
    hasher.update("\0");
  }
  return {
    policy: "allowlist-v1",
    names: entries.map(([name]) => name),
    sha256: hasher.digest("hex"),
  };
};

