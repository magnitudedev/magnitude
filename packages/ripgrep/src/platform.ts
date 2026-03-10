export const RIPGREP_VERSION = 'v15.0.0'
export const MULTI_ARCH_LINUX_VERSION = 'v13.0.0-4'

const MULTI_ARCH_TARGETS = new Set([
  'arm-unknown-linux-gnueabihf',
  'powerpc64le-unknown-linux-gnu',
  's390x-unknown-linux-gnu',
])

export function isWindows(): boolean {
  return process.platform === 'win32'
}

export function getTarget(): string {
  const arch = process.env.npm_config_arch || process.arch

  switch (process.platform) {
    case 'darwin':
      return arch === 'arm64' ? 'aarch64-apple-darwin' : 'x86_64-apple-darwin'
    case 'win32':
      return arch === 'x64'
        ? 'x86_64-pc-windows-msvc'
        : arch === 'arm64'
          ? 'aarch64-pc-windows-msvc'
          : 'i686-pc-windows-msvc'
    case 'linux':
      return arch === 'x64'
        ? 'x86_64-unknown-linux-musl'
        : arch === 'arm'
          ? 'arm-unknown-linux-gnueabihf'
          : arch === 'armv7l'
            ? 'arm-unknown-linux-gnueabihf'
            : arch === 'arm64'
              ? 'aarch64-unknown-linux-musl'
              : arch === 'ppc64'
                ? 'powerpc64le-unknown-linux-gnu'
                : arch === 'riscv64'
                  ? 'riscv64gc-unknown-linux-gnu'
                  : arch === 's390x'
                    ? 's390x-unknown-linux-gnu'
                    : 'i686-unknown-linux-musl'
    default:
      throw new Error(`[ripgrep] Unsupported platform: ${process.platform}`)
  }
}

export function getVersion(target: string): string {
  return MULTI_ARCH_TARGETS.has(target) ? MULTI_ARCH_LINUX_VERSION : RIPGREP_VERSION
}