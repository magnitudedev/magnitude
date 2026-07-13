export interface TargetInfo {
  readonly target: string
  readonly platform: string
  readonly arch: string
  readonly platformKey: string
  readonly executableExt: string
}

export function getDefaultBunTarget(): string {
  const platform = process.platform === 'darwin'
    ? 'darwin'
    : process.platform === 'win32'
      ? 'windows'
      : 'linux'
  const arch = process.arch === 'arm64' ? 'arm64' : 'x64'
  return `bun-${platform}-${arch}`
}

export function getTargetInfo(target: string): TargetInfo {
  const parts = target.replace(/^bun-/, '').split('-')
  const platform = parts[0]
  const arch = parts[1]
  if (!platform || !arch) throw new Error(`Invalid Bun target: ${target}`)
  return {
    target,
    platform,
    arch,
    platformKey: `${platform}-${arch}`,
    executableExt: platform === 'windows' ? '.exe' : '',
  }
}

export function bunTargetToRipgrepTarget(target: string): string {
  const { platform, arch } = getTargetInfo(target)
  const map: Record<string, Record<string, string>> = {
    darwin: { arm64: 'aarch64-apple-darwin', x64: 'x86_64-apple-darwin' },
    linux: { x64: 'x86_64-unknown-linux-musl', arm64: 'aarch64-unknown-linux-musl' },
    windows: { x64: 'x86_64-pc-windows-msvc' },
  }
  const ripgrepTarget = map[platform]?.[arch]
  if (!ripgrepTarget) throw new Error(`No ripgrep target for ${target}`)
  return ripgrepTarget
}
