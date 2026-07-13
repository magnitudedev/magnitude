import { Context, Layer } from 'effect'

export interface VersionShape {
  getVersion(): string
}

export class Version extends Context.Tag('Version')<Version, VersionShape>() {}

export function VersionLive(version: string): Layer.Layer<Version> {
  return Layer.succeed(
    Version,
    Version.of({
      getVersion: () => version,
    })
  )
}
