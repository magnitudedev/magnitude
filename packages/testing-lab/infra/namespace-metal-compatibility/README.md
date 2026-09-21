# Namespace Metal compatibility profile

Namespace macOS workers expose Apple's host-backed `Apple Paravirtual device`, but the guest reports
Apple GPU family support conservatively. The lab compiles and runs `probe.swift` before enabling the
profile. It enables the shim only when the device is paravirtualized, Apple family 7 is not advertised,
and an actual SIMD-group reduction executes with the expected result.

`LumeMetalCapabilities.m` is the MIT-licensed Cua capability shim from upstream revision
`9bbfa7dd3e27ca7f1861ede70aaca390174493f9`. The lab configures only Apple family 7 and retains the
device's existing 32 KiB threadgroup-memory limit. This profile is passed to candidate processes on
qualified Namespace workers; it is not part of Magnitude production packages or physical-Mac policy.
