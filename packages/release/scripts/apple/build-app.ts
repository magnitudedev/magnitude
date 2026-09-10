import * as FileSystem from "@effect/platform/FileSystem"
import { Effect } from "effect"
import { resolve } from "node:path"
import { ACN_EXECUTABLE_NAME } from "../../src/executables"
import { MACOS_APP_NAME, MACOS_BUNDLE_ID } from "../../src/macos-app"
import { appleCommand, signAppleCode } from "./signing"

const resources = resolve(import.meta.dir, "../../resources/macos")

const plist = (version: string, revision: number) => `<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
<key>CFBundleIdentifier</key><string>${MACOS_BUNDLE_ID}</string>
<key>CFBundleName</key><string>Magnitude</string>
<key>CFBundleDisplayName</key><string>Magnitude</string>
<key>CFBundleExecutable</key><string>${ACN_EXECUTABLE_NAME}</string>
<key>CFBundlePackageType</key><string>APPL</string>
<key>CFBundleIconFile</key><string>Magnitude.icns</string>
<key>CFBundleShortVersionString</key><string>${version.split("-")[0]}</string>
<key>CFBundleVersion</key><string>${revision}</string>
<key>MagnitudeReleaseVersion</key><string>${version}</string>
<key>LSMinimumSystemVersion</key><string>13.0</string>
<key>LSUIElement</key><true/>
<key>NSHighResolutionCapable</key><true/>
</dict></plist>
`

/** The service is the bundle's main executable, so sealing the app signs it with the Bun profile. */
export const buildMacApp = (directory: string, acn: string, version: string, revision: number) => Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const app = resolve(directory, MACOS_APP_NAME)
  const contents = resolve(app, "Contents")
  for (const child of ["MacOS", "Resources"]) {
    yield* fs.makeDirectory(resolve(contents, child), { recursive: true })
  }
  yield* fs.writeFileString(resolve(contents, "Info.plist"), plist(version, revision))
  yield* fs.copyFile(acn, resolve(contents, "MacOS", ACN_EXECUTABLE_NAME))
  yield* fs.chmod(resolve(contents, "MacOS", ACN_EXECUTABLE_NAME), 0o755)
  yield* fs.copyFile(resolve(resources, "Magnitude.icns"), resolve(contents, "Resources/Magnitude.icns"))
  yield* signAppleCode(app, MACOS_BUNDLE_ID, "bun")
  yield* appleCommand("/usr/bin/codesign", "--verify", "--deep", "--strict", app)
  return app
})
