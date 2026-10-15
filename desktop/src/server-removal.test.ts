import { expect, it } from "vitest"
import { serverRemovalCommand, serverRemovalFailure } from "./server-removal"

it("asks macOS for an administrator with a prompt that says what Magnitude is doing", () => {
  const command = serverRemovalCommand("darwin", "/Applications/Magnitude.app/Contents/Resources/magnitude")
  expect(command._tag).toBe("StandardCommand")
  if (command._tag !== "StandardCommand") return
  expect(command.command).toBe("/usr/bin/osascript")
  expect(command.args[1]).toBe('do shell script quoted form of "/Applications/Magnitude.app/Contents/Resources/magnitude" & " _server-remove" '
    + 'with prompt "Magnitude needs to remove the server set up on this Mac so the app can run. The server\'s data is kept." with administrator privileges')
})

it("says a dismissed administrator prompt was cancelled rather than asking for an administrator", () => {
  expect(serverRemovalFailure("darwin", 1, "0:72: execution error: User canceled. (-128)\n")).toBe("Removing the server was cancelled, so it is still running.")
  expect(serverRemovalFailure("linux", 126, "")).toBe("Removing the server was cancelled, so it is still running.")
  expect(serverRemovalFailure("darwin", 1, "execution error: The administrator user name or password was incorrect. (-60007)")).toContain("needs an administrator")
  expect(serverRemovalFailure("linux", 127, "")).toContain("needs an administrator")
})
