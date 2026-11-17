import { Option } from "effect"
import { renderToStaticMarkup } from "react-dom/server"
import { describe, expect, it } from "vitest"
import { RestartNotice, reachableAt, restartReason } from "./restart-notice"

describe("reachable address", () => {
  it("is none right after network access is turned on, while the service still listens on loopback only", () => {
    expect(reachableAt({ enabled: true, bind: Option.none(), reachableAddresses: [] })).toEqual(Option.none())
  })
  it("is the address the service listens on, not a newly saved one", () => {
    expect(reachableAt({ enabled: true, bind: Option.some("100.70.1.2"), reachableAddresses: ["192.168.1.67"] })).toEqual(Option.some("192.168.1.67"))
  })
  it("is the saved address once the service listens on it among others", () => {
    expect(reachableAt({ enabled: true, bind: Option.some("100.70.1.2"), reachableAddresses: ["192.168.1.67", "100.70.1.2"] })).toEqual(Option.some("100.70.1.2"))
  })
  it("is none once network access is turned off, though other devices can connect until a restart", () => {
    expect(reachableAt({ enabled: false, bind: Option.none(), reachableAddresses: ["192.168.1.67"] })).toEqual(Option.none())
  })
})

describe("restart reason", () => {
  const network = (enabled: boolean, reachableAddresses: ReadonlyArray<string>) => Option.some({ enabled, bind: Option.none(), reachableAddresses })
  it("says network access starts only after a restart when it was just turned on", () => {
    expect(restartReason({ storage: false, network: network(true, []) })).toBe("Network access starts when Magnitude restarts. Until then, only this computer can connect.")
  })
  it("says other devices can still connect when it was just turned off", () => {
    expect(restartReason({ storage: false, network: network(false, ["192.168.1.67"]) })).toBe("Other devices can still connect until Magnitude restarts.")
  })
  it("names both changes when the model folder changed too", () => {
    expect(restartReason({ storage: true, network: network(true, ["192.168.1.67"]) })).toBe("Magnitude is still using the previous model folder and network settings.")
  })
})

describe("restart notice", () => {
  const render = (canRestart: boolean) => renderToStaticMarkup(<RestartNotice reason="Network access starts when Magnitude restarts." canRestart={canRestart} failed={false} restarting={false} onRestart={() => {}} />)
  it("sits in the flow of the page or dialog that shows it instead of floating over its content", () => {
    expect(render(true)).not.toMatch(/\b(fixed|absolute|sticky)\b/)
  })
  it("offers the restart where the service can be restarted, and says how otherwise", () => {
    expect(render(true)).toContain("Restart Magnitude")
    expect(render(false)).not.toContain("Restart Magnitude")
    expect(render(false)).toContain("Restart the Magnitude server to apply them.")
  })
})
