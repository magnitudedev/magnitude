import { describe, expect, it } from "vitest"
import { RpcGroup, RpcTest } from "@effect/rpc"
import { Effect, Option, Schema } from "effect"
import type { OpenDialogOptions } from "electron"
import { ChooseOpenPath, InferenceHostRpcs, OpenPathRequest, OpenPathResult } from "./desktop-rpc"
import { chooseOpenPath, openDialogOptions } from "./open-dialog"

const folder = { title: "Add a folder of models", defaultPath: Option.none<string>() }
const storage = { title: "Choose a folder for downloaded models", defaultPath: Option.some("C:\\Users\\me\\models") }

describe("native open dialog", () => {
  it("opens one folder on every desktop OS", () => {
    expect(openDialogOptions(folder)).toEqual({ title: "Add a folder of models", properties: ["openDirectory", "createDirectory", "dontAddToRecent"] })
    expect(openDialogOptions(storage)).toEqual({ title: "Choose a folder for downloaded models", defaultPath: "C:\\Users\\me\\models", properties: ["openDirectory", "createDirectory", "dontAddToRecent"] })
  })
  it("answers the chosen path, none when cancelled, and a host error when the dialog fails", async () => {
    const shown: OpenDialogOptions[] = []
    const chosen = await Effect.runPromise(chooseOpenPath(async options => { shown.push(options); return { canceled: false, filePaths: ["/models"] } })(storage))
    expect(chosen).toEqual({ path: Option.some("/models") })
    expect(shown[0]?.properties).toContain("openDirectory")
    expect(await Effect.runPromise(chooseOpenPath(async () => ({ canceled: true, filePaths: [] }))(folder))).toEqual({ path: Option.none() })
    const failed = await Effect.runPromise(Effect.flip(chooseOpenPath(() => Promise.reject(new Error("PRIVATE")))(folder)))
    expect(failed).toMatchObject({ _tag: "HostError", message: "Magnitude couldn’t open the file dialog. Try again." })
  })
  it("serves ChooseOpenPath over the host RPC group", async () => {
    expect(InferenceHostRpcs.requests.get("ChooseOpenPath")).toBe(ChooseOpenPath)
    const group = RpcGroup.make(ChooseOpenPath)
    const handlers = group.toLayer({ ChooseOpenPath: chooseOpenPath(async options => ({ canceled: false, filePaths: [options.properties?.includes("openDirectory") ? "/Volumes/models" : "/elsewhere"] })) })
    const result = await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const client = yield* RpcTest.makeClient(group)
      return yield* client.ChooseOpenPath(folder)
    })).pipe(Effect.provide(handlers)))
    expect(result).toEqual({ path: Option.some("/Volumes/models") })
  })
  it("crosses the preload bridge as plain encoded values", () => {
    expect(Schema.encodeSync(OpenPathResult)({ path: Option.none() })).toEqual({})
    expect(Schema.decodeUnknownSync(OpenPathResult)(Schema.encodeSync(OpenPathResult)({ path: Option.some("/models") }))).toEqual({ path: Option.some("/models") })
    expect(Schema.decodeUnknownSync(OpenPathRequest)({ title: "Add a folder of models" })).toEqual({ title: "Add a folder of models", defaultPath: Option.none() })
  })
})
