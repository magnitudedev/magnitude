import { describe, expect, test } from "vitest"
import {
  developmentBuildEnvironment,
  developmentBuildProfile,
} from "./build-local"
import { readCargoMessages, runCargoBuild } from "./compile"
import { ICN_EXECUTABLE_NAME } from "@magnitudedev/release/executables"

const stream = (...chunks: readonly string[]): ReadableStream<Uint8Array> => {
  const encoder = new TextEncoder()
  return new ReadableStream({
    start(controller) {
      for (const chunk of chunks) controller.enqueue(encoder.encode(chunk))
      controller.close()
    },
  })
}

describe("ICN compilation", () => {
  test.each(["all", "errors"] as const)("retains stderr and structured diagnostics in %s mode when Cargo fails", async diagnostics => {
    const diagnostic = JSON.stringify({ reason: "compiler-message", message: { rendered: "tensor binding failed" } })
    await expect(runCargoBuild([process.execPath, "-e", `console.log(${JSON.stringify(diagnostic)}); process.stderr.write("native linker failed"); process.exitCode = 101`], {
      cwd: process.cwd(), env: process.env, diagnostics,
    })).rejects.toThrow(/native linker failed[\s\S]*tensor binding failed/)
  })

  test("targets only attached GPUs for local CUDA builds", () => {
    expect(developmentBuildEnvironment("cuda")).toEqual({
      CMAKE_CUDA_ARCHITECTURES: "native",
      LLAMA_CPU_ALL_VARIANTS: "0",
    })
    expect(developmentBuildEnvironment("cpu")).toEqual({})
    expect(developmentBuildEnvironment("metal")).toEqual({
      LLAMA_CPU_ALL_VARIANTS: "0",
    })
    expect(developmentBuildEnvironment("vulkan")).toEqual({
      LLAMA_CPU_ALL_VARIANTS: "0",
    })
    expect(developmentBuildProfile("cuda")).toBe("development-cuda-native")
    expect(developmentBuildProfile("cpu")).toBe("development-cpu")
  })

  test("retains streamed Cargo messages and emits rendered diagnostics", async () => {
    const rendered: string[] = []
    const messages = await readCargoMessages(
      stream(
        '{"reason":"compiler-message","message":{"rendered":"warn',
        'ing\\n"}}\n{"reason":"build-script-executed",',
        '"package_id":"native","out_dir":"/output"}\n',
      ),
      (diagnostic) => rendered.push(diagnostic),
    )

    expect(messages).toEqual([
      {
        reason: "build-script-executed",
        package_id: "native",
        out_dir: "/output",
      },
    ])
    expect(rendered).toEqual(["warning\n"])
  })

  test("retains a final Cargo message without a trailing newline", async () => {
    await expect(
      readCargoMessages(
        stream(
          `{"reason":"compiler-artifact","target":{"name":"${ICN_EXECUTABLE_NAME}"}}`,
        ),
        () => {},
      ),
    ).resolves.toEqual([
      {
        reason: "compiler-artifact",
        target: { name: ICN_EXECUTABLE_NAME },
      },
    ])
  })
})
