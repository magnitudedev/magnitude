import { Duration, Option } from "effect";
import { describe, expect, it } from "vitest";
import { IcnProcessOptions, renderIcnArguments } from "./process.js";

const options = () =>
  new IcnProcessOptions({
    executable: "/opt/magnitude/magnitude-icn",
    modelPath: Option.some("/models/model.gguf"),
    modelAlias: Option.some("coding-model"),
    host: "127.0.0.1",
    port: 18_082,
    contextSize: 8_192,
    batchSize: 512,
    ubatchSize: 256,
    maxSequences: 4,
    prefillQuantum: 128,
    gpuLayers: 999,
    threads: Option.some(8),
    threadsBatch: Option.none(),
    flashAttention: "auto",
    startupTimeout: Duration.seconds(90),
    outputLimitBytes: 64 * 1_024,
  });

describe("IcnProcessOptions", () => {
  it("renders the complete typed executor configuration without implicit defaults", () => {
    expect(renderIcnArguments(options())).toEqual([
      "serve",
      "--bind",
      "127.0.0.1:18082",
      "--model",
      "/models/model.gguf",
      "--model-alias",
      "coding-model",
      "--context-size",
      "8192",
      "--batch-size",
      "512",
      "--ubatch-size",
      "256",
      "--max-sequences",
      "4",
      "--prefill-quantum",
      "128",
      "--gpu-layers",
      "999",
      "--threads",
      "8",
      "--flash-attention",
      "auto",
    ]);
  });

  it("brackets the IPv6 loopback address", () => {
    const ipv6 = new IcnProcessOptions({ ...options(), host: "::1" });
    expect(renderIcnArguments(ipv6).slice(0, 3)).toEqual([
      "serve",
      "--bind",
      "[::1]:18082",
    ]);
  });

  it("starts a model-free singleton without model-scoped arguments", () => {
    const singleton = new IcnProcessOptions({
      ...options(),
      modelPath: Option.none(),
      modelAlias: Option.none(),
    });
    expect(renderIcnArguments(singleton)).not.toContain("--model");
    expect(renderIcnArguments(singleton)).not.toContain("--model-alias");
  });
});
