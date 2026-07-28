// @ts-expect-error Bun resolves this executable through its compile-time file loader.
export { default as rgPath } from "../bin/rg" with { type: "file" };
