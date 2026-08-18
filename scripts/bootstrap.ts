import { $ } from "bun"

await $`git submodule update --init --recursive`

console.log("Initialized Git submodules recursively.")
