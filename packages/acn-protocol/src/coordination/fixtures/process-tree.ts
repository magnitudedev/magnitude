const child = Bun.spawn([process.execPath, "-e", "setInterval(() => {}, 1000)"], {
  stdin: "ignore",
  stdout: "ignore",
  stderr: "ignore",
})

process.stdout.write(`${process.pid}:${child.pid}\n`)
setInterval(() => {}, 1_000)
