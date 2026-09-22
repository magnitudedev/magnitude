# Remote access

Magnitude serves its inference API on port 10100 of the computer running it. By default only that computer can reach it.

## Base URLs

- OpenAI-compatible: `http://ADDRESS:10100/inference/v1`
- Anthropic-compatible: `http://ADDRESS:10100/inference/anthropic`

On the same computer, `ADDRESS` is `127.0.0.1` and no API key is needed. Send any value if the client requires one.

## From another device

The user turns on Network access in Magnitude Settings. That setting shows the address to use and an API key. Send the key as `Authorization: Bearer KEY` or `x-api-key: KEY`. Only inference is available from other devices; model management and `/rpc` are not.

Check the connection:

```sh
curl -H "Authorization: Bearer KEY" http://ADDRESS:10100/inference/v1/models
```

| Response | Meaning |
| --- | --- |
| Connection refused | Network access is off, or the chosen address does not include this network. |
| 401 | Missing or changed API key. |
| 421 | Hostname not accepted. Use an IP address, or the user adds the name to `network.allowedHosts` in `~/.magnitude/config.json`. |

## WSL

Inside WSL on Windows, `127.0.0.1` is the Linux distribution, not Windows. With WSL mirrored networking, `http://127.0.0.1:10100` works. Otherwise use the Windows host address from `ip route show default | awk '{print $3}'` with Network access on and the API key.
