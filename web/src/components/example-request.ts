/** One request to an inference route: its path under the base URL and its JSON body. */
export interface ExampleRoute {
  readonly path: string
  readonly body: unknown
}

export const chatCompletionExample = (model: string): ExampleRoute => ({
  path: "/chat/completions",
  body: { model, messages: [{ role: "user", content: "Hello" }] },
})

/** The decision every example asks, written once and rendered into each wire format. */
const ticket = "I was charged twice for my order last week. Please refund one of the payments."
const billing = { name: "billing", instructions: "Is this message about billing?" }
const route = {
  name: "route",
  instructions: "Which team should handle this message?",
  options: [["billing", "Payments, charges and refunds"], ["shipping", "Delivery and tracking"], ["technical", "Bugs and errors"]] as const,
}
const urgency = { name: "urgency", instructions: "How urgent is this message?", levels: ["Not urgent", "Somewhat urgent", "Urgent"] as const }

/** TypeSafe System One: questions keyed by ID, options as a map of key to description. */
export const systemOneExample = (model: string) => ({
  path: "/systemone",
  body: {
    model,
    state: ticket,
    questions: {
      [billing.name]: { type: "noul", instructions: billing.instructions },
      [route.name]: { type: "choice", instructions: route.instructions, criteria: Object.fromEntries(route.options) },
      [urgency.name]: { type: "score", instructions: urgency.instructions, criteria: urgency.levels },
    },
  },
})

/** OpenAI Decisions: questions in order, each with an optional name echoed in its answer. */
export const decisionsExample = (model: string) => ({
  path: "/decisions",
  body: {
    model,
    input: ticket,
    questions: [
      { type: "predicate", name: billing.name, instructions: billing.instructions },
      { type: "choice", name: route.name, instructions: route.instructions, choices: route.options.map(([value, description]) => ({ value, description })) },
      { type: "score", name: urgency.name, instructions: urgency.instructions, levels: urgency.levels.map(label => ({ label })) },
    ],
  },
})

/**
 * A request for the user's shell: POSIX quoting, or PowerShell on Windows. From another device it
 * sends the Network access key, shown as a placeholder.
 */
export const exampleRequest = (baseUrl: string, route: ExampleRoute, platform: string, withKey = false): string => {
  const body = JSON.stringify(route.body).replaceAll("'", platform === "win32" ? "''" : "'\\''")
  return platform === "win32"
    ? `Invoke-RestMethod ${baseUrl}${route.path} -Method Post -ContentType "application/json"${withKey ? ' -Headers @{ Authorization = "Bearer YOUR_KEY" }' : ""} -Body '${body}'`
    : `curl ${baseUrl}${route.path} \\\n${withKey ? '  -H "Authorization: Bearer YOUR_KEY" \\\n' : ""}  -H "Content-Type: application/json" \\\n  -d '${body}'`
}

/** A JSON value as a Python literal: the examples hold only strings, arrays and objects, whose JSON is valid Python. */
const python = (value: unknown, indent: string) => JSON.stringify(value, null, 4).replaceAll("\n", `\n${indent}`)
/** The SDKs need a key; locally any key works. */
const pythonKey = (withKey: boolean) => withKey ? "YOUR_KEY" : "local"

/**
 * The System One example through TypeSafe's Python SDK. Its base URL is the API root, without `/v1`,
 * since it appends `/v1/systemone` itself; it accepts questions as dictionaries in the wire format.
 */
export const systemOnePython = (apiRoot: string, model: string, withKey = false): string => {
  const { body } = systemOneExample(model)
  return `from typesafe_sdk import TypeSafeClient

client = TypeSafeClient(api_key="${pythonKey(withKey)}", base_url="${apiRoot}")
result = client.system_one(
    model=${JSON.stringify(model)},
    state=${JSON.stringify(body.state)},
    questions=${python(body.questions, "    ")},
)
print(result.nouls["${billing.name}"].noul, result.choices["${route.name}"].choice)`
}

/** The Decisions example through the OpenAI Python SDK pointed at the base URL. */
export const decisionsPython = (baseUrl: string, model: string, withKey = false): string => {
  const { body } = decisionsExample(model)
  return `from openai import OpenAI

client = OpenAI(api_key="${pythonKey(withKey)}", base_url="${baseUrl}")
response = client.decisions.create(
    model=${JSON.stringify(model)},
    input=${JSON.stringify(body.input)},
    questions=${python(body.questions, "    ")},
)
for answer in response.answers:
    print(answer)`
}
