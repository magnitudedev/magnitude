<div align="center">
  <p>
    <img src="https://magnitude.run/logo.svg" alt="Magnitude Logo" width="100" style="vertical-align: middle; margin-right: 20px" />
  </p>

  <h3>
    <a href="https://magnitude.run/">Homepage</a> &nbsp;&nbsp;|&nbsp;&nbsp; 
    <a href="https://docs.magnitude.run/getting-started/introduction">Read the docs</a> &nbsp;&nbsp;|&nbsp;&nbsp; 
    <a href="https://discord.gg/VcdpMh9tTy">Discord</a>
  </h3>

  <hr style="height: 1px; border: none; background-color: #e1e4e8; margin: 24px 0;">
</div>

Magnitude is an SDK and platform for AI-powered E2E testing.

Use Magnitude to write flexible natural language E2E tests that don't break when your interface changes.

Replace your flaky Playwright tests, cover new flows quickly, all without having to install and manage browser infrastructure.

## Installation

```bash
npm install --save-dev magnitude-test
```

## Setup

Once installed, setup Magnitude in your project by running:
```
npx magnitude init
```
This will create a basic tests directory `tests/magnitude` with:
- `magnitude.config.ts`: Magnitude test configuration file
- `example.mag.ts`: An example test file

Then, you'll need to initialize the Magnitude client with your API key. You can get a free API key by signing up at https://app.magnitude.run/signup, then creating one in Settings->API Keys. Then configure it in one of two ways:

1. Configure in the generated `magnitude.config.ts`:
```ts
import { defineConfig } from 'magnitude-test';

export default defineConfig({
    baseUrl: "localhost:5173",
    // Add API key:
    apiKey: 'your-api-key-here'
});
```

2. Or set it as an environment variable:
```bash
export MAGNITUDE_API_KEY=your-api-key-here
```

## Running Test Cases

To run your Magnitude tests, simply run:
```
npx magnitude
```
This will run all Magnitude test files discovered with the `*.mag.ts` pattern.

### Automatically start your dev server

Add a `webServer` section in `magnitude.config.ts` to have Magnitude start your development server when tests run:

```ts
import { defineConfig } from 'magnitude-test';

export default defineConfig({
    baseUrl: 'http://localhost:3000',
    webServer: {
        command: 'npm run start',
        url: 'http://localhost:3000',
        reuseExistingServer: true
    }
});
```

Magnitude checks if the server is already running and skips the command when `reuseExistingServer` is `true`.

Here's an example of a basic test case:
```ts
// tests/example.mag.ts
import { test } from 'magnitude-test';

// Example URL override, defaults to configured baseUrl
test('can login with valid credentials', { url: "https://qa-bench.com" })
    .step('Log in to the app')
        .data({ username: "test-user@magnitude.run" }) // arbitrary key/values
        .secureData({ password: "test" }) // sensitive data
        .check('Can see dashboard') // natural language assertion
    .step('Create a new company')
        .data("Make up the first 2 values and use defaults for the rest")
        .check("Company added successfully");
```

Any step descriptions, checks, or data are represented in natural language. You can be as vague or specific as you'd like - though more specificity does generally lead to more consistent test runs.

For more examples see the [examples folder](./examples).

## Tunneling

Magnitude runs the browser and AI agent so you don't have to. In order to access locally running sites, a secure HTTP tunnel is established from our servers to your localhost when you specify a private URL (for example `localhost:3000` or `127.0.0.1`).

This is handled automatically - when you specify a private URL the SDK will detect this and establish a reverse tunnel first.

## How it Works

Magnitude uses multi-modal LLMs and state-of-the-art web interaction techniques to follow the intention of a natural language test case, rather than specific brittle instructions that rely on selectors (e.g. selenium or playwright).

The LLM decides at runtime what to do based on the test case steps, a screenshot of the browser, a history of its own activity, and other information. If it encounters something unexpected, a problem will be reported and categorized. If the problem encountered inhibits the further execution of the test case, the test case fails. Some problems identiifed may not justify a failure of the test case, but indicate some smaller visual issue or bug.

## Contact

To get a personalized demo or see how Magnitude can help your company, feel free to reach out to us at founders@magnitude.run

You can also join our Discord community for help or any suggestions! https://discord.gg/VcdpMh9tTy
