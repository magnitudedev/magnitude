# `@magnitudedev/openapi-effect`

Strict, Effect-native OpenAPI 3.1 compilation for Magnitude packages.

The package decodes an OpenAPI document with Effect Schema, normalizes it into a typed protocol IR,
and emits Effect Schema declarations, Effect `HttpApi` declarations, operation descriptors, and a
deterministic generation manifest. Unsupported semantics fail through typed Effect error channels.

```ts
import { Effect } from "effect";
import {
  compileOpenApi,
  OpenApiEffectConfig,
} from "@magnitudedev/openapi-effect";

const config = new OpenApiEffectConfig({ apiName: "ExampleApi" });
const generated = await Effect.runPromise(compileOpenApi(document, config));
```

The only untyped boundary is the raw document value passed to `compileOpenApi`; all later phases use
Schema-decoded values. The compiler performs no filesystem writes.
