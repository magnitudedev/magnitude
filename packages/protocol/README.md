# Protocol Notes

## Optional Wire Fields Use `Option`

Protocol schemas cross a JSON boundary, and `undefined` does not serialize. Do not use `Schema.optional(...)` for optional payload/result/event fields. Use `Option`-decoded fields instead:

```ts
field: Schema.optionalWith(Schema.String, { as: "Option", exact: true })
```

Only use defaults when omission should decode to a concrete value, such as `limit: 50`.
