---
"magnitude-test": patch
---

beforeAll, beforeEach, afterEach, afterAll hooks may now be registered within groups to scope them only to tests within the group. For the time being, afterAll hooks regardless of group will only run after all tests in a module are complete. This may change in the future.
