// Needed for React to work properly
process.env.NODE_ENV = process.env.NODE_ENV || "production";
export { test } from "@/discovery/testDeclaration";
export {
  beforeAll,
  afterAll,
  beforeEach,
  afterEach,
} from "@/discovery/testDeclaration";
export { type MagnitudeConfig, type WebServerConfig } from "@/discovery/types";
