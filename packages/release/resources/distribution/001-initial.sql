BEGIN;
CREATE SCHEMA IF NOT EXISTS magnitude_distribution;
REVOKE ALL ON SCHEMA magnitude_distribution FROM PUBLIC;

CREATE TABLE IF NOT EXISTS magnitude_distribution.releases (
  version text PRIMARY KEY,
  channel text NOT NULL CHECK (channel IN ('stable', 'beta', 'alpha')),
  source_commit text NOT NULL CHECK (source_commit ~ '^[a-f0-9]{40}$'),
  published_at timestamptz NOT NULL DEFAULT now(),
  withdrawn boolean NOT NULL DEFAULT false
);
CREATE TABLE IF NOT EXISTS magnitude_distribution.artifacts (
  release_version text NOT NULL REFERENCES magnitude_distribution.releases(version),
  artifact_id text NOT NULL,
  os text NOT NULL CHECK (os IN ('darwin', 'windows', 'linux')),
  arch text NOT NULL CHECK (arch IN ('arm64', 'x64')),
  package text NOT NULL CHECK (package IN ('mac-zip', 'windows-exe', 'deb', 'rpm')),
  envelope jsonb NOT NULL,
  PRIMARY KEY (release_version, artifact_id),
  UNIQUE (release_version, os, arch, package)
);
CREATE TABLE IF NOT EXISTS magnitude_distribution.channels (
  channel text NOT NULL CHECK (channel IN ('stable', 'beta', 'alpha')),
  os text NOT NULL,
  arch text NOT NULL,
  package text NOT NULL,
  release_version text NOT NULL,
  artifact_id text NOT NULL,
  PRIMARY KEY (channel, os, arch, package),
  FOREIGN KEY (release_version, artifact_id) REFERENCES magnitude_distribution.artifacts(release_version, artifact_id)
);
CREATE TABLE IF NOT EXISTS magnitude_distribution.installations (
  installation_id text PRIMARY KEY CHECK (installation_id ~ '^[a-f0-9]{64}$'),
  first_seen timestamptz NOT NULL DEFAULT now(),
  last_seen timestamptz NOT NULL DEFAULT now(),
  version text NOT NULL,
  os text NOT NULL,
  os_version text NOT NULL,
  arch text NOT NULL,
  package text NOT NULL,
  distro text NOT NULL DEFAULT '',
  distro_version text NOT NULL DEFAULT '',
  country text NOT NULL DEFAULT '' CHECK (country = '' OR country ~ '^[A-Z]{2}$')
);
CREATE INDEX IF NOT EXISTS installations_last_seen ON magnitude_distribution.installations(last_seen);
CREATE TABLE IF NOT EXISTS magnitude_distribution.installation_daily (
  day date NOT NULL DEFAULT (now() AT TIME ZONE 'UTC')::date,
  installation_id text NOT NULL REFERENCES magnitude_distribution.installations(installation_id),
  version text NOT NULL,
  os text NOT NULL,
  os_version text NOT NULL,
  arch text NOT NULL,
  package text NOT NULL,
  distro text NOT NULL DEFAULT '',
  distro_version text NOT NULL DEFAULT '',
  country text NOT NULL DEFAULT '',
  checks bigint NOT NULL DEFAULT 1,
  first_seen timestamptz NOT NULL DEFAULT now(),
  last_seen timestamptz NOT NULL DEFAULT now(),
  offered_version text,
  PRIMARY KEY (day, installation_id, version, os, os_version, arch, package, distro, distro_version, country)
);
CREATE TABLE IF NOT EXISTS magnitude_distribution.request_nonces (
  installation_id text NOT NULL CHECK (installation_id ~ '^[a-f0-9]{64}$'),
  nonce text NOT NULL,
  expires_at timestamptz NOT NULL,
  PRIMARY KEY (installation_id, nonce)
);
CREATE INDEX IF NOT EXISTS request_nonces_expiry ON magnitude_distribution.request_nonces(expires_at);

-- This schema is not exposed to PostgREST. Runtime roles receive explicit privileges.
REVOKE ALL ON ALL TABLES IN SCHEMA magnitude_distribution FROM PUBLIC, anon, authenticated;
GRANT USAGE ON SCHEMA magnitude_distribution TO magnitude_distribution_runtime;
GRANT SELECT ON magnitude_distribution.releases, magnitude_distribution.artifacts, magnitude_distribution.channels TO magnitude_distribution_runtime;
GRANT SELECT, INSERT, UPDATE ON magnitude_distribution.installations, magnitude_distribution.installation_daily TO magnitude_distribution_runtime;
GRANT SELECT, INSERT ON magnitude_distribution.request_nonces TO magnitude_distribution_runtime;
COMMIT;
