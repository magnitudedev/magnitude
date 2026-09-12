BEGIN;
ALTER TABLE magnitude_distribution.artifacts DROP CONSTRAINT artifacts_package_check;
ALTER TABLE magnitude_distribution.artifacts ADD CONSTRAINT artifacts_package_check
  CHECK (package IN ('mac-zip', 'dmg', 'windows-exe', 'deb', 'rpm'));
CREATE TABLE magnitude_distribution.installer_daily (
  day date NOT NULL DEFAULT (now() AT TIME ZONE 'UTC')::date,
  release_version text NOT NULL,
  artifact_id text NOT NULL,
  os text NOT NULL,
  arch text NOT NULL,
  package text NOT NULL,
  country text NOT NULL DEFAULT '' CHECK (country = '' OR country ~ '^[A-Z]{2}$'),
  requests bigint NOT NULL DEFAULT 1,
  PRIMARY KEY(day, release_version, artifact_id, country)
);
REVOKE ALL ON magnitude_distribution.installer_daily FROM PUBLIC, anon, authenticated;
GRANT SELECT, INSERT, UPDATE ON magnitude_distribution.installer_daily TO magnitude_distribution_runtime;
COMMIT;
