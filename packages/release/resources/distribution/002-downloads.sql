BEGIN;
CREATE TABLE magnitude_distribution.artifact_daily (
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
  release_version text NOT NULL,
  artifact_id text NOT NULL,
  requests bigint NOT NULL DEFAULT 1,
  first_seen timestamptz NOT NULL DEFAULT now(),
  last_seen timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (day, installation_id, version, os, os_version, arch, package, distro, distro_version, country, release_version, artifact_id)
);
REVOKE ALL ON magnitude_distribution.artifact_daily FROM PUBLIC, anon, authenticated;
GRANT SELECT, INSERT, UPDATE ON magnitude_distribution.artifact_daily TO magnitude_distribution_runtime;
COMMIT;
