BEGIN;
CREATE TABLE magnitude_distribution.daily_totals (
  day date PRIMARY KEY,
  active_installations bigint NOT NULL,
  checks bigint NOT NULL,
  download_requests bigint NOT NULL
);
CREATE TABLE magnitude_distribution.daily_dimensions (
  day date NOT NULL,
  version text NOT NULL,
  os text NOT NULL,
  os_version text NOT NULL,
  arch text NOT NULL,
  package text NOT NULL,
  distro text NOT NULL,
  distro_version text NOT NULL,
  country text NOT NULL,
  active_installations bigint NOT NULL,
  checks bigint NOT NULL,
  PRIMARY KEY (day, version, os, os_version, arch, package, distro, distro_version, country)
);
CREATE TABLE magnitude_distribution.artifact_totals (
  day date NOT NULL,
  release_version text NOT NULL,
  artifact_id text NOT NULL,
  os text NOT NULL,
  arch text NOT NULL,
  country text NOT NULL,
  installations bigint NOT NULL,
  requests bigint NOT NULL,
  PRIMARY KEY (day, release_version, artifact_id, os, arch, country)
);
REVOKE ALL ON magnitude_distribution.daily_totals, magnitude_distribution.daily_dimensions, magnitude_distribution.artifact_totals FROM PUBLIC, anon, authenticated;

-- Only the database scheduler/operator can invoke retention. Runtime request credentials cannot.
CREATE FUNCTION magnitude_distribution.maintain() RETURNS void
LANGUAGE plpgsql SET search_path = pg_catalog AS $$
DECLARE cutoff date := (now() AT TIME ZONE 'UTC')::date - 90;
BEGIN
  IF NOT pg_try_advisory_xact_lock(hashtextextended('magnitude_distribution.maintain', 0)) THEN RETURN; END IF;
  DELETE FROM magnitude_distribution.request_nonces WHERE expires_at < now();
  INSERT INTO magnitude_distribution.daily_totals (day, active_installations, checks, download_requests)
    SELECT coalesce(i.day,d.day), coalesce(i.installations,0), coalesce(i.checks,0), coalesce(d.requests,0)
    FROM (SELECT day,count(DISTINCT installation_id) AS installations,sum(checks) AS checks
      FROM magnitude_distribution.installation_daily WHERE day < cutoff GROUP BY day) i
    FULL JOIN (SELECT day,sum(requests) AS requests FROM magnitude_distribution.artifact_daily WHERE day < cutoff GROUP BY day) d ON i.day=d.day
    ON CONFLICT (day) DO UPDATE SET active_installations=EXCLUDED.active_installations, checks=EXCLUDED.checks, download_requests=EXCLUDED.download_requests;
  INSERT INTO magnitude_distribution.daily_dimensions
    SELECT day,version,os,os_version,arch,package,distro,distro_version,country,count(DISTINCT installation_id),sum(checks)
    FROM magnitude_distribution.installation_daily WHERE day < cutoff
    GROUP BY day,version,os,os_version,arch,package,distro,distro_version,country
    ON CONFLICT (day,version,os,os_version,arch,package,distro,distro_version,country)
    DO UPDATE SET active_installations=EXCLUDED.active_installations,checks=EXCLUDED.checks;
  INSERT INTO magnitude_distribution.artifact_totals
    SELECT day,release_version,artifact_id,os,arch,country,count(DISTINCT installation_id),sum(requests)
    FROM magnitude_distribution.artifact_daily WHERE day < cutoff GROUP BY day,release_version,artifact_id,os,arch,country
    ON CONFLICT (day,release_version,artifact_id,os,arch,country) DO UPDATE SET installations=EXCLUDED.installations,requests=EXCLUDED.requests;
  DELETE FROM magnitude_distribution.installation_daily WHERE day < cutoff;
  DELETE FROM magnitude_distribution.artifact_daily WHERE day < cutoff;
  DELETE FROM magnitude_distribution.installations WHERE last_seen < now() - interval '180 days'
    AND NOT EXISTS (SELECT 1 FROM magnitude_distribution.installation_daily d WHERE d.installation_id=installations.installation_id)
    AND NOT EXISTS (SELECT 1 FROM magnitude_distribution.artifact_daily d WHERE d.installation_id=installations.installation_id);
END;
$$;
REVOKE ALL ON FUNCTION magnitude_distribution.maintain() FROM PUBLIC, anon, authenticated;
COMMIT;
