-- CI supplies a separately provisioned login password. Publishing cannot read installation telemetry.
DO $$ BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'magnitude_distribution_publisher') THEN
    CREATE ROLE magnitude_distribution_publisher NOLOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOINHERIT NOBYPASSRLS CONNECTION LIMIT 5;
  END IF;
END $$;
GRANT USAGE ON SCHEMA magnitude_distribution TO magnitude_distribution_publisher;
GRANT SELECT, INSERT ON magnitude_distribution.releases, magnitude_distribution.artifacts TO magnitude_distribution_publisher;
GRANT SELECT, INSERT, UPDATE ON magnitude_distribution.channels TO magnitude_distribution_publisher;
