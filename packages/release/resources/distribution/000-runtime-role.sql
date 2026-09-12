-- Provision a login password separately and store its pooled URL only in server secrets.
-- Schema initialization itself must never contain a password.
DO $$ BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'magnitude_distribution_runtime') THEN
    CREATE ROLE magnitude_distribution_runtime NOLOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOINHERIT NOBYPASSRLS CONNECTION LIMIT 20;
  END IF;
END $$;
