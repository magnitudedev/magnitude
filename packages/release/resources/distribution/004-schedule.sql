-- Run as the distribution schema owner. Jobs execute with this operator's database privileges.
CREATE EXTENSION IF NOT EXISTS pg_cron WITH SCHEMA pg_catalog;
SELECT cron.schedule('magnitude-distribution-maintenance', '*/5 * * * *', 'SELECT magnitude_distribution.maintain()');
