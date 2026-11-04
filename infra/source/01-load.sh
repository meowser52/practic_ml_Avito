#!/usr/bin/env bash
set -euo pipefail
# Runs only when PostgreSQL initializes a new empty volume.
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<'SQL'
CREATE TABLE public.orders (
    order_id TEXT PRIMARY KEY,
    order_purchase_timestamp TIMESTAMP NOT NULL,
    price DOUBLE PRECISION NOT NULL,
    freight DOUBLE PRECISION NOT NULL,
    weight DOUBLE PRECISION NOT NULL,
    n_items INTEGER NOT NULL,
    installments DOUBLE PRECISION NOT NULL,
    est_days DOUBLE PRECISION NOT NULL,
    customer_state TEXT NOT NULL,
    seller_state TEXT NOT NULL,
    category TEXT NOT NULL,
    payment_type TEXT NOT NULL,
    late SMALLINT NOT NULL CHECK (late IN (0, 1))
);
SQL
gzip -dc /seed/olist_orders.csv.gz | psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" -c '\copy public.orders FROM STDIN WITH (FORMAT csv, HEADER true)'
