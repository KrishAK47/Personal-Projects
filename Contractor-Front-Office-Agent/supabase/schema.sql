-- Contractor Front Office Agent: Postgres schema for Supabase.
-- Run in the Supabase dashboard: SQL Editor -> New query -> paste -> Run. Safe to re-run.
--
-- Every row belongs to a business_id, so several contractors can share one database.
-- Row level security is ON with no policies: only the server-side secret key can read or
-- write. Per-user access policies get added together with contractor login.

-- ---------- Tables ----------

create table if not exists customers (
  business_id   uuid not null,
  customer_id   text not null,
  name          text not null,
  phone         text not null default '',
  phone_digits  text generated always as (right(regexp_replace(phone, '\D', '', 'g'), 10)) stored,
  email         text not null default '',
  address       text not null default '',
  total_jobs    int  not null default 0,
  last_job_date date,
  notes         text not null default '',
  created_at    timestamptz not null default now(),
  primary key (business_id, customer_id)
);
create index if not exists customers_phone_idx on customers (business_id, phone_digits);

create table if not exists slots (
  business_id    uuid not null,
  slot_id        text not null,
  date           date not null,
  time           text not null check (time ~ '^\d{2}:\d{2}$'),
  duration_hours int  not null default 2,
  status         text not null default 'available' check (status in ('available', 'booked')),
  job_id         text,
  primary key (business_id, slot_id)
);
create index if not exists slots_open_idx on slots (business_id, status, date, time);

create table if not exists jobs (
  business_id   uuid not null,
  job_id        text not null,
  customer_id   text,
  customer_name text not null,
  address       text not null default '',
  phone         text not null default '',
  email         text not null default '',
  job_type      text not null,
  urgency       text not null check (urgency in ('emergency', 'this week', 'flexible')),
  description   text not null default '',
  slot_id       text not null,
  status        text not null default 'pending_confirmation'
                check (status in ('pending_confirmation', 'confirmed', 'completed', 'cancelled')),
  created_date  date not null default current_date,
  notes         text not null default '',
  created_at    timestamptz not null default now(),
  primary key (business_id, job_id),
  foreign key (business_id, slot_id) references slots (business_id, slot_id) on delete cascade
);
-- The real double-booking guard: at most one active job per slot.
create unique index if not exists jobs_one_active_per_slot
  on jobs (business_id, slot_id) where status <> 'cancelled';

alter table customers enable row level security;
alter table slots     enable row level security;
alter table jobs      enable row level security;

-- ---------- Functions ----------

-- Creates a customer with the next C### id for this business.
create or replace function create_customer(
  p_business uuid, p_name text, p_phone text, p_email text, p_address text
) returns customers
language plpgsql set search_path = public as $$
declare
  c customers;
  n int;
begin
  perform pg_advisory_xact_lock(hashtext('customers:' || p_business::text));
  select coalesce(max(substring(customer_id from 2)::int), 0) + 1 into n
    from customers where business_id = p_business;
  insert into customers (business_id, customer_id, name, phone, email, address)
  values (p_business, 'C' || lpad(n::text, 3, '0'), p_name, coalesce(p_phone, ''),
          coalesce(p_email, ''), coalesce(p_address, ''))
  returning * into c;
  return c;
end $$;

-- Books a slot in one transaction: claim the slot (fails if taken), create the job,
-- link it to the slot, and bump the customer's job count. Any error rolls back everything.
create or replace function book_job_atomic(
  p_business uuid, p_slot_id text, p_customer_id text, p_job jsonb, p_today date
) returns jobs
language plpgsql set search_path = public as $$
declare
  c customers;
  j jobs;
  n int;
begin
  update slots set status = 'booked'
   where business_id = p_business and slot_id = p_slot_id and status = 'available';
  if not found then
    raise exception 'SLOT_UNAVAILABLE';
  end if;

  select * into c from customers where business_id = p_business and customer_id = p_customer_id;
  if not found then
    raise exception 'UNKNOWN_CUSTOMER';
  end if;

  perform pg_advisory_xact_lock(hashtext('jobs:' || p_business::text));
  select coalesce(max(substring(job_id from 2)::int), 1000) + 1 into n
    from jobs where business_id = p_business;

  insert into jobs (business_id, job_id, customer_id, customer_name, address, phone, email,
                    job_type, urgency, description, slot_id, status, created_date, notes)
  values (p_business, 'J' || n, c.customer_id, c.name, c.address, c.phone, c.email,
          p_job ->> 'job_type', p_job ->> 'urgency', coalesce(p_job ->> 'description', ''),
          p_slot_id, 'pending_confirmation', p_today, coalesce(p_job ->> 'notes', ''))
  returning * into j;

  update slots set job_id = j.job_id where business_id = p_business and slot_id = p_slot_id;
  update customers set total_jobs = total_jobs + 1, last_job_date = p_today
   where business_id = p_business and customer_id = p_customer_id;
  return j;
end $$;

-- Replaces all of one business's data (used to load and refresh the demo data).
create or replace function reset_business_data(
  p_business uuid, p_customers jsonb, p_slots jsonb, p_jobs jsonb
) returns void
language plpgsql set search_path = public as $$
begin
  delete from jobs      where business_id = p_business;
  delete from slots     where business_id = p_business;
  delete from customers where business_id = p_business;

  insert into customers (business_id, customer_id, name, phone, email, address,
                         total_jobs, last_job_date, notes)
  select p_business, x.customer_id, x.name, x.phone, x.email, x.address,
         coalesce(nullif(x.total_jobs, '')::int, 0), nullif(x.last_job_date, '')::date, x.notes
    from jsonb_to_recordset(p_customers) as x(customer_id text, name text, phone text,
         email text, address text, total_jobs text, last_job_date text, notes text);

  insert into slots (business_id, slot_id, date, time, duration_hours, status, job_id)
  select p_business, x.slot_id, x.date::date, x.time, coalesce(nullif(x.duration_hours, '')::int, 2),
         x.status, nullif(x.job_id, '')
    from jsonb_to_recordset(p_slots) as x(slot_id text, date text, time text,
         duration_hours text, status text, job_id text);

  insert into jobs (business_id, job_id, customer_id, customer_name, address, phone, email,
                    job_type, urgency, description, slot_id, status, created_date, notes)
  select p_business, x.job_id, nullif(x.customer_id, ''), x.customer_name, x.address, x.phone,
         x.email, x.job_type, x.urgency, x.description, x.slot_id, x.status,
         x.created_date::date, x.notes
    from jsonb_to_recordset(p_jobs) as x(job_id text, customer_id text, customer_name text,
         address text, phone text, email text, job_type text, urgency text, description text,
         slot_id text, status text, created_date text, notes text);
end $$;

-- Only the server (secret key) may call these.
revoke execute on function create_customer(uuid, text, text, text, text) from public, anon, authenticated;
revoke execute on function book_job_atomic(uuid, text, text, jsonb, date) from public, anon, authenticated;
revoke execute on function reset_business_data(uuid, jsonb, jsonb, jsonb) from public, anon, authenticated;
grant  execute on function create_customer(uuid, text, text, text, text) to service_role;
grant  execute on function book_job_atomic(uuid, text, text, jsonb, date) to service_role;
grant  execute on function reset_business_data(uuid, jsonb, jsonb, jsonb) to service_role;
