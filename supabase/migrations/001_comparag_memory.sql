create table if not exists public.comparag_threads (
  id text primary key,
  metadata jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create table if not exists public.comparag_messages (
  id uuid primary key default gen_random_uuid(),
  thread_id text not null references public.comparag_threads(id),
  role text not null check (role in ('user', 'assistant', 'system', 'tool')),
  content text not null,
  metadata jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now()
);

create table if not exists public.comparag_memory_summaries (
  thread_id text primary key references public.comparag_threads(id),
  summary text not null default '',
  metadata jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create index if not exists comparag_messages_thread_created_idx
  on public.comparag_messages(thread_id, created_at desc);

alter table public.comparag_threads enable row level security;
alter table public.comparag_messages enable row level security;
alter table public.comparag_memory_summaries enable row level security;

do $$
begin
  if not exists (
    select 1 from pg_policies
    where schemaname = 'public'
      and tablename = 'comparag_threads'
      and policyname = 'comparag_threads_service_role_all'
  ) then
    create policy comparag_threads_service_role_all
      on public.comparag_threads
      for all
      using (auth.role() = 'service_role')
      with check (auth.role() = 'service_role');
  end if;

  if not exists (
    select 1 from pg_policies
    where schemaname = 'public'
      and tablename = 'comparag_messages'
      and policyname = 'comparag_messages_service_role_all'
  ) then
    create policy comparag_messages_service_role_all
      on public.comparag_messages
      for all
      using (auth.role() = 'service_role')
      with check (auth.role() = 'service_role');
  end if;

  if not exists (
    select 1 from pg_policies
    where schemaname = 'public'
      and tablename = 'comparag_memory_summaries'
      and policyname = 'comparag_memory_summaries_service_role_all'
  ) then
    create policy comparag_memory_summaries_service_role_all
      on public.comparag_memory_summaries
      for all
      using (auth.role() = 'service_role')
      with check (auth.role() = 'service_role');
  end if;
end $$;
