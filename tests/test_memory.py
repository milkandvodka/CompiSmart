import unittest
from unittest.mock import Mock, patch

from pathlib import Path

from comparag.memory import (
    COMPARAG_SUPABASE_TABLES,
    FallbackConversationMemory,
    InMemoryConversationMemory,
    SupabaseConversationMemory,
    ensure_comparag_table,
)


class SupabaseMemoryTests(unittest.TestCase):
    def test_get_reads_recent_messages_in_chronological_order(self):
        memory = SupabaseConversationMemory(url="https://project.supabase.co", service_role_key="key")
        response = Mock(status_code=200)
        response.json.return_value = [
            {"role": "assistant", "content": "second"},
            {"role": "user", "content": "first"},
        ]
        with patch.object(memory, "session_get", return_value=response) as session_get:
            messages = memory.get("thread")

        self.assertEqual(messages, [{"role": "user", "content": "first"}, {"role": "assistant", "content": "second"}])
        self.assertEqual(session_get.call_args.kwargs["params"]["order"], "created_at.desc")

    def test_append_turn_upserts_thread_and_inserts_two_messages(self):
        memory = SupabaseConversationMemory(url="https://project.supabase.co", service_role_key="key")
        response = Mock()
        response.raise_for_status.return_value = None
        with patch.object(memory, "upsert_thread") as upsert:
            with patch.object(memory, "session_post", return_value=response) as session_post:
                memory.append_turn("thread", user="hi", assistant="hello")

        upsert.assert_called_once_with("thread")
        payload = session_post.call_args.args[1]
        self.assertEqual([row["role"] for row in payload], ["user", "assistant"])

    def test_in_memory_summary_round_trips_and_counts_messages(self):
        memory = InMemoryConversationMemory()

        memory.append_turn("thread", user="hi", assistant="hello")
        memory.save_summary("thread", "User cares about Video B.", {"message_count": 2})

        self.assertEqual(memory.message_count("thread"), 2)
        self.assertEqual(memory.get_summary("thread")["summary"], "User cares about Video B.")
        self.assertEqual(memory.get_summary("thread")["metadata"]["message_count"], 2)

    def test_supabase_summary_uses_comparag_summary_table(self):
        memory = SupabaseConversationMemory(url="https://project.supabase.co", service_role_key="key")
        get_response = Mock(status_code=200)
        get_response.json.return_value = [{"summary": "remember this", "metadata": {"message_count": 4}}]
        post_response = Mock()
        post_response.raise_for_status.return_value = None

        with patch.object(memory, "session_get", return_value=get_response) as session_get:
            summary = memory.get_summary("thread")
        with patch.object(memory, "upsert_thread") as upsert:
            with patch.object(memory, "session_post", return_value=post_response) as session_post:
                memory.save_summary("thread", "new summary", {"message_count": 6})

        self.assertEqual(summary["summary"], "remember this")
        self.assertEqual(session_get.call_args.args[0], "comparag_memory_summaries")
        upsert.assert_called_once_with("thread")
        self.assertEqual(session_post.call_args.args[0], "comparag_memory_summaries")
        self.assertEqual(session_post.call_args.args[1]["thread_id"], "thread")

    def test_supabase_message_count_uses_exact_count_header(self):
        memory = SupabaseConversationMemory(url="https://project.supabase.co", service_role_key="key")
        response = Mock(status_code=200)
        response.headers = {"Content-Range": "0-0/42"}
        response.raise_for_status.return_value = None

        with patch.object(memory, "session_get", return_value=response) as session_get:
            count = memory.message_count("thread")

        self.assertEqual(count, 42)
        self.assertEqual(session_get.call_args.args[0], "comparag_messages")
        self.assertEqual(session_get.call_args.kwargs["headers"]["Prefer"], "count=exact")

    def test_fallback_memory_uses_local_when_primary_fails(self):
        primary = Mock()
        primary.get.side_effect = RuntimeError("missing table")
        primary.append_turn.side_effect = RuntimeError("missing table")
        memory = FallbackConversationMemory(primary, InMemoryConversationMemory())

        self.assertEqual(memory.get("thread"), [])
        memory.append_turn("thread", user="hi", assistant="hello")
        self.assertIn("memory write failed", memory.last_error)

        self.assertEqual(len(memory.get("thread")), 2)

    def test_supabase_table_allowlist_rejects_non_comparag_tables(self):
        with self.assertRaises(ValueError):
            ensure_comparag_table("profiles")

        for table in COMPARAG_SUPABASE_TABLES:
            ensure_comparag_table(table)

    def test_memory_migration_is_non_destructive_and_scoped(self):
        sql = Path("supabase/migrations/001_comparag_memory.sql").read_text(encoding="utf-8").lower()

        for forbidden in ["drop table", "drop schema", "truncate ", "delete from", "alter table public.profiles"]:
            self.assertNotIn(forbidden, sql)
        self.assertIn("create table if not exists public.comparag_threads", sql)
        self.assertIn("create table if not exists public.comparag_messages", sql)


if __name__ == "__main__":
    unittest.main()
