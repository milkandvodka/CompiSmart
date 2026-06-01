import unittest

from comparag.indexing import build_chunk_manifest, chunk_fingerprint, plan_index_update
from comparag.models import RagChunk


def make_chunk(chunk_id: str, text: str, *, doc_type: str = "full_transcript") -> RagChunk:
    return RagChunk(
        id=chunk_id,
        comparison_id="demo",
        video_id="A",
        doc_type=doc_type,
        text=text,
        display_text=text,
        metadata={"citation_label": f"Video A, {chunk_id}"},
    )


class ComparagIndexingTests(unittest.TestCase):
    def test_chunk_fingerprint_changes_when_text_changes(self):
        first = make_chunk("a", "hello")
        second = make_chunk("a", "hello again")

        self.assertNotEqual(chunk_fingerprint(first), chunk_fingerprint(second))

    def test_incremental_plan_skips_unchanged_chunks(self):
        chunks = [make_chunk("a", "same"), make_chunk("b", "same")]
        previous_record = {"chunk_manifest": build_chunk_manifest(chunks)}

        plan = plan_index_update(chunks, previous_record=previous_record)

        self.assertEqual(plan.mode, "incremental")
        self.assertEqual(plan.upsert_count, 0)
        self.assertEqual(plan.delete_count, 0)
        self.assertEqual(plan.unchanged_count, 2)

    def test_incremental_plan_upserts_changed_and_deletes_removed(self):
        old_chunks = [make_chunk("a", "old"), make_chunk("b", "remove")]
        new_chunks = [make_chunk("a", "new"), make_chunk("c", "add")]
        previous_record = {"chunk_manifest": build_chunk_manifest(old_chunks)}

        plan = plan_index_update(new_chunks, previous_record=previous_record)

        self.assertEqual([chunk.id for chunk in plan.chunks_to_upsert], ["a", "c"])
        self.assertEqual(plan.chunk_ids_to_delete, ["b"])
        self.assertEqual(plan.unchanged_count, 0)

    def test_no_previous_manifest_forces_full_upsert(self):
        chunks = [make_chunk("a", "hello")]

        plan = plan_index_update(chunks, previous_record={})

        self.assertEqual(plan.mode, "full_no_previous_manifest")
        self.assertEqual(plan.upsert_count, 1)

    def test_no_delete_preserves_stale_chunk_ids(self):
        old_chunks = [make_chunk("a", "old"), make_chunk("b", "stale")]
        new_chunks = [make_chunk("a", "old")]
        previous_record = {"chunk_manifest": build_chunk_manifest(old_chunks)}

        plan = plan_index_update(new_chunks, previous_record=previous_record, no_delete=True)

        self.assertEqual(plan.chunk_ids_to_delete, [])
        self.assertEqual(plan.unchanged_count, 1)


if __name__ == "__main__":
    unittest.main()
