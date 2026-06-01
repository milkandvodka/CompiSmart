from __future__ import annotations

from ..models import JsonDict, RagChunk, VideoProfile
from .utils import chunk_id, stable_hash


def build_analysis_chunks(video: JsonDict, profile: VideoProfile) -> list[RagChunk]:
    analysis = video.get("analysis") or {}
    chunks: list[RagChunk] = []
    chunks.extend(build_comment_intelligence_chunks(analysis.get("comment_intelligence") or {}, profile))
    chunks.extend(build_creative_feature_chunks(analysis.get("creative_features") or {}, profile))
    return chunks


def build_comment_intelligence_chunks(comment_intelligence: JsonDict, profile: VideoProfile) -> list[RagChunk]:
    if not comment_intelligence.get("available"):
        return []
    chunks: list[RagChunk] = []
    summary_lines = [
        f"Video {profile.video_id} comment intelligence summary.",
        f"Fetched comments: {comment_intelligence.get('total_fetched_comments', 0)}",
        f"Useful comments: {comment_intelligence.get('useful_comment_count', 0)}",
        f"Comment clusters: {comment_intelligence.get('cluster_count', 0)}",
        f"Compression ratio: {comment_intelligence.get('compression_ratio', 0)}",
        f"Noise summary: {comment_intelligence.get('noise_summary') or {}}",
    ]
    for takeaway in comment_intelligence.get("audience_takeaways") or []:
        summary_lines.append(f"Audience takeaway: {takeaway}")
    for action in comment_intelligence.get("recommended_creator_actions") or []:
        summary_lines.append(f"Recommended action: {action}")
    chunks.append(
        RagChunk(
            id=chunk_id(profile, "comment_intelligence", "summary"),
            comparison_id=profile.comparison_id,
            video_id=profile.video_id,
            doc_type="comment_intelligence_summary",
            text="\n".join(summary_lines),
            display_text="\n".join(summary_lines),
            metadata={
                "platform": profile.platform,
                "source_url": profile.url,
                "citation_label": f"Video {profile.video_id}, comment intelligence summary",
            },
        )
    )

    for index, theme in enumerate(comment_intelligence.get("themes") or []):
        text = "\n".join(
            [
                f"Video {profile.video_id} comment theme: {theme.get('label')}",
                f"Description: {theme.get('description')}",
                f"Comment count: {theme.get('comment_count') or theme.get('comment_count_estimate') or theme.get('cluster_count')}",
                f"Total likes: {theme.get('total_likes')}",
                f"Evidence: {theme.get('evidence') or theme.get('examples')}",
            ]
        )
        chunks.append(
            RagChunk(
                id=chunk_id(profile, "comment_theme", str(index), stable_hash(text)),
                comparison_id=profile.comparison_id,
                video_id=profile.video_id,
                doc_type="comment_theme",
                text=text,
                display_text=text,
                metadata={
                    "platform": profile.platform,
                    "theme_index": index,
                    "citation_label": f"Video {profile.video_id}, comment theme {index + 1}",
                },
            )
        )

    for index, cluster in enumerate((comment_intelligence.get("top_clusters") or [])[:12]):
        text = "\n".join(
            [
                f"Video {profile.video_id} comment cluster {index + 1}.",
                f"Representative: {cluster.get('representative_text')}",
                f"Labels: {', '.join(cluster.get('labels') or [])}",
                f"Count: {cluster.get('count')}",
                f"Total likes: {cluster.get('total_likes')}",
                f"Examples: {cluster.get('examples')}",
            ]
        )
        chunks.append(
            RagChunk(
                id=chunk_id(profile, "comment_cluster", str(index), stable_hash(text)),
                comparison_id=profile.comparison_id,
                video_id=profile.video_id,
                doc_type="comment_cluster",
                text=text,
                display_text=text,
                metadata={
                    "platform": profile.platform,
                    "cluster_index": index,
                    "citation_label": f"Video {profile.video_id}, comment cluster {index + 1}",
                },
            )
        )

    noise_summary = comment_intelligence.get("noise_summary") or {}
    if noise_summary:
        text = f"Video {profile.video_id} comment noise summary.\nNoise buckets: {noise_summary}"
        chunks.append(
            RagChunk(
                id=chunk_id(profile, "comment_noise", stable_hash(text)),
                comparison_id=profile.comparison_id,
                video_id=profile.video_id,
                doc_type="comment_noise_summary",
                text=text,
                display_text=text,
                metadata={
                    "platform": profile.platform,
                    "citation_label": f"Video {profile.video_id}, comment noise summary",
                },
            )
        )
    return chunks


def build_creative_feature_chunks(creative_features: JsonDict, profile: VideoProfile) -> list[RagChunk]:
    if not creative_features.get("available"):
        return []
    lines = [
        f"Video {profile.video_id} transcript-only creative features.",
        f"Hook type: {creative_features.get('hook_type')}",
        f"First 5s promise: {creative_features.get('first_5s_promise')}",
        f"Target audience: {creative_features.get('target_audience')}",
        f"Pain points: {creative_features.get('pain_points')}",
        f"Proof elements: {creative_features.get('proof_elements')}",
        f"CTA: {creative_features.get('cta')}",
        f"Emotional angle: {creative_features.get('emotional_angle')}",
        f"Claims: {creative_features.get('claims')}",
        f"Risk flags: {creative_features.get('risk_flags')}",
        f"Improvement opportunities: {creative_features.get('improvement_opportunities')}",
    ]
    text = "\n".join(lines)
    return [
        RagChunk(
            id=chunk_id(profile, "creative_features", stable_hash(text)),
            comparison_id=profile.comparison_id,
            video_id=profile.video_id,
            doc_type="creative_features",
            text=text,
            display_text=text,
            metadata={
                "platform": profile.platform,
                "citation_label": f"Video {profile.video_id}, creative features",
            },
        )
    ]
