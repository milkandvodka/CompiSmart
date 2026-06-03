import { formatNumber, thumbnailFallback, thumbnailImage, thumbnailProxyUrl } from "../format.js";
import { Metric } from "./Metric.jsx";

export function VideoCard({ video }) {
  const fallback = thumbnailFallback(video);
  const fallbackUrl = thumbnailProxyUrl(fallback);
  const imageUrl = thumbnailImage(video);

  function handleImageError(event) {
    if (fallbackUrl && event.currentTarget.src !== fallbackUrl) {
      event.currentTarget.src = fallbackUrl;
      return;
    }
    event.currentTarget.replaceWith(Object.assign(document.createElement("div"), {
      className: "thumbnail-missing",
      textContent: "Thumbnail unavailable",
    }));
  }

  return (
    <article className="video-card">
      {imageUrl ? (
        <img src={imageUrl} alt="" loading="lazy" onError={handleImageError} />
      ) : (
        <div className="thumbnail-missing">No thumbnail</div>
      )}
      <div className="video-body">
        <div className="video-title">
          <h3>Video {video.video_id}</h3>
          <span>{video.platform}</span>
        </div>
        <p className="title">{video.title || "Untitled"}</p>
        <dl>
          <Metric label="Creator" value={video.creator || "unavailable"} />
          <Metric label="Followers" value={formatNumber(video.follower_count)} />
          <Metric label="Views" value={formatNumber(video.views)} />
          <Metric label="Likes" value={formatNumber(video.likes)} />
          <Metric label="Comments" value={formatNumber(video.comments)} />
          <Metric label="Fetched comments" value={formatNumber(video.fetched_comment_count)} />
          <Metric
            label="Engagement"
            value={video.engagement_rate == null ? "unavailable" : `${Number(video.engagement_rate).toFixed(2)}%`}
          />
          <Metric
            label="Duration"
            value={video.duration_seconds == null ? "unavailable" : `${Number(video.duration_seconds).toFixed(1)}s`}
          />
          <Metric label="Upload" value={video.upload_date || "unavailable"} />
        </dl>
      </div>
    </article>
  );
}
