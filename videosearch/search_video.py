import os
import json
import requests
from youtube_search import YoutubeSearch as YTScraper


def youtube_search(query: str) -> list:
    """
    Uses youtube-search library to find videos with rich metadata.
    Returns a list of dicts with title, link, snippet, duration, etc.
    """
    try:
        results = YTScraper(query, max_results=10).to_dict()

        candidates = []
        for v in results:
            suffix = v.get('url_suffix', '')
            link = f"https://www.youtube.com{suffix}" if suffix else v.get('link', '')

            thumbnails = v.get("thumbnails", [])
            image_url = thumbnails[0] if isinstance(thumbnails, list) and thumbnails else ""

            candidates.append({
                "title": v.get("title"),
                "link": link,
                "snippet": v.get("long_desc") or v.get("title"),
                "duration": v.get("duration", "unknown"),
                "imageurl": image_url,
                "videourl": link,
                "source": "YouTube",
                "channel": v.get("channel", "unknown"),
                "date": v.get("publish_time", "unknown"),
                "position": "unknown"
            })
        return candidates

    except Exception as e:
        print(f"[YouTube] Request failed: {e}")
        return []


def serper_search(query: str) -> list:
    """
    Uses Serper.dev API to find videos.
    Returns a list of dicts with title, link, snippet, duration, etc.
    """
    url = "https://google.serper.dev/videos"

    api_key = os.getenv("SERPER_API_KEY")
    if not api_key:
        print("[Serper] Missing SERPER_API_KEY. Returning empty list.")
        return []

    payload = json.dumps({"q": query, "num": 10})
    headers = {
        'X-API-KEY': api_key,
        'Content-Type': 'application/json'
    }

    candidates = []
    try:
        response = requests.post(url, headers=headers, data=payload)
        response.raise_for_status()
        data = response.json()

        if 'videos' in data:
            for v in data['videos']:
                candidates.append({
                    "title": v.get("title"),
                    "link": v.get("link"),
                    "snippet": v.get("snippet", "") or v.get("title"),
                    "duration": v.get("duration", "unknown"),
                    "imageurl": v.get("imageUrl", ""),
                    "videourl": v.get("videoUrl", ""),
                    "source": v.get("source", "unknown"),
                    "channel": v.get("channel", "unknown"),
                    "date": v.get("date", "unknown"),
                    "position": v.get("position", "unknown"),
                })
        else:
            print(f"[Serper] No 'videos' key in response for query: {query}")

    except Exception as e:
        print(f"[Serper] Request failed: {e}")

    return candidates


if __name__ == "__main__":
    query = "how to cook pasta"
    print(f"Testing YouTube search: {query}")
    results = youtube_search(query)
    print(f"\nFound {len(results)} results:")
    for res in results:
        print(f"- {res['title']} ({res['link']})")
