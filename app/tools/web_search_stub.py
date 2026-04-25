"""
Web search stub — returns a placeholder result.

TODO (Part A — student extension point):
  Replace this stub with a real free web search. Options:
  - Wikipedia API (recommended — reliable, free, citeable):
      import requests
      resp = requests.get("https://en.wikipedia.org/w/api.php", params={
          "action": "query", "list": "search", "srsearch": query,
          "format": "json", "srlimit": 3,
      })
  - DuckDuckGo Instant Answer API (no key required):
      resp = requests.get("https://api.duckduckgo.com/", params={
          "q": query, "format": "json", "no_redirect": 1,
      })

  Keep the return type: list[{"id": str, "text": str, "score": float}]
"""


def web_search(query: str, top_k: int = 3) -> list[dict]:
    """
    Stub: always returns an empty list.
    Replace with a real implementation for Part A.
    """
    # TODO: implement real web search here
    return []
