"""
Thin client for Google Scholar search/author lookup, adapted from
https://github.com/JackKuo666/Google-Scholar-MCP-Server (vendored under
google-scholar/ for reference).

search_google_scholar and search_google_scholar_advanced scrape scholar.google.com
directly; get_author_info uses the `scholarly` package.
"""

from __future__ import annotations

import requests
from bs4 import BeautifulSoup
from scholarly import scholarly

SEARCH_URL = "https://scholar.google.com/scholar"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}


def _parse_results(html: str, num_results: int) -> list[dict]:
    soup = BeautifulSoup(html, "html.parser")
    results = []
    for item in soup.find_all("div", class_="gs_ri")[:num_results]:
        title_tag = item.find("h3", class_="gs_rt")
        title = title_tag.get_text() if title_tag else "No title available"
        link = title_tag.find("a")["href"] if title_tag and title_tag.find("a") else "No link available"
        authors_tag = item.find("div", class_="gs_a")
        authors = authors_tag.get_text() if authors_tag else "No authors available"
        abstract_tag = item.find("div", class_="gs_rs")
        abstract = abstract_tag.get_text() if abstract_tag else "No abstract available"
        results.append({"title": title, "authors": authors, "abstract": abstract, "url": link})
    return results


def search_google_scholar(query: str, num_results: int = 5) -> list[dict]:
    """Search Google Scholar with a plain keyword query.

    Args:
        query: Search query string (e.g. paper title or author).
        num_results: Number of results to return.

    Raises:
        requests.HTTPError: Google Scholar returned a non-2xx status.
    """
    response = requests.get(SEARCH_URL, params={"q": query}, headers=HEADERS, timeout=30)
    response.raise_for_status()
    return _parse_results(response.text, num_results)


def search_google_scholar_advanced(
    query: str,
    author: str | None = None,
    year_range: tuple[int, int] | None = None,
    num_results: int = 5,
) -> list[dict]:
    """Search Google Scholar filtered by author and/or publication year range.

    Args:
        query: General search query.
        author: Author name to filter by.
        year_range: (start_year, end_year) to filter by publication year.
        num_results: Number of results to return.

    Raises:
        requests.HTTPError: Google Scholar returned a non-2xx status.
    """
    params = {"q": query}
    if author:
        params["as_auth"] = author
    if year_range:
        params["as_ylo"], params["as_yhi"] = year_range
    response = requests.get(SEARCH_URL, params=params, headers=HEADERS, timeout=30)
    response.raise_for_status()
    return _parse_results(response.text, num_results)


def get_author_info(author_name: str) -> dict:
    """Look up an author's profile and top publications on Google Scholar.

    Args:
        author_name: Name of the author to search for.

    Raises:
        StopIteration: No author matching `author_name` was found.
    """
    search_query = scholarly.search_author(author_name)
    author = next(search_query)
    filled = scholarly.fill(author)
    return {
        "name": filled.get("name", "N/A"),
        "affiliation": filled.get("affiliation", "N/A"),
        "interests": filled.get("interests", []),
        "citedby": filled.get("citedby", 0),
        "publications": [
            {
                "title": pub.get("bib", {}).get("title", "N/A"),
                "year": pub.get("bib", {}).get("pub_year", "N/A"),
                "citations": pub.get("num_citations", 0),
            }
            for pub in filled.get("publications", [])[:5]
        ],
    }
