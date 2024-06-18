import os
from typing import Optional, List

from elevio.models import ElevioArticle, ElevioTranslation, ElevioSearchResultSummary
from elevio.utils import iso_to_unix_ms, VALID_LANGUAGE_CODES, validate_language_code

import requests
import streamlit as st

BASE_ELEVIO_URL = st.secrets["BASE_ELEVIO_URL"]
ELEVIO_API_KEY = st.secrets["ELEVIO_API_KEY"]
ELEVIO_JWT = st.secrets["ELEVIO_JWT"]


class ElevioClient:
    def __init__(self):
        self.base_url = BASE_ELEVIO_URL
        self.session = requests.Session()
        self.auth_headers = {
            'x-api-key': ELEVIO_API_KEY,
            'Authorization': f'Bearer {ELEVIO_JWT}'
        }
        self.session.headers.update(self.auth_headers)

    def list_articles(self, page: Optional[int] = None, page_size: Optional[int] = None, status: Optional[str] = None,
                      from_created_at: Optional[int] = None, to_created_at: Optional[int] = None,
                      from_published_at: Optional[int] = None, to_published_at: Optional[int] = None,
                      tags: Optional[List[str]] = None) -> dict:
        """
        List articles from Elevio API

        :param page: Page number
        :param page_size: Number of articles per page
        :param status: Article status. Must be one of ['draft', 'published'] or None.
        :param from_created_at: From created date in Unix timestamp
        :param to_created_at: To created date in Unix timestamp
        :param from_published_at: From published date in Unix timestamp
        :param to_published_at: To published date in Unix timestamp
        :param tags: List of tags to filter articles
        :return: Dictionary with list of ElevioArticle objects and metadata
        """
        url = f"{self.base_url}/articles"

        # Check that status is either None or one of ['draft', 'published'], otherwise raise ValueError
        if status not in [None, "draft", "published"]:
            raise ValueError(f"Invalid status: {status}")

        # Check that page is valid
        if page is not None and page < 1:
            raise ValueError(f"Invalid page: {page}")

        # Check that page_size is valid
        if page_size is not None and page_size < 1:
            raise ValueError(f"Invalid page_size: {page_size}")

        # Check that page_size is not above the maximum allowed by the API (500)
        if page_size is not None and page_size > 500:
            raise ValueError(f"page_size cannot be greater than 500")

        params = {
            "page": page,
            "page_size": page_size,
            "status": status,
            "from_created_at": from_created_at,
            "to_created_at": to_created_at,
            "from_published_at": from_published_at,
            "to_published_at": to_published_at,
            "tag[]": tags
        }

        response = self.session.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
        else:
            raise Exception(f"Failed to list articles: {response.text}")

        results: List[ElevioArticle] = []
        articles: List[dict] = data['articles']

        for a in articles:
            article = ElevioArticle.from_dict(a)
            results.append(article)

        # Add metadata about the list
        page_number = data.get('page_number')
        page_size = data.get('page_size')
        total_pages = data.get('total_pages')
        total_entries = data.get('total_entries')

        return {
            "articles": results,
            "page_number": page_number,
            "page_size": page_size,
            "total_pages": total_pages,
            "total_entries": total_entries,
        }

    def get_article(self, article_id: int) -> ElevioArticle:
        """
        Get an article by ID from Elevio API

        :param article_id: Article ID
        :return: ElevioArticle object
        """
        url = f"{self.base_url}/articles/{article_id}"
        response = self.session.get(url)
        if response.status_code == 200:
            data = response.json()
        else:
            raise Exception(f"Failed to get article: {response.text}")

        article = ElevioArticle.from_dict(data.get('article'))

        return article

    def search(self, query: str, lang_code: str = "en") -> ElevioSearchResultSummary:
        """
        Search articles by query from Elevio API

        :param query: Search query
        :param lang_code: Language code
        :return: ElevioSearchResultSummary object
        """
        # Validate the language code
        if not validate_language_code(lang_code):
            raise ValueError(f"Invalid language code: {lang_code} (valid codes are {', '.join(VALID_LANGUAGE_CODES)})")

        url = f"{self.base_url}/search/{lang_code}"
        params = {
            "query": query,
        }

        response = self.session.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
        else:
            raise Exception(f"Failed to search articles: {response.text}")

        return ElevioSearchResultSummary.from_dict(data)


