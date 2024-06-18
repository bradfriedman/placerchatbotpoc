from typing import Optional, List

from elevio.utils import unix_ms_to_datetime, datetime_to_unix_ms, iso_to_unix_ms

class ElevioTranslation:
    def __init__(self, language_id: str = "en", title: Optional[str] = None, body: Optional[str] = None,
                 summary: Optional[str] = None, keywords: Optional[List[str]] = None,
                 tags: Optional[List[str]] = None):
        self.language_id: str = language_id
        self.title: Optional[str] = title
        self.body: Optional[str] = body
        self.summary: Optional[str] = summary
        self.keywords: Optional[List[str]] = keywords
        self.tags: Optional[List[str]] = tags

    def get_body(self, truncate_length: int = 0) -> str:
        if self.body is None:
            return "[empty]"
        if len(self.body) > truncate_length > 0:
            return f"{self.body[:truncate_length]}...[{len(self.body) - truncate_length} more characters]"
        return self.body

    def __str__(self) -> str:
        return f"Title: {self.title}\nSummary: {self.summary}\nKeywords: {self.keywords}\nTags: {self.tags}\n" \
               f"Language ID: {self.language_id}\nBody: {self.get_body(60)}"

    @classmethod
    def from_dict(cls, data: dict) -> "ElevioTranslation":
        return ElevioTranslation(
            language_id=data.get("language_id"),
            title=data.get("title"),
            body=data.get("body"),
            summary=data.get("summary"),
            keywords=data.get("keywords"),
            tags=data.get("tags")
        )


class ElevioArticle:
    def __init__(self, article_id: int, category_id: int, order: int = 999, title: Optional[str] = None, notes: Optional[str] = None,
                 keywords: Optional[List[str]] = None,
                 discoverable: bool = True, restriction: str = "unrestricted", is_internal: bool = False,
                 status: str = "published", created_at: int = 0, updated_at: int = 0,
                 last_published_at: Optional[int] = None,
                 translations: Optional[List[ElevioTranslation]] = None, tags: Optional[List[str]] = None):
        self.id: int = article_id
        self.order: int = order
        self.title: Optional[str] = title
        self.notes: Optional[str] = notes
        self.keywords: Optional[List[str]] = keywords
        self.category_id: int = category_id
        self.discoverable: bool =  discoverable
        self.restriction: str = restriction
        self.is_internal: bool = is_internal
        self.status: str = status
        self.created_at: int = created_at
        self.updated_at: int = updated_at
        self.last_published_at: Optional[int] = last_published_at
        self.translations: List[ElevioTranslation] = translations if translations else []
        self.tags: Optional[List[str]] = tags

    # Define how this object would be printed when converted to a string
    def __str__(self) -> str:
        return f"Article ID: {self.id}\nTitle: {self.title}\nCategory ID: {self.category_id}\n" \
               f"Order: {self.order}\nNotes: {self.notes}\nKeywords: {self.keywords}\n"\
               f"Discoverable: {self.discoverable}\n" \
               f"Restriction: {self.restriction}\nIs Internal: {self.is_internal}\nStatus: {self.status}\n" \
               f"Created At: {unix_ms_to_datetime(self.created_at)}\n" \
               f"Updated At: {unix_ms_to_datetime(self.updated_at)}\n" \
               f"Last Published At: {unix_ms_to_datetime(self.last_published_at)}\n" \
               f"Tags: {self.tags}\n" \
               f"Translations:\n{'\n\n**\n\n'.join([str(translation) for translation in self.translations])}"

    @classmethod
    def from_dict(cls, data: dict) -> "ElevioArticle":
        return ElevioArticle(
            article_id=data.get("id"),
            category_id=data.get("category_id"),
            order=data.get("order"),
            title=data.get("title"),
            notes=data.get("notes"),
            keywords=data.get("keywords"),
            discoverable=data.get("discoverable"),
            restriction=data.get("restriction"),
            status=data.get("status"),
            created_at=iso_to_unix_ms(data.get("created_at")),
            updated_at=iso_to_unix_ms(data.get("updated_at")),
            last_published_at=iso_to_unix_ms(data.get("last_published_at")),
            tags=data.get("tags"),
            translations=[ElevioTranslation.from_dict(t) for t in data.get("translations")]
        )


class ElevioSearchResult:
    def __init__(self, article_id: int, title: str):
        self.id: int = article_id
        self.title: str = title

    def __str__(self) -> str:
        return f"Article ID: {self.id}\nTitle: {self.title}"

    @classmethod
    def from_dict(cls, data: dict) -> "ElevioSearchResult":
        return ElevioSearchResult(
            article_id=data.get("id"),
            title=data.get("title")
        )


class ElevioSearchResultSummary:
    def __init__(self, total_results: int, total_pages: int, current_page: int, count: int,
                 results: Optional[List[ElevioSearchResult]] = None):
        self.total_results: int = total_results
        self.total_pages: int = total_pages
        self.current_page: int = current_page
        self.count: int = count
        self.results: List[ElevioSearchResult] = results if results else []

    def __str__(self) -> str:
        return f"Total Results: {self.total_results}\nTotal Pages: {self.total_pages}\n" \
               f"Current Page: {self.current_page}\nCount: {self.count}\n" \
               f"Results:\n{'\n\n**\n\n'.join([str(result) for result in self.results])}"

    @classmethod
    def from_dict(cls, data: dict) -> "ElevioSearchResultSummary":
        return ElevioSearchResultSummary(
            total_results=data.get("total_results"),
            total_pages=data.get("total_pages"),
            current_page=data.get("current_page"),
            count=data.get("count"),
            results=[ElevioSearchResult.from_dict(r) for r in data.get("results")],
        )
