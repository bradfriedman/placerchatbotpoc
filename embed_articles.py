import os
import pprint

from dotenv import load_dotenv
load_dotenv()

from elevio.client import ElevioClient
from elevio.models import ElevioArticle

from typing import List, Tuple
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone
from pinecone.grpc import PineconeGRPC, GRPCIndex

from utils import clean_html


PINECONE_BATCH_SIZE = 25


def process_articles(client: ElevioClient, embeddings: OpenAIEmbeddings, index: GRPCIndex, limit: int = 0):
    page = 1
    articles_processed = 0
    article_bodies = []
    article_metadatas = []

    namespace = 'elevio_articles'

    while True:
        # Break now if we've processed the limit of articles
        if 0 < limit <= articles_processed:
            break

        list_articles_result: dict = client.list_articles(page=page, page_size=100)
        #article_list: List[ElevioArticle] = client.list_articles(page=page, page_size=100, status="published")
        article_list: List[ElevioArticle] = list_articles_result.get("articles", [])
        print("Number of articles retrieved: ", len(article_list))

        articles = [client.get_article(article.id) for article in article_list]
        if not articles:
            break

        unpublished_articles = list(filter(lambda a: a.status != "published", articles))
        unpublished_count = len(unpublished_articles)
        if unpublished_count > 0:
            print(f"Skipping {unpublished_count} unpublished articles")
            articles = list(filter(lambda a: a.status == "published", articles))

        for article in articles:
            print(f"Processing article: {article.title} ({article.id})")
            for translation in article.translations:
                print(f"Processing translation: {translation.language_id}")
                content = translation.body
                if not content:
                    print(f"Skipping translation with empty content: {translation.language_id}")
                    articles_processed -= 1
                    continue
                else:
                    content = (f"Title: {translation.title if translation.title else article.title}\n\n"
                               f"{clean_html(content)}")
                    if article.keywords:
                        content += f"\n\nKeywords: {article.keywords}"
                    if translation.keywords:
                        content += f"\n\nLanguage-specific Keywords: {translation.keywords}"

                article_bodies.append(content)

                metadata = {
                    "article_id": article.id,
                    "title": article.title,
                    "translation_title": translation.title,
                    "category_id": article.category_id,
                    "text": content,
                    "notes": article.notes,
                    "summary": translation.summary,
                    "language_id": translation.language_id,
                    "keywords": article.keywords,
                    "tags": article.tags,
                    "discoverable": article.discoverable,
                    "restriction": article.restriction,
                    "is_internal": article.is_internal,
                    "status": article.status,
                    "created_at": article.created_at,
                    "updated_at": article.updated_at,
                    "last_published_at": article.last_published_at,
                }
                article_metadatas.append(metadata)

            articles_processed += 1
            if 0 < limit <= articles_processed:
                break

        # If the result's page_number is greater than or equal to total_pages, break now
        page_number = list_articles_result.get("page_number")
        total_pages = list_articles_result.get("total_pages")
        if page_number >= total_pages:
            break

        page += 1

    # Embed the article bodies
    article_embeddings: List[List[float]] = embeddings.embed_documents(article_bodies)

    # Zip the article embeddings and metadata together
    zipped_embeddings_with_metadata: List[Tuple[List[float], dict]] = list(zip(article_embeddings, article_metadatas))
    vector_dict = [
        {
            "id": str(m['article_id']),
            "values": e,
            "metadata": {k: v for (k, v) in m.items() if v is not None},
        } for e, m in zipped_embeddings_with_metadata
    ]

    # Upsert the article embeddings to the Pinecone index, no more than PINECONE_BATCH_SIZE at a time
    print("Upserting article embeddings to Pinecone...")
    total_upserted = 0
    for i in range(0, len(vector_dict), PINECONE_BATCH_SIZE):
        batch = vector_dict[i:i + PINECONE_BATCH_SIZE]
        index.upsert(vectors=batch, namespace=namespace)
        total_upserted += len(batch)
        print(f"...Upserted {len(batch)} article embeddings to Pinecone (total: {total_upserted})")


def main():
    pinecone_api_key = os.environ.get("PINECONE_API_KEY")
    index_name = "articles"

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    elevio_client = ElevioClient()

    pc = PineconeGRPC(api_key=pinecone_api_key)
    index: GRPCIndex = pc.Index(index_name)

    process_articles(elevio_client, embeddings, index)


if __name__ == "__main__":
    main()