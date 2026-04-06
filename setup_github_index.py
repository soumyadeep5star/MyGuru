"""
Setup script to fetch GitHub README content and create FAISS index.
Run this script to build a GitHub-focused vector database with one chunk per README.
"""

import argparse
import logging
import os

from config import Config
from github_search import GitHubSearcher
from vector_store import VectorStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _validate_required_config():
    missing = []
    required = [
        ("AZURE_OPENAI_API_KEY", Config.AZURE_OPENAI_API_KEY),
        ("AZURE_OPENAI_ENDPOINT", Config.AZURE_OPENAI_ENDPOINT),
        ("GITHUB_TOKEN", Config.GITHUB_TOKEN),
    ]

    for name, value in required:
        if not value:
            missing.append(name)

    if missing:
        raise ValueError(f"Missing required environment variables: {', '.join(missing)}")


def _resolve_repo_list(searcher: GitHubSearcher, cli_repos, max_repos: int):
    if cli_repos:
        return cli_repos

    if Config.GITHUB_INDEX_REPOS:
        return [r.strip() for r in Config.GITHUB_INDEX_REPOS.split(",") if r.strip()]

    accessible = searcher.get_accessible_repos()
    return accessible[:max_repos]


def main():
    parser = argparse.ArgumentParser(description="Build GitHub FAISS index from README files only")
    parser.add_argument(
        "--repos",
        nargs="*",
        help="Optional repository list in owner/repo format. Example: --repos org/repo1 org/repo2",
    )
    parser.add_argument(
        "--max-repos",
        type=int,
        default=50,
        help="When --repos is not provided, index at most this many accessible repositories",
    )
    args = parser.parse_args()

    try:
        logger.info("Validating GitHub indexing configuration...")
        _validate_required_config()

        logger.info("Initializing GitHub searcher...")
        github_searcher = GitHubSearcher(
            github_token=Config.GITHUB_TOKEN,
            organization=Config.GITHUB_ORGANIZATION,
        )

        repo_list = _resolve_repo_list(github_searcher, args.repos, args.max_repos)
        if not repo_list:
            logger.error("No repositories available for indexing.")
            return

        logger.info("Selected repositories for indexing: %s", ", ".join(repo_list))

        logger.info("Preparing README-only GitHub documents for indexing from %s repositories", len(repo_list))
        documents = []
        skipped_repos = []
        for repo_name in repo_list:
            logger.info("Collecting repository content: %s", repo_name)
            repo_docs = github_searcher.get_repository_documents_for_indexing(
                repo_name,
                include_code_files=False,
            )
            documents.extend(repo_docs)
            logger.info("Collected %s documents from %s", len(repo_docs), repo_name)
            if len(repo_docs) == 0:
                skipped_repos.append(repo_name)

        if not documents:
            logger.error("No GitHub documents were extracted. Nothing to index.")
            return

        logger.info("Initializing vector store...")
        vector_store = VectorStore(
            azure_endpoint=Config.AZURE_OPENAI_ENDPOINT,
            api_key=Config.AZURE_OPENAI_API_KEY,
            api_version=Config.AZURE_OPENAI_API_VERSION,
            embedding_deployment=Config.AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
        )

        logger.info("Creating FAISS index with one chunk per README...")
        vector_store.create_index(
            documents,
            source_name="github",
            force_single_chunk=True,
        )

        logger.info("Saving extracted GitHub documents...")
        os.makedirs(Config.VECTOR_STORE_PATH, exist_ok=True)
        with open(Config.GITHUB_DOCUMENTS_PATH, "wb") as file_obj:
            import pickle

            pickle.dump(documents, file_obj)

        logger.info("Saving GitHub FAISS index and metadata...")
        vector_store.save_index(Config.GITHUB_FAISS_INDEX_PATH, Config.GITHUB_METADATA_PATH)

        logger.info("Saving GitHub chunking report...")
        vector_store.save_chunk_report(Config.GITHUB_CHUNK_REPORT_PATH)

        for entry in vector_store.chunk_report:
            if entry.get("chunks", 0) > 0 and entry.get("url"):
                logger.info("Chunked GitHub source: %s | chunks=%s", entry.get("url"), entry.get("chunks"))

        if skipped_repos:
            logger.warning(
                "Skipped %s repositories with no readable README: %s",
                len(skipped_repos),
                ", ".join(skipped_repos),
            )

        logger.info("GitHub indexing completed successfully")
        logger.info("Indexed %s extracted GitHub documents", len(documents))
        logger.info("GitHub index path: %s", Config.GITHUB_FAISS_INDEX_PATH)
        logger.info("GitHub metadata path: %s", Config.GITHUB_METADATA_PATH)
        logger.info("GitHub chunk report path: %s", Config.GITHUB_CHUNK_REPORT_PATH)

    except Exception as exc:
        logger.error("GitHub indexing failed: %s", exc)
        raise


if __name__ == "__main__":
    main()
