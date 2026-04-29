"""
Data Acquisition

Requests various scholarly articles from arxiv.org to be used
in the data corpus powering the application's RAG capabilities.

Run this file with the current working directory at the project root.
"""

import os
import time

from arxiv import Client, Search

CATEGORY_CODES: list[str] = [
    "cs.DB",  # Databases
    "cs.LG",  # Machine learning
    "cs.SE",  # Software Engineering
]

CORPUS_DIR: str = os.path.join("data", "corpus")

arxiv_client = Client()


def _get_downloaded_papers(dir: str):
    """
    Yields names of files in given directory, excluding extension.

    dir: String representing directory path to gather file names from
    """

    for file in os.listdir(dir):
        *file, ext = file.split(".")
        yield (".".join(file), ext)


def download_category_papers(cat_code: str, num_results=5) -> tuple[int, int]:
    """
    Downloads `results` papers from `cat_code` category
    from the arxiv.org website.

    - cat_code: String code representing the category of research paper to query
    - num_results: Integer, denotes how many files to download from the given category
    """

    # search arxiv papers belonging to one category
    search = Search(query=f"cat:{cat_code}", max_results=num_results)
    results = arxiv_client.results(search)

    total_downloaded = total_skipped = 0

    # set holds downloaded papers
    local_files: set[int] = set()
    for file_name, _ in _get_downloaded_papers(CORPUS_DIR):
        local_files.add(file_name)

    # download the papers found from search if not already stored
    for paper in results:
        id = paper.get_short_id()
        local_file_name = f"{cat_code}_{id}"
        if local_file_name in local_files:  # already downloaded this paper
            total_skipped += 1
            continue

        paper.download_pdf(dirpath=CORPUS_DIR, filename=f"{cat_code}_{id}.pdf")
        local_files.add(id)

        total_downloaded += 1
        time.sleep(2)

    return (total_downloaded, total_skipped)


if __name__ == "__main__":
    # save papers from each listed category
    for code in CATEGORY_CODES:
        downloaded, skipped = download_category_papers(code, num_results=8)
        print(
            f"For {code}: Downloaded {downloaded} papers; already had {skipped} papers locally."
        )
