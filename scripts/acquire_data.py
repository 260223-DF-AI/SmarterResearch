"""
Data Acquisition

Requests various scholarly articles from arxiv.org to be used
in the data corpus powering the application's RAG capabilities.
"""

from arxiv import Client, Search
import time

CATEGORY_CODES: list[str] = [
    "cs.DB",  # Databases
    "cs.LG",  # Machine learning
    "cs.SE",  # Software Engineering
]

arxiv_client = Client()

def download_category_papers(cat_code: str, results=5):
    """
    Downloads `results` papers from `cat_code` category
    from the arxiv.org website.
    """

    search = Search(query=f"cat:{cat_code}", max_results=results)
    results = arxiv_client.results(search)

    paper_ids: set[int] = set() # update to use file names in /data/corpus directory

    for paper in results:
        print(f"Downloading {paper.title}")
        
        id = paper.get_short_id()
        if id in paper_ids: # already downloaded this paper
            continue
        paper.download_pdf(dirpath="./data/corpus/", filename=f"{id}.pdf")
        paper_ids.add(id)
        time.sleep(2)

if __name__ == "__main__":
    for code in CATEGORY_CODES:
        download_category_papers(code, max_results=1)
