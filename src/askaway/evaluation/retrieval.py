def is_relevant(
    result: dict,
    relevant_pages: list[dict],
) -> bool:
    for relevant in relevant_pages:
        same_document = (
            result["document_id"]
            == relevant["document_id"]
        )

        same_page = (
            result["page_number"]
            == relevant["page"]
        )

        if same_document and same_page:
            return True

    return False


def reciprocal_rank(
    results: list[dict],
    relevant_pages: list[dict],
) -> float:
    for result in results:
        if is_relevant(result, relevant_pages):
            return 1.0 / result["rank"]

    return 0.0


def recall_at_k(
    results: list[dict],
    relevant_pages: list[dict],
    k: int,
) -> float:
    return float(
        any(
            is_relevant(result, relevant_pages)
            for result in results[:k]
        )
    )