from typing import List, NamedTuple

from database.qdrant import client as qdrant_store

from qa_router.schemas import SearchResult

RelevantChunksResult = NamedTuple('RelevantChunksResult', [('text', str), ('search_result', List[SearchResult])])


async def search_relevant_chunks(vault_id: str, vector: list[float], top_k: int) -> RelevantChunksResult:
    response = await qdrant_store.search(collection_name=vault_id,
                                         query_vector=vector,
                                         limit=top_k)

    result_text = '\n\n'.join(list(map(lambda x: x.payload['page_content'], response)))
    search_result = [SearchResult(document_id=x.payload['document_id'],
                                  information=x.payload['page_content'],
                                  document_name=x.payload.get('document_name', "no_name_found")) for x in response]
    return RelevantChunksResult(text=result_text, search_result=search_result)
