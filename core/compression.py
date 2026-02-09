from langchain_core.runnables import RunnableLambda


def build_compressor(compressor):
    def _compress(x):
        docs = x.get("docs") or []
        if not docs:
            return []
        return compressor.compress_documents(docs, x.get("query", ""))

    return RunnableLambda(_compress)
