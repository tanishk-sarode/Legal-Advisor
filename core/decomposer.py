from typing import List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnableLambda

PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You decompose legal queries into atomic retrieval sub-queries.\n"
        "Rules:\n"
        "- Output valid JSON only\n"
        "- Max 5 sub-queries\n"
        "- Prefer exact article/section references when present\n"
        "- If a specific act is provided, keep sub-queries within that act\n\n"
        "Output format:\n"
        "{{ \"sub_queries\": [\"...\"] }}"
    ),
    ("human", "Act filter: {act}\nQuery: {query}")
])


def build_decomposer(llm):
    parser = JsonOutputParser()

    def _extract_sub_queries(x) -> List[str]:
        if isinstance(x, dict):
            subs = x.get("sub_queries")
            if isinstance(subs, list):
                return [s for s in subs if isinstance(s, str) and s.strip()]
        return []

    return PROMPT | llm | parser | RunnableLambda(_extract_sub_queries)
