from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from core.schema import ExpandedQuery

query_parser = PydanticOutputParser(pydantic_object=ExpandedQuery)

QUERY_GENERATOR_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a legal query expansion engine for Indian law.\n"
     "Your task is to expand a user question into a detailed primary query "
     "and multiple semantically similar legal queries.\n"
     "- Assume Indian jurisdiction\n"
     "- Reference Acts/Sections where relevant\n"
     "- Do NOT answer the question\n"
     "{format_instructions}"
    ),
    ("human", "User question:\n{query}")
]).partial(
    format_instructions=query_parser.get_format_instructions()
)



ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a legal assistant grounded ONLY in the provided context.\n"
        "Rules:\n"
        "- Use only the provided context\n"
        "- Cite exact article/section citations (e.g., Article 21 (COI), Section 279 (IPC))\n"
        "- Some legal terms are composite; if the question is about a commonly used term that is not a named offence, answer by combining the relevant retrieved sections\n"
        "- Do not guess or use external knowledge\n"
        "- Be detailed, clear, and human in tone\n"
        "- Prefer 2-4 short paragraphs plus a compact bullet list of cited sections and penalties\n"
        "- If key details are missing in the context, briefly state what is missing in one short sentence at the end without boilerplate disclaimers\n"
        "{format_instructions}"
    ),
    ("human", "Context:\n{context}\n\nQuestion:\n{query}")
])
