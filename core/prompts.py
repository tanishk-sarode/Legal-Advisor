from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from core.schema import ExpandedQuery, FinalAnswer

query_generator_parser = PydanticOutputParser(pydantic_object=ExpandedQuery)

QUERY_GENERATOR_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a legal issue decomposition engine for Indian law.\n"
     "Break the user's problem into structured legal retrieval queries.\n"
     "\n"
     "Output JSON with:\n"
     "- primary_issue: short legal description\n"
     "- sub_queries: list of focused legal search queries\n"
     "\n"
     "Rules:\n"
     "- Do NOT answer the question\n"
     "- Cover proof requirements, remedies, procedure, enforcement\n"
     "- Include both civil and criminal options if applicable\n"
     "- Generate 6-10 high-quality retrieval queries\n"
     "{format_instructions}"
    ),
    ("human", "User problem:\n{query}")
]).partial(
    format_instructions=query_generator_parser.get_format_instructions()
)


answer_parser = PydanticOutputParser(pydantic_object=FinalAnswer)

ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a legal assistant grounded ONLY in the provided context.\n"
        "You MUST respond with ONLY a valid JSON object. No other text, no explanations, no preamble.\n"
        "\n"
        "Rules for your response:\n"
        "- Output ONLY valid JSON. Nothing else.\n"
        "- Use only the provided context\n"
        "- Cite exact article/section citations (e.g., Article 21 (COI), Section 279 (IPC))\n"
        "- Some legal terms are composite; if the question is about a commonly used term that is not a named offence, answer by combining the relevant retrieved sections\n"
        "- Be detailed, clear, and human in tone when context exists\n"
        "- Provide 2-4 short paragraphs in the answer field plus a compact bullet list of cited sections and penalties\n"
        "- If the exact article/section is not in the context, set answer to \"Not found in provided context.\" and cited_sections to empty list\n"
        "\n"
        "{format_instructions}"
    ),
    ("human", "Context:\n{context}\n\nQuestion:\n{query}\n\nRespond with ONLY the JSON object, no other text:")
]).partial(
    format_instructions=answer_parser.get_format_instructions()
)
