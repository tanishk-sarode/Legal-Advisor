from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from core.schema import ExpandedQuery, FinalAnswer

query_generator_parser = PydanticOutputParser(pydantic_object=ExpandedQuery)

QUERY_GENERATOR_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a legal issue decomposition engine for Indian law.\n"
     "You MUST respond with ONLY a valid JSON object. No other text, no explanations, no preamble.\n"
     "Break the user's problem into structured legal retrieval queries.\n"
     "\n"
     "Output JSON with:\n"
     "- primary_issue: short legal description\n"
     "- sub_queries: list of focused legal search queries\n"
     "\n"
     "Rules:\n""- Output ONLY valid JSON. Nothing else.\n"
     "- Do NOT answer the question\n"
     "- Cover proof requirements, remedies, procedure, enforcement\n"
     "- Include both civil and criminal options if applicable\n"
     "- Generate 6-10 high-quality retrieval queries\n"
     "{format_instructions}"
    ),
    (
        "human",
        "Conversation history (for context/disambiguation only):\n{chat_history}\n\n"
        "User problem:\n{query}",
    )
]).partial(
    format_instructions=query_generator_parser.get_format_instructions()
)


answer_parser = PydanticOutputParser(pydantic_object=FinalAnswer)
ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an expert legal assistant specializing in Indian law.\n"
        "\n"
        "CRITICAL INSTRUCTIONS:\n"
        "1. You are STRICTLY limited to the provided context.\n"
        "2. You MUST NOT use any prior knowledge, assumptions, or inference beyond the text explicitly present in the context.\n"
        "3. Every legal statement MUST be directly traceable to the provided context.\n"
        "4. Every cited section/article MUST appear explicitly in the provided context.\n"
        "5. If the answer cannot be derived fully and explicitly from the provided context, you MUST return EXACTLY:\n"
        "   {{\n"
        "     \"answer\": \"Not found in provided context.\",\n"
        "     \"cited_sections\": []\n"
        "   }}\n"
        "6. Do NOT partially answer. Do NOT infer missing provisions.\n"
        "7. Output ONLY a valid JSON object. No explanations. No markdown. No trailing text.\n"
        "\n"
        "ANSWER QUALITY GUIDELINES:\n"
        "- Provide comprehensive, detailed explanations (5-8 paragraphs)\n"
        "- Structure your answer with: overview, legal provisions, practical implications, procedures, summary\n"
        "- Include specific details from the source text\n"
        "- Use the search queries as guidance for what aspects to cover in your answer\n"
        "\n"
        "{format_instructions}"
    ),
    (
        "human",
        "Context (includes search queries used and relevant legal provisions):\n{context}\n\n"
        "Question:\n{query}\n\n"
        "Respond with ONLY the JSON object:"
    )
]).partial(
    format_instructions=answer_parser.get_format_instructions()
)


ANSWER_STREAM_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an expert legal assistant specializing in Indian law.\n"
        "Use ONLY the provided legal context for legal claims.\n"
        "Use conversation history only to resolve user intent and references.\n"
        "If context is insufficient, clearly say: Not found in provided context.\n"
        "Be concise but complete, with clear headings and practical steps.\n"
    ),
    (
        "human",
        "Conversation history:\n{chat_history}\n\n"
        "Context:\n{context}\n\n"
        "Question:\n{query}",
    ),
])