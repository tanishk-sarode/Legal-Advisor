from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from core.schema import ExpandedQuery, FinalAnswer, IntentDecision, RefinementQuery

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


router_parser = PydanticOutputParser(pydantic_object=IntentDecision)
ROUTER_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an intent router for an Indian legal assistant.\n"
        "Classify the user's request into exactly one intent and detect if clarification is required.\n"
        "Intents: legal_qa, section_finder, procedure_checklist, draft_helper, compare_provisions, risk_flagger.\n"
        "Set needs_clarification=true only when key facts are missing and without them an answer would be unreliable.\n"
        "If needs_clarification=true, provide one concise clarifying question.\n"
        "Respond ONLY valid JSON.\n"
        "{format_instructions}"
    ),
    (
        "human",
        "Conversation history:\n{chat_history}\n\nUser query:\n{query}",
    ),
]).partial(
    format_instructions=router_parser.get_format_instructions()
)


refinement_parser = PydanticOutputParser(pydantic_object=RefinementQuery)
REFINEMENT_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You generate follow-up retrieval queries for Indian legal research.\n"
        "Use the initial retrieved context to identify missing angles and produce 2-4 focused follow-up queries.\n"
        "Output ONLY JSON.\n"
        "{format_instructions}"
    ),
    (
        "human",
        "Intent: {intent}\n\nUser query:\n{query}\n\nInitial context:\n{context}",
    ),
]).partial(
    format_instructions=refinement_parser.get_format_instructions()
)


AGENT_ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an Indian legal copilot.\n"
        "You must follow strict grounding and safety constraints:\n"
        "1) Use ONLY provided context for legal claims.\n"
        "2) Every citation in cited_sections must appear in context exactly.\n"
        "3) If context is insufficient, answer must state: Not found in provided context.\n"
        "4) Respect selected scope_act. If context extends beyond scope_act, set scope_warning.\n"
        "5) Add escalation_notice for high-risk or high-impact legal situations.\n"
        "Tool behaviors by intent:\n"
        "- section_finder: provide most relevant sections/articles with short why.\n"
        "- procedure_checklist: provide practical numbered checklist.\n"
        "- draft_helper: produce a safe template with placeholders and assumptions.\n"
        "- compare_provisions: compare provisions side-by-side in plain language.\n"
        "- risk_flagger: highlight legal risks, missing facts, and uncertainties.\n"
        "- legal_qa: provide direct grounded legal answer.\n"
        "Respond ONLY valid JSON.\n"
        "{format_instructions}"
    ),
    (
        "human",
        "Intent: {intent}\n"
        "Selected scope_act: {scope_act}\n\n"
        "Conversation history:\n{chat_history}\n\n"
        "Context:\n{context}\n\n"
        "Question:\n{query}"
    ),
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