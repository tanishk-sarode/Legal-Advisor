from typing import Dict, Iterator, List
import re

from langchain_core.documents import Document
from langchain_core.output_parsers import PydanticOutputParser
from langchain_community.vectorstores import OpenSearchVectorSearch

from core.prompts import (
    QUERY_GENERATOR_PROMPT,
    ROUTER_PROMPT,
    REFINEMENT_PROMPT,
    AGENT_ANSWER_PROMPT,
    ANSWER_STREAM_PROMPT,
)
from core.schema import ExpandedQuery, FinalAnswer, IntentDecision, RefinementQuery


# ---------- Utilities ----------

def format_docs(docs: List[Document]) -> str:
    return "\n\n".join(
        f"[{d.metadata.get('citation')}]\n{d.page_content}"
        for d in docs
    )


def stringify_history(chat_history: str | None) -> str:
    if chat_history and chat_history.strip():
        return chat_history
    return "No prior conversation."

def dedupe_docs(docs: List[Document]) -> List[Document]:
    seen = set()
    unique = []
    for d in docs:
        key = (
            d.metadata.get("citation"),
            d.metadata.get("source"),
            d.page_content,
        )
        if key not in seen:
            seen.add(key)
            unique.append(d)
    return unique


class RetrievalLegalChain:
    def __init__(
        self,
        answer_llm,
        vectorstore: OpenSearchVectorSearch,
        answer_parser: PydanticOutputParser,
        query_parser: PydanticOutputParser,
        *,
        similarity_k: int = 12,
        max_context_docs: int = 16,
        max_sources: int = 24,
    ):
        self.answer_llm = answer_llm
        self.vectorstore = vectorstore
        self.answer_parser = answer_parser
        self.query_parser = query_parser
        self.similarity_k = similarity_k
        self.max_context_docs = max_context_docs
        self.max_sources = max_sources

        self.router_chain = ROUTER_PROMPT | self.answer_llm
        self.query_generator_chain = (
            QUERY_GENERATOR_PROMPT
            | self.answer_llm
        )
        self.refinement_chain = REFINEMENT_PROMPT | self.answer_llm
        self.agent_answer_chain = (
            AGENT_ANSWER_PROMPT.partial(
                format_instructions=self.answer_parser.get_format_instructions()
            )
            | self.answer_llm
        )
        self.fallback_answer_chain = ANSWER_STREAM_PROMPT | self.answer_llm

    @staticmethod
    def _parse_with_parser(raw, parser: PydanticOutputParser):
        if isinstance(raw, str):
            return parser.parse(raw)
        content = getattr(raw, "content", "")
        if isinstance(content, str):
            return parser.parse(content)
        if isinstance(content, list):
            text = "".join(
                part.get("text", "") if isinstance(part, dict) else getattr(part, "text", "")
                for part in content
            )
            return parser.parse(text)
        return parser.parse(str(raw))

    @staticmethod
    def _extract_text(raw) -> str:
        if raw is None:
            return ""
        if isinstance(raw, str):
            return raw
        content = getattr(raw, "content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            text_parts = []
            for part in content:
                if isinstance(part, dict):
                    text_parts.append(part.get("text", ""))
                else:
                    text_parts.append(getattr(part, "text", ""))
            return "".join(text_parts)
        return str(raw)

    @staticmethod
    def _extract_json_candidate(text: str) -> str:
        stripped = (text or "").strip()
        if not stripped:
            return stripped
        if stripped.startswith("{") and stripped.endswith("}"):
            return stripped
        match = re.search(r"\{[\s\S]*\}", stripped)
        return match.group(0) if match else stripped

    def _safe_parse_intent(self, raw) -> IntentDecision:
        parser = PydanticOutputParser(pydantic_object=IntentDecision)
        text = self._extract_json_candidate(self._extract_text(raw))
        try:
            return parser.parse(text)
        except Exception:
            return IntentDecision(
                intent="legal_qa",
                needs_clarification=False,
                clarifying_question="",
            )

    def _safe_parse_expanded_query(self, raw, query: str) -> ExpandedQuery:
        text = self._extract_json_candidate(self._extract_text(raw))
        try:
            return self.query_parser.parse(text)
        except Exception:
            return ExpandedQuery(primary_issue=query, sub_queries=[query])

    def _safe_parse_refinement(self, raw) -> RefinementQuery:
        parser = PydanticOutputParser(pydantic_object=RefinementQuery)
        text = self._extract_json_candidate(self._extract_text(raw))
        try:
            return parser.parse(text)
        except Exception:
            return RefinementQuery(sub_queries=[])

    def _safe_parse_final_answer(self, raw, *, intent: str) -> FinalAnswer:
        text = self._extract_json_candidate(self._extract_text(raw))
        try:
            answer = self.answer_parser.parse(text)
            return answer
        except Exception:
            fallback_text = self._extract_text(raw).strip() or "Not found in provided context."
            return FinalAnswer(
                answer=fallback_text,
                cited_sections=[],
                tool_used=intent,
                risk_flags=["Model output parse fallback used."],
                escalation_notice="",
                scope_warning="",
            )

    @staticmethod
    def _format_query_list(queries: List[str]) -> str:
        return "\n".join(f"{index}. {item}" for index, item in enumerate(queries, 1))

    def _route_intent(self, *, query: str, chat_history: str) -> IntentDecision:
        routed = self.router_chain.invoke({"query": query, "chat_history": chat_history})
        return self._safe_parse_intent(routed)

    def _collect_docs(self, queries: List[str]) -> List[Document]:
        docs: List[Document] = []
        for sub_query in queries:
            docs.extend(self.vectorstore.similarity_search(sub_query, k=self.similarity_k))
        return dedupe_docs(docs)

    def _two_pass_retrieval(
        self,
        *,
        query: str,
        chat_history: str,
        intent: str,
        act: str | None,
    ) -> tuple[List[str], List[Document], bool]:
        expanded_raw = self.query_generator_chain.invoke(
            {"query": query, "chat_history": chat_history}
        )
        expanded: ExpandedQuery = self._safe_parse_expanded_query(expanded_raw, query=query)
        pass1_queries = expanded.sub_queries if expanded.sub_queries else [query]
        docs_pass1 = self._collect_docs(pass1_queries)

        context_pass1 = format_docs(docs_pass1[: min(len(docs_pass1), 8)])
        refined_raw = self.refinement_chain.invoke(
            {
                "intent": intent,
                "query": query,
                "context": context_pass1,
            }
        )
        refined: RefinementQuery = self._safe_parse_refinement(refined_raw)
        pass2_queries = [item for item in refined.sub_queries if item.strip()][:4]

        docs_pass2 = self._collect_docs(pass2_queries) if pass2_queries else []
        combined_docs = dedupe_docs(docs_pass1 + docs_pass2)

        scope_warning = False
        if act and act != "All":
            filtered = [doc for doc in combined_docs if doc.metadata.get("act_abbrev") == act]
            if filtered:
                combined_docs = filtered
            else:
                scope_warning = True

        combined_queries = pass1_queries + pass2_queries
        return combined_queries, combined_docs, scope_warning

    @staticmethod
    def _validate_citations(answer: FinalAnswer, docs: List[Document]) -> FinalAnswer:
        ordered_available = []
        for doc in docs:
            citation = doc.metadata.get("citation")
            if citation and citation not in ordered_available:
                ordered_available.append(citation)

        available_set = set(ordered_available)
        valid = [citation for citation in answer.cited_sections if citation in available_set]
        if not valid and answer.answer.strip() != "Not found in provided context.":
            if ordered_available:
                answer.cited_sections = ordered_available[:3]
                if "Auto-attached citations from retrieved context." not in answer.risk_flags:
                    answer.risk_flags.append("Auto-attached citations from retrieved context.")
                return answer

            answer.answer = "Not found in provided context."
            answer.cited_sections = []
            if "Insufficient grounded evidence in retrieved context." not in answer.risk_flags:
                answer.risk_flags.append("Insufficient grounded evidence in retrieved context.")
            return answer

        answer.cited_sections = valid
        return answer

    @staticmethod
    def _compose_final_text(answer: FinalAnswer) -> str:
        tool_title_map = {
            "legal_qa": "Legal Answer",
            "section_finder": "Section Finder",
            "procedure_checklist": "Procedure Checklist",
            "draft_helper": "Draft Helper",
            "compare_provisions": "Provision Comparison",
            "risk_flagger": "Risk Review",
        }
        label = tool_title_map.get(answer.tool_used, "Legal Answer")

        chunks = [f"{label}\n\n{answer.answer.strip()}"]
        if answer.scope_warning:
            chunks.append(f"Scope note: {answer.scope_warning}")
        if answer.risk_flags:
            risk_items = "\n".join(f"- {item}" for item in answer.risk_flags)
            chunks.append(f"Risk flags:\n{risk_items}")
        if answer.escalation_notice:
            chunks.append(f"Escalation: {answer.escalation_notice}")
        return "\n\n".join(chunk for chunk in chunks if chunk)

    def _run_agent(self, query: str, act: str | None, chat_history: str | None):
        history = stringify_history(chat_history)
        decision = self._route_intent(query=query, chat_history=history)

        if decision.needs_clarification and decision.clarifying_question.strip():
            answer = FinalAnswer(
                answer=decision.clarifying_question.strip(),
                cited_sections=[],
                tool_used=decision.intent,
                risk_flags=[],
                escalation_notice="",
                scope_warning="",
            )
            return answer, [], []

        retrieval_queries, docs, scope_warning_flag = self._two_pass_retrieval(
            query=query,
            chat_history=history,
            intent=decision.intent,
            act=act,
        )

        if len(docs) > self.max_sources:
            docs = docs[: self.max_sources]

        docs_for_answer = docs[: self.max_context_docs]

        context = format_docs(docs_for_answer)
        if retrieval_queries:
            context = (
                "Search queries used:\n"
                f"{self._format_query_list(retrieval_queries)}\n\n"
                + context
            )

        answer_raw = self.agent_answer_chain.invoke(
            {
                "intent": decision.intent,
                "scope_act": act or "All",
                "chat_history": history,
                "query": query,
                "context": context,
            }
        )
        answer: FinalAnswer = self._safe_parse_final_answer(answer_raw, intent=decision.intent)

        if "Model output parse fallback used." in answer.risk_flags and docs_for_answer:
            fallback_raw = self.fallback_answer_chain.invoke(
                {
                    "chat_history": history,
                    "context": context,
                    "query": query,
                }
            )
            fallback_text = self._extract_text(fallback_raw).strip()
            if fallback_text:
                answer.answer = fallback_text
                answer.risk_flags = [
                    item
                    for item in answer.risk_flags
                    if item != "Model output parse fallback used."
                ]

        answer.tool_used = decision.intent
        answer = self._validate_citations(answer, docs_for_answer)
        if scope_warning_flag:
            answer.scope_warning = (
                f"No strong evidence was found within selected scope '{act}'. "
                "Results may include broader acts."
            )

        if not answer.escalation_notice and any(
            marker in query.lower()
            for marker in ["arrest", "bail", "domestic violence", "custody", "criminal", "firing", "termination"]
        ):
            answer.escalation_notice = "Consult a qualified lawyer before taking legal action."

        return answer, docs, retrieval_queries

    def invoke(self, inputs):
        query = inputs["query"]
        act = inputs.get("act")
        chat_history = inputs.get("chat_history")
        answer, docs, _ = self._run_agent(
            query=query,
            act=act,
            chat_history=chat_history,
        )
        return {"answer": answer, "sources": docs}

    @staticmethod
    def _chunk_to_text(chunk) -> str:
        content = getattr(chunk, "content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            output = []
            for part in content:
                if isinstance(part, dict) and "text" in part:
                    output.append(part["text"])
                elif hasattr(part, "text"):
                    output.append(getattr(part, "text"))
            return "".join(output)
        return ""

    def stream(self, inputs) -> Iterator[Dict[str, object]]:
        query = inputs["query"]
        act = inputs.get("act")
        chat_history = inputs.get("chat_history")

        answer, docs, _ = self._run_agent(
            query=query,
            act=act,
            chat_history=chat_history,
        )

        full_text = self._compose_final_text(answer)
        for token in full_text.split(" "):
            yield {"type": "token", "content": token + " "}

        yield {
            "type": "done",
            "content": full_text.strip(),
            "sources": docs,
        }


# ---------- Chain Builder ----------

def build_chain(
    answer_llm,
    vectorstore: OpenSearchVectorSearch,
    answer_parser: PydanticOutputParser,
    query_parser: PydanticOutputParser,
    *,
    similarity_k: int = 12,
    max_context_docs: int = 16,
    max_sources: int = 24,
):
    return RetrievalLegalChain(
        answer_llm=answer_llm,
        vectorstore=vectorstore,
        answer_parser=answer_parser,
        query_parser=query_parser,
        similarity_k=similarity_k,
        max_context_docs=max_context_docs,
        max_sources=max_sources,
    )
