from typing import Dict, Iterator, List
import logging
import re
import time

from langchain_core.documents import Document
from langchain_core.output_parsers import PydanticOutputParser
from langchain_community.vectorstores import OpenSearchVectorSearch

from core.prompts import (
    QUERY_GENERATOR_PROMPT,
    ROUTER_PROMPT,
    REFINEMENT_PROMPT,
    AGENT_ANSWER_PROMPT,
    ANSWER_STREAM_PROMPT,
    CITATION_TRIGGER_PROMPT,
    SOURCE_FILTER_PROMPT,
)
from core.schema import (
    CitationTriggerDecision,
    ExpandedQuery,
    FinalAnswer,
    IntentDecision,
    RefinementQuery,
    SourceFilterDecision,
)

logger = logging.getLogger(__name__)


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
    _ENRICHMENT_TRIGGER_KEYWORDS: frozenset = frozenset([
        "article", "section", "clause", "act",
        "ipc", "crpc", "cpc", "hma", "iea", "nia", "mva",
        "constitution", "sub-section", "rule",
    ])

    def __init__(
        self,
        answer_llm,
        source_filter_llm,
        vectorstore: OpenSearchVectorSearch,
        answer_parser: PydanticOutputParser,
        query_parser: PydanticOutputParser,
        *,
        similarity_k: int = 12,
        max_context_docs: int = 16,
        max_sources: int = 24,
        hybrid_k: int = 20,
        hybrid_fallback_threshold: int = 6,
        enrichment_confidence_threshold: float = 0.5,
    ):
        self.answer_llm = answer_llm
        self.source_filter_llm = source_filter_llm
        self.vectorstore = vectorstore
        self.answer_parser = answer_parser
        self.query_parser = query_parser
        self.source_filter_parser = PydanticOutputParser(pydantic_object=SourceFilterDecision)
        self.citation_trigger_parser = PydanticOutputParser(pydantic_object=CitationTriggerDecision)
        self.similarity_k = similarity_k
        self.max_context_docs = max_context_docs
        self.max_sources = max_sources
        self.max_enrichment_loops = 1
        self.max_direct_reference_queries = 4
        self.direct_reference_k = 4
        self.hybrid_k = hybrid_k
        self.hybrid_fallback_threshold = hybrid_fallback_threshold
        self.enrichment_confidence_threshold = enrichment_confidence_threshold

        self.router_chain = ROUTER_PROMPT | self.answer_llm
        self.query_generator_chain = QUERY_GENERATOR_PROMPT | self.answer_llm
        self.refinement_chain = REFINEMENT_PROMPT | self.answer_llm
        self.agent_answer_chain = (
            AGENT_ANSWER_PROMPT.partial(
                format_instructions=self.answer_parser.get_format_instructions()
            )
            | self.answer_llm
        )
        self.fallback_answer_chain = ANSWER_STREAM_PROMPT | self.answer_llm
        self.source_filter_chain = (
            SOURCE_FILTER_PROMPT.partial(
                format_instructions=self.source_filter_parser.get_format_instructions()
            )
            | self.source_filter_llm
        )
        self.citation_trigger_chain = (
            CITATION_TRIGGER_PROMPT.partial(
                format_instructions=self.citation_trigger_parser.get_format_instructions()
            )
            | self.source_filter_llm
        )

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

    def _safe_parse_source_filter(self, raw) -> SourceFilterDecision | None:
        text = self._extract_json_candidate(self._extract_text(raw))
        try:
            return self.source_filter_parser.parse(text)
        except Exception:
            return None

    def _safe_parse_citation_trigger(self, raw) -> CitationTriggerDecision:
        text = self._extract_json_candidate(self._extract_text(raw))
        try:
            parsed = self.citation_trigger_parser.parse(text)
            return CitationTriggerDecision(
                has_explicit_references=parsed.has_explicit_references,
                references=parsed.references,
                confidence=max(0.0, min(1.0, parsed.confidence)),
            )
        except Exception:
            return CitationTriggerDecision(
                has_explicit_references=False,
                references=[],
                confidence=0.0,
            )

    @staticmethod
    def _format_query_list(queries: List[str]) -> str:
        return "\n".join(f"{index}. {item}" for index, item in enumerate(queries, 1))

    @staticmethod
    def _format_sources_for_filter(docs: List[Document], *, snippet_chars: int = 1200) -> str:
        chunks = []
        for index, doc in enumerate(docs, 1):
            source_id = f"S{index}"
            citation = doc.metadata.get("citation") or "Unknown citation"
            act = doc.metadata.get("act_abbrev") or doc.metadata.get("act") or "N/A"
            text = (doc.page_content or "").strip()
            if len(text) > snippet_chars:
                text = text[:snippet_chars].rstrip() + "..."
            chunks.append(
                f"[{source_id}] citation={citation} | act={act}\n{text}"
            )
        return "\n\n".join(chunks)

    def _filter_useful_docs(self, *, query: str, docs: List[Document]) -> List[Document]:
        if not docs:
            return []

        filter_input = self._format_sources_for_filter(docs)
        raw = self.source_filter_chain.invoke({"query": query, "sources": filter_input})
        decision = self._safe_parse_source_filter(raw)

        if decision is None:
            logger.warning("Source filter parse failed; keeping original retrieved docs")
            return docs

        source_id_to_doc = {f"S{index}": doc for index, doc in enumerate(docs, 1)}
        useful_docs: List[Document] = []
        for source_id in decision.useful_source_ids:
            doc = source_id_to_doc.get(source_id)
            if doc is not None:
                useful_docs.append(doc)

        logger.info("Source filter kept %s of %s docs", len(useful_docs), len(docs))
        return useful_docs

    @staticmethod
    def _normalize_act_abbrev(act_abbrev: str) -> str:
        alias_map = {
            "IP": "IPC",
            "IPC": "IPC",
            "CRPC": "CrPC",
            "CR. P.C": "CrPC",
            "CR.P.C": "CrPC",
            "CPC": "CPC",
            "COI": "COI",
            "CONSTITUTION": "COI",
            "HMA": "HMA",
            "IDA": "IDA",
            "IEA": "IEA",
            "NIA": "NIA",
            "MVA": "MVA",
        }
        key = (act_abbrev or "").strip().upper()
        return alias_map.get(key, (act_abbrev or "").strip())

    @staticmethod
    def _query_has_citation_keywords(query: str) -> bool:
        lowered = query.lower()
        return any(kw in lowered for kw in RetrievalLegalChain._ENRICHMENT_TRIGGER_KEYWORDS)

    def _extract_citation_triggers(self, *, query: str, chat_history: str) -> CitationTriggerDecision:
        raw = self.citation_trigger_chain.invoke(
            {
                "query": query,
                "chat_history": chat_history,
            }
        )
        decision = self._safe_parse_citation_trigger(raw)
        logger.info(
            "Citation triggers extracted: has_refs=%s count=%s confidence=%.2f",
            decision.has_explicit_references,
            len(decision.references),
            decision.confidence,
        )
        return decision

    def _build_direct_reference_queries(self, trigger: CitationTriggerDecision) -> List[str]:
        queries: List[str] = []
        for ref in trigger.references:
            act_abbrev = self._normalize_act_abbrev(ref.act_abbrev)
            identifier = (ref.identifier or "").strip()
            if not identifier and ref.kind != "act":
                continue

            if ref.kind == "article":
                query = f"Article {identifier} {act_abbrev}".strip()
            elif ref.kind == "section":
                query = f"Section {identifier} {act_abbrev}".strip()
            else:
                query = f"{identifier} {act_abbrev}".strip()

            if query and query not in queries:
                queries.append(query)
            if len(queries) >= self.max_direct_reference_queries:
                break

        return queries

    def _fetch_direct_reference_docs(self, trigger: CitationTriggerDecision) -> List[Document]:
        queries = self._build_direct_reference_queries(trigger)
        if not queries:
            return []

        docs: List[Document] = []
        for query in queries:
            logger.info("Direct citation fetch query: %s", query)
            docs.extend(self.vectorstore.similarity_search(query, k=self.direct_reference_k))

        deduped = dedupe_docs(docs)
        logger.info("Direct citation fetch returned %s unique docs", len(deduped))
        return deduped

    def _route_intent(self, *, query: str, chat_history: str) -> IntentDecision:
        logger.info("Routing intent")
        routed = self.router_chain.invoke({"query": query, "chat_history": chat_history})
        decision = self._safe_parse_intent(routed)
        logger.info(
            "Intent routed: intent=%s needs_clarification=%s",
            decision.intent,
            decision.needs_clarification,
        )
        return decision

    def _collect_docs(self, queries: List[str]) -> List[Document]:
        docs: List[Document] = []
        for index, sub_query in enumerate(queries, 1):
            logger.info("Retrieval query %s/%s", index, len(queries))
            query_docs = self.vectorstore.similarity_search(sub_query, k=self.similarity_k)
            logger.info("Retrieved %s docs for query %s", len(query_docs), index)
            docs.extend(query_docs)

        deduped = dedupe_docs(docs)
        logger.info("Deduped docs: raw=%s unique=%s", len(docs), len(deduped))
        return deduped

    def _hybrid_search(
        self,
        query: str,
        *,
        k: int,
        filter_clause: dict | None = None,
    ) -> List[Document]:
        """Run kNN (semantic) + BM25 (keyword) legs and return merged deduped results."""
        knn_kwargs: dict = {}
        if filter_clause:
            knn_kwargs["filter"] = filter_clause
        knn_docs = self.vectorstore.similarity_search(query, k=k, **knn_kwargs)

        bm25_docs: List[Document] = []
        try:
            if filter_clause:
                bm25_body: dict = {
                    "query": {
                        "bool": {
                            "must": {
                                "multi_match": {
                                    "query": query,
                                    "fields": ["text", "metadata.title^2", "metadata.citation^3"],
                                    "type": "best_fields",
                                }
                            },
                            "filter": filter_clause,
                        }
                    },
                    "size": k,
                }
            else:
                bm25_body = {
                    "query": {
                        "multi_match": {
                            "query": query,
                            "fields": ["text", "metadata.title^2", "metadata.citation^3"],
                            "type": "best_fields",
                        }
                    },
                    "size": k,
                }
            resp = self.vectorstore.client.search(
                index=self.vectorstore.index_name,
                body=bm25_body,
            )
            for hit in resp.get("hits", {}).get("hits", []):
                source = hit.get("_source", {})
                text = source.get("text") or source.get("page_content") or ""
                meta = source.get("metadata") or {}
                if text:
                    bm25_docs.append(Document(page_content=text, metadata=meta))
        except Exception as exc:
            logger.warning("BM25 leg of hybrid search failed: %s", exc)

        merged = dedupe_docs(knn_docs + bm25_docs)
        logger.info(
            "Hybrid search: knn=%s bm25=%s merged=%s query_snippet=%.80s",
            len(knn_docs),
            len(bm25_docs),
            len(merged),
            query,
        )
        return merged

    def _hybrid_retrieval(
        self,
        *,
        query: str,
        chat_history: str,
        intent: str,
        act: str | None,
    ) -> tuple[List[str], List[Document], bool]:
        """Single-pass hybrid retrieval: 1 primary call + 1 optional fallback."""
        filter_clause: dict | None = None
        scope_warning = False

        if act and act != "All":
            filter_clause = {"term": {"metadata.act_abbrev.keyword": act}}
            docs = self._hybrid_search(query, k=self.hybrid_k, filter_clause=filter_clause)
            logger.info("Primary hybrid (scoped act=%s): %s docs", act, len(docs))
            if len(docs) < self.hybrid_fallback_threshold:
                logger.warning(
                    "Scoped hybrid weak (%s docs); widening to all acts", len(docs)
                )
                docs = self._hybrid_search(query, k=self.hybrid_k, filter_clause=None)
                scope_warning = True
        else:
            docs = self._hybrid_search(query, k=self.hybrid_k, filter_clause=None)
            logger.info("Primary hybrid (unscoped): %s docs", len(docs))
            if len(docs) < self.hybrid_fallback_threshold:
                fallback_query = f"{intent.replace('_', ' ')} India law {query}"
                logger.info("Weak primary (%s docs); running fallback hybrid", len(docs))
                fallback_docs = self._hybrid_search(
                    fallback_query, k=self.hybrid_k, filter_clause=None
                )
                docs = dedupe_docs(docs + fallback_docs)
                logger.info("After fallback: %s docs", len(docs))

        return [query], docs, scope_warning

    # NOTE: _two_pass_retrieval kept for reference; _hybrid_retrieval is the active path.
    def _two_pass_retrieval(
        self,
        *,
        query: str,
        chat_history: str,
        intent: str,
        act: str | None,
    ) -> tuple[List[str], List[Document], bool]:
        logger.info("Starting two-pass retrieval")
        expanded_raw = self.query_generator_chain.invoke(
            {"query": query, "chat_history": chat_history}
        )
        expanded: ExpandedQuery = self._safe_parse_expanded_query(expanded_raw, query=query)
        pass1_queries = expanded.sub_queries if expanded.sub_queries else [query]
        logger.info("Pass-1 queries generated: %s", len(pass1_queries))
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
        logger.info("Pass-2 queries generated: %s", len(pass2_queries))

        docs_pass2 = self._collect_docs(pass2_queries) if pass2_queries else []
        combined_docs = dedupe_docs(docs_pass1 + docs_pass2)
        logger.info("Combined retrieved docs after dedupe: %s", len(combined_docs))

        scope_warning = False
        if act and act != "All":
            filtered = [doc for doc in combined_docs if doc.metadata.get("act_abbrev") == act]
            if filtered:
                logger.info("Applied scope filter: act=%s docs=%s", act, len(filtered))
                combined_docs = filtered
            else:
                logger.warning("No docs found inside selected scope: act=%s", act)
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
        start_time = time.perf_counter()
        logger.info("Agent run started: act=%s", act or "All")
        history = stringify_history(chat_history)
        decision = self._route_intent(query=query, chat_history=history)

        if decision.needs_clarification and decision.clarifying_question.strip():
            logger.info("Clarification required by router")
            answer = FinalAnswer(
                answer=decision.clarifying_question.strip(),
                cited_sections=[],
                tool_used=decision.intent,
                risk_flags=[],
                escalation_notice="",
                scope_warning="",
            )
            logger.info("Agent run finished with clarification in %.2fs", time.perf_counter() - start_time)
            return answer, [], []

        retrieval_queries, docs, scope_warning_flag = self._hybrid_retrieval(
            query=query,
            chat_history=history,
            intent=decision.intent,
            act=act,
        )

        trigger = None
        if self._query_has_citation_keywords(query):
            trigger = self._extract_citation_triggers(query=query, chat_history=history)
            if trigger.confidence < self.enrichment_confidence_threshold:
                logger.info(
                    "Enrichment skipped: confidence=%.2f < threshold=%.2f",
                    trigger.confidence,
                    self.enrichment_confidence_threshold,
                )
                trigger = None
        if trigger is not None:
            enrichment_loops = 0
            while enrichment_loops < self.max_enrichment_loops:
                if not trigger.has_explicit_references:
                    break
                direct_docs = self._fetch_direct_reference_docs(trigger)
                docs = dedupe_docs(docs + direct_docs)
                enrichment_loops += 1
                logger.info(
                    "Citation enrichment applied: loop=%s merged_docs=%s",
                    enrichment_loops,
                    len(docs),
                )
                break

        if len(docs) > self.max_sources:
            logger.info("Trimming sources from %s to %s", len(docs), self.max_sources)
            docs = docs[: self.max_sources]

        docs = self._filter_useful_docs(query=query, docs=docs)

        docs_for_answer = docs[: self.max_context_docs]
        logger.info(
            "Preparing answer context with docs_for_answer=%s total_docs=%s",
            len(docs_for_answer),
            len(docs),
        )

        context = format_docs(docs_for_answer)
        if retrieval_queries:
            context = (
                "Search queries used:\n"
                f"{self._format_query_list(retrieval_queries)}\n\n"
                + context
            )

        logger.info("Invoking structured answer chain")
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
            logger.warning("Structured parser fallback triggered; invoking fallback answer chain")
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

        logger.info(
            "Agent run finished: cited_sections=%s retrieval_queries=%s elapsed=%.2fs",
            len(answer.cited_sections),
            len(retrieval_queries),
            time.perf_counter() - start_time,
        )
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
        request_id = inputs.get("request_id", "na")

        stream_start = time.perf_counter()
        logger.info("Stream started: request_id=%s act=%s", request_id, act or "All")

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
            full_text = self._compose_final_text(answer)
            yield {"type": "token", "content": full_text}
            yield {
                "type": "done",
                "content": full_text,
                "sources": [],
            }
            logger.info("Stream finished with clarification: request_id=%s", request_id)
            return

        retrieval_queries, docs, scope_warning_flag = self._hybrid_retrieval(
            query=query,
            chat_history=history,
            intent=decision.intent,
            act=act,
        )

        trigger = None
        if self._query_has_citation_keywords(query):
            trigger = self._extract_citation_triggers(query=query, chat_history=history)
            if trigger.confidence < self.enrichment_confidence_threshold:
                logger.info(
                    "Enrichment skipped (stream): confidence=%.2f < threshold=%.2f",
                    trigger.confidence,
                    self.enrichment_confidence_threshold,
                )
                trigger = None
        if trigger is not None:
            enrichment_loops = 0
            while enrichment_loops < self.max_enrichment_loops:
                if not trigger.has_explicit_references:
                    break
                direct_docs = self._fetch_direct_reference_docs(trigger)
                docs = dedupe_docs(docs + direct_docs)
                enrichment_loops += 1
                logger.info(
                    "Citation enrichment applied (stream): loop=%s merged_docs=%s",
                    enrichment_loops,
                    len(docs),
                )
                break

        if len(docs) > self.max_sources:
            logger.info("Trimming streamed sources from %s to %s", len(docs), self.max_sources)
            docs = docs[: self.max_sources]

        docs = self._filter_useful_docs(query=query, docs=docs)

        docs_for_answer = docs[: self.max_context_docs]
        context = format_docs(docs_for_answer)
        if retrieval_queries:
            context = (
                "Search queries used:\n"
                f"{self._format_query_list(retrieval_queries)}\n\n"
                + context
            )

        label_map = {
            "legal_qa": "Legal Answer",
            "section_finder": "Section Finder",
            "procedure_checklist": "Procedure Checklist",
            "draft_helper": "Draft Helper",
            "compare_provisions": "Provision Comparison",
            "risk_flagger": "Risk Review",
        }
        prefix = f"{label_map.get(decision.intent, 'Legal Answer')}\n\n"
        streamed_answer = ""
        chunk_count = 0

        yield {"type": "token", "content": prefix}

        for chunk in self.fallback_answer_chain.stream(
            {
                "chat_history": history,
                "context": context,
                "query": query,
            }
        ):
            text = self._chunk_to_text(chunk)
            if not text:
                continue
            chunk_count += 1
            streamed_answer += text
            if chunk_count == 1:
                logger.info("First stream chunk emitted: request_id=%s", request_id)
            elif chunk_count % 50 == 0:
                logger.debug(
                    "Stream progress: request_id=%s chunks=%s chars=%s",
                    request_id,
                    chunk_count,
                    len(streamed_answer),
                )
            yield {"type": "token", "content": text}

        final_answer_text = streamed_answer.strip() or "Not found in provided context."
        answer = FinalAnswer(
            answer=final_answer_text,
            cited_sections=[],
            tool_used=decision.intent,
            risk_flags=[],
            escalation_notice="",
            scope_warning="",
        )

        if scope_warning_flag:
            answer.scope_warning = (
                f"No strong evidence was found within selected scope '{act}'. "
                "Results may include broader acts."
            )

        if any(
            marker in query.lower()
            for marker in ["arrest", "bail", "domestic violence", "custody", "criminal", "firing", "termination"]
        ):
            answer.escalation_notice = "Consult a qualified lawyer before taking legal action."

        full_text = self._compose_final_text(answer)
        logger.info(
            "Stream finished: request_id=%s chunks=%s sources=%s elapsed=%.2fs",
            request_id,
            chunk_count,
            len(docs),
            time.perf_counter() - stream_start,
        )

        yield {
            "type": "done",
            "content": full_text.strip(),
            "sources": docs,
        }


# ---------- Chain Builder ----------

def build_chain(
    answer_llm,
    source_filter_llm,
    vectorstore: OpenSearchVectorSearch,
    answer_parser: PydanticOutputParser,
    query_parser: PydanticOutputParser,
    *,
    similarity_k: int = 12,
    max_context_docs: int = 16,
    max_sources: int = 24,
    hybrid_k: int = 20,
    hybrid_fallback_threshold: int = 6,
    enrichment_confidence_threshold: float = 0.5,
):
    return RetrievalLegalChain(
        answer_llm=answer_llm,
        source_filter_llm=source_filter_llm,
        vectorstore=vectorstore,
        answer_parser=answer_parser,
        query_parser=query_parser,
        similarity_k=similarity_k,
        max_context_docs=max_context_docs,
        max_sources=max_sources,
        hybrid_k=hybrid_k,
        hybrid_fallback_threshold=hybrid_fallback_threshold,
        enrichment_confidence_threshold=enrichment_confidence_threshold,
    )
