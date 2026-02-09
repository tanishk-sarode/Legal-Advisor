from langchain_classic.chains.query_constructor.schema import AttributeInfo
from langchain_classic.retrievers import SelfQueryRetriever
from langchain_community.query_constructors.opensearch import OpenSearchTranslator


def build_retriever(vectorstore, llm):
    document_contents = (
        "Text of Indian legal provisions from the Constitution and statutory acts."
    )
    metadata_field_info = [
        AttributeInfo(
            name="act_abbrev",
            description="Act abbreviation, e.g., IPC, CrPC, CPC, MVA, IEA, HMA, IDA, NIA, COI.",
            type="string",
        ),
        AttributeInfo(
            name="source_type",
            description="Type of source: article, section, or clause.",
            type="string",
        ),
        AttributeInfo(
            name="article_id",
            description="Article identifier (e.g., 21, 14A).",
            type="string",
        ),
        AttributeInfo(
            name="section_id",
            description="Section identifier (e.g., 279, 304A).",
            type="string",
        ),
        AttributeInfo(
            name="citation",
            description="Citation string for display (e.g., Section 279 (IPC)).",
            type="string",
        ),
        AttributeInfo(
            name="chapter",
            description="Chapter number if present.",
            type="string",
        ),
        AttributeInfo(
            name="chapter_title",
            description="Chapter title if present.",
            type="string",
        ),
    ]

    return SelfQueryRetriever.from_llm(
        llm=llm,
        vectorstore=vectorstore,
        document_contents=document_contents,
        metadata_field_info=metadata_field_info,
        structured_query_translator=OpenSearchTranslator(),
        search_kwargs={"k": 12},
    )
