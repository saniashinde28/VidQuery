
import os

from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

from core.vector_store import (
    build_vector_store,
    load_vector_store
)

from core.semantic_cache import SemanticCache


TOP_K = int(os.getenv("TOP_K", "4"))

FALLBACK_ANSWER = (
    "I could not find this information in the meeting transcript."
)


def get_llm():
    return init_chat_model(
        "groq:openai/gpt-oss-120b",
        temperature=0.3
    )


def format_docs(docs):
    return "\n\n".join(
        f"[Chunk {doc.metadata.get('chunk_index')}]\n{doc.page_content}"
        for doc in docs
    )


def create_rag_chain(vector_store):

    llm = get_llm()

    # Fresh cache for this recording
    semantic_cache = SemanticCache(
        embedding_model=vector_store.embeddings,
        similarity_threshold=0.90
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert video transcript assistant.

Answer the user's question based ONLY on the transcript context
provided below.

Important instructions:

1. Answer only when the context contains sufficient evidence
   to answer the specific question.

2. Do not assume that related information answers the question.

3. If the context discusses the topic but does not contain
   the information needed to answer the question, say exactly:

   "I could not find this information in the meeting transcript."

4. Never use your general knowledge to fill in missing information.

5. Be concise and precise.

6. Cite the supporting chunk numbers in your answer,
   for example [Chunk 2].

7. Only cite chunks that actually support the answer.

Transcript context:
{context}
"""),

        ("human", "{question}")
    ])

    answer_chain = (
        prompt
        | llm
        | StrOutputParser()
    )

    def answer_with_sources(question):

        print("\n--- RAG Query ---")
        print("Question:", question)

        # STEP 1: Check semantic cache
        cached_result = semantic_cache.get(question)

        if cached_result is not None:

            print("Returning cached result.")

            return cached_result

        # STEP 2: Cache miss -> retrieve relevant chunks
        docs = vector_store.similarity_search(
            question,
            k=TOP_K
        )

        # STEP 3: Handle empty retrieval
        if not docs:

            return {
                "answer": FALLBACK_ANSWER,
                "sources": []
            }

        # STEP 4: Format context
        context = format_docs(docs)

        # STEP 5: Generate answer
        answer = answer_chain.invoke({
            "context": context,
            "question": question
        })

        # STEP 6: Prepare sources
        sources = [
            {
                "chunk_index": doc.metadata.get("chunk_index"),
                "start_index": doc.metadata.get("start_index"),
                "content": doc.page_content
            }
            for doc in docs
        ]

        result = {
            "answer": answer,
            "sources": sources
        }

        # STEP 7: Save successful result in cache
        semantic_cache.add(question, result)

        return result

    return RunnableLambda(answer_with_sources)


def build_rag_chain(transcript: str):

    vector_store = build_vector_store(transcript)

    return create_rag_chain(vector_store)


def load_rag_chain():

    vector_store = load_vector_store()

    return create_rag_chain(vector_store)


def ask_questions(rag_chain, question: str):

    print("Question:", question)

    result = rag_chain.invoke(question)

    print("Answer:", result["answer"])

    return result
