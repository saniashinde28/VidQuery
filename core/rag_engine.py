import os
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda,RunnablePassthrough
from core.vector_store import build_vector_store,load_vector_store,get_retriver

def get_llm():
    return init_chat_model("groq:openai/gpt-oss-120b",temperature=0.3)

def format_docs(docs):
    return "\n\n".join([doc.page_content] for doc in docs)

def build_rag_chain(transcript:str):

    vector_store=build_vector_store(transcript)
    retriever=get_retriver(vector_store)

    llm=get_llm()

    prompt=ChatPromptTemplate.from_messages([
        ("system","""You are an expert meeting assistant. Answer the user's question 
based ONLY on the meeting transcript context provided below.

If the answer is not found in the context, say: 
"I could not find this information in the meeting transcript."

Always be concise and precise. If quoting someone, mention it clearly.

Context from meeting transcript:
{context}"""),
("human","{question}")
    ])

    rag_chain=(
        {"context":retriever | RunnableLambda(format_docs),
         "question":RunnablePassthrough()}
         | prompt | llm | StrOutputParser()
    )

    return rag_chain

def load_rag_chain():
    vector_store=load_vector_store()

    retriever=get_retriver(vector_store)

    llm=get_llm()
    
    prompt=ChatPromptTemplate.from_messages([
            ("system","""You are an expert meeting assistant. Answer the user's question 
    based ONLY on the meeting transcript context provided below.
    
    If the answer is not found in the context, say: 
    "I could not find this information in the meeting transcript."
    
    Always be concise and precise. If quoting someone, mention it clearly.
    
    Context from meeting transcript:
    {context}"""),
    ("human","{question}")
        ])
    
    rag_chain=(
        {"context":retriever | RunnableLambda(format_docs),
        "question":RunnablePassthrough()}
             | prompt | llm | StrOutputParser()
        )
    
    return rag_chain

def ask_questions(rag_chain,question:str)->str:
    print("Question :",{question})
    answer=rag_chain.invoke(question)
    print("answer :" ,{answer})
    return answer