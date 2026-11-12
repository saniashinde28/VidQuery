from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough,RunnableLambda
import os
from dotenv import load_dotenv
load_dotenv()

def get_llm():
    return init_chat_model("groq:openai/gpt-oss-120b",temperature=0.3)

def split_transcript(transcript:str)->list:
    splitter=RecursiveCharacterTextSplitter(
        chunk_size=3000,
        chunk_overlap=200
    )

    chunks=splitter.split_text(transcript)
    return chunks

def summarize(transcript:str)->str:
    llm=get_llm()
    map_prompt=ChatPromptTemplate.from_messages([
        ("system","Summarize this portion of a meeting transcript concisely"),
        ("human","{text}")
    ])

    map_chain=map_prompt | llm | StrOutputParser()

    chunks=split_transcript(transcript)

    chunk_summaries=[map_chain.invoke({"text":chunk}) for chunk in chunks]

    combined="\n\n".join(chunk_summaries)

    combined_prompt=ChatPromptTemplate.from_messages([
        ("system","You are an expert meeting summarizer, combine these partial summaries into one final professional meeting summary in bullet points"),
        ("human","{text}")
    ])

    combined_chain=(
        RunnablePassthrough() | RunnableLambda(lambda x:{"text":x}) | combined_prompt | llm | StrOutputParser()
    )

    return combined_chain.invoke(combined)

def generate_title(transcript:str)->str:
    llm=get_llm()

    prompt=ChatPromptTemplate.from_messages([
        ("system","based on the meeting transcript, generate a short professional meeting title(max 8 words), only return the title, nothing else"),
        ("human","{text}")
    ])

    title_chain=(
        RunnablePassthrough() | RunnableLambda(lambda x:{"text":x}) | prompt | llm | StrOutputParser()
    )

    return title_chain.invoke(transcript[:2000])