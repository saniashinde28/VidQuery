#Actionable items, decisions, questions
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter

from dotenv import load_dotenv
load_dotenv()

def get_llm():
    return init_chat_model("groq:openai/gpt-oss-120b",temperature=0.2)

def build_chain(system_prompt:str):
    llm=get_llm()
    return (RunnablePassthrough() | RunnableLambda(lambda x:{"text":x}) | ChatPromptTemplate.from_messages([
        ("system",system_prompt),
        ("human","{text}")
    ]) | llm |StrOutputParser())

def split_transcript(transcript:str)->list:
    splitter=RecursiveCharacterTextSplitter(
        chunk_size=3000,
        chunk_overlap=200
    )

    chunks=splitter.split_text(transcript)
    return chunks

def extract_action_items(transcript:str)->str:
    chain=build_chain("You are an expert meeting analyst. From the meeting transcript, "
        "extract all action items. For each provide:\n"
        "- Task description\n"
        "- Owner (who is responsible)\n"
        "- Deadline (if mentioned, else write 'Not specified')\n\n"
        "Format as a numbered list. If none found say 'No action items found.'")
    chunks=split_transcript(transcript)
    chunk_extr=[chain.invoke(chunk) for chunk in chunks]
    combined="\n\n".join(chunk_extr)
    return combined
    
def extract_key_decisions(transcript: str) -> str:
    chain = build_chain(
        "You are an expert meeting analyst. From the meeting transcript, "
        "extract all key decisions made. Format as a numbered list. "
        "If none found say 'No key decisions found.'"
    )
    chunks=split_transcript(transcript)
    chunk_extr=[chain.invoke(chunk) for chunk in chunks]
    combined="\n\n".join(chunk_extr)
    return combined


def extract_questions(transcript: str) -> str:
    chain = build_chain(
        "From the meeting transcript, extract all unresolved questions "
        "or topics needing follow-up. Format as a numbered list. "
        "If none found say 'No open questions found.'"
    )
    chunks=split_transcript(transcript)
    chunk_extr=[chain.invoke(chunk) for chunk in chunks]
    combined="\n\n".join(chunk_extr)
    return combined

