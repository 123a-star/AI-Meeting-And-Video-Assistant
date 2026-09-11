from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import os


def get_llm():
    return ChatGroq(
        model="openai/gpt-oss-20b",
        groq_api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.2
    )


def extract_action_items(transcript: str) -> str:
    llm = get_llm()

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Extract all action items from the meeting transcript. "
                "Include the task and responsible person if mentioned. "
                "If there are no action items, clearly say so."
            ),
            ("human", "{text}"),
        ]
    )

    chain = prompt | llm | StrOutputParser()

    return chain.invoke({"text": transcript})


def extract_key_decisions(transcript: str) -> str:
    llm = get_llm()

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Extract the key decisions made during the meeting. "
                "Return them as concise bullet points. "
                "Do not invent decisions that are not present in the transcript."
            ),
            ("human", "{text}"),
        ]
    )

    chain = prompt | llm | StrOutputParser()

    return chain.invoke({"text": transcript})


def extract_questions(transcript: str) -> str:
    llm = get_llm()

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Extract all important questions raised during the meeting. "
                "Return them as bullet points. "
                "If no questions were asked, clearly say so."
            ),
            ("human", "{text}"),
        ]
    )

    chain = prompt | llm | StrOutputParser()

    return chain.invoke({"text": transcript})


