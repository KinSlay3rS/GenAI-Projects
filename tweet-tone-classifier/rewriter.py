from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model='gemini-2.5-flash',
    temperature=0.7
    )


TONE_PROMPTS = {
    "formal":     "You are an expert writer. Rewrite the given tweet in a professional, formal tone. Keep it under 280 characters. Return only the rewritten tweet, nothing else.",
    "casual":     "You are an expert writer. Rewrite the given tweet in a friendly, casual tone. Keep it under 280 characters. Return only the rewritten tweet, nothing else.",
    "empathetic": "You are an expert writer. Rewrite the given tweet with warmth and empathy. Keep it under 280 characters. Return only the rewritten tweet, nothing else.",
    "assertive":  "You are an expert writer. Rewrite the given tweet in a confident, assertive tone. Keep it under 280 characters. Return only the rewritten tweet, nothing else.",
}

def rewrite_tweet(tweet: str, tone: str) -> str:
    prompt = ChatPromptTemplate.from_messages([
        ("system", TONE_PROMPTS[tone]),
        ("human", "Tweet: {tweet}")
    ])

    chain = prompt | llm

    response = chain.invoke({"tweet": tweet})
    return response.content.strip()