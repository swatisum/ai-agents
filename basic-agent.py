"""
AI agentic pipeline to find trending topics & YouTube shorts for a user persona.

User will be prompted for:
- Domain
- Region
- Niche
"""

import json
from typing import List, Dict, Any

from pydantic import BaseModel

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain.tools import tool
from langchain_community.tools import YouTubeSearchTool


# -------------------------------------------------------------------
# Persona model
# -------------------------------------------------------------------

class Persona(BaseModel):
    domain: str
    region: str
    niche: str

    def to_description(self) -> str:
        return (
            f"Domain: {self.domain}; "
            f"Region: {self.region}; "
            f"Niche: {self.niche}."
        )


# -------------------------------------------------------------------
# LLM setup + keyword generator chain
# -------------------------------------------------------------------

def build_llm():
    return ChatOpenAI(
        model="gpt-4.1-mini",
        temperature=0.3,
    )


def build_keyword_chain(llm: ChatOpenAI):
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert content and social media strategist. "
                "You generate trending topics and hashtags tailored to a given persona."
            ),
            (
                "user",
                (
                    "Persona details:\n{persona_description}\n\n"
                    "1. List 5 highly trending topics or keywords for this persona.\n"
                    "2. List 5 relevant hashtags.\n\n"
                    "Return ONLY valid JSON with exactly this structure:\n"
                    "{{\n"
                    '  "topics": [],\n'
                    '  "hashtags": []\n'
                    "}}\n"
                    "Do NOT include any extra text."
                )
            )
        ]
    )

    parser = JsonOutputParser()
    return prompt | llm | parser


# -------------------------------------------------------------------
# YouTube tool
# -------------------------------------------------------------------

_youtube_search = YouTubeSearchTool()


@tool
def get_youtube_trends(keywords: List[str], max_results_per_keyword: int = 5):
    """
    Search YouTube for short-form trending videos for each keyword.
    Returns a list of dictionaries with keyword, query, and video URLs.
    """
    results = []

    for kw in keywords:
        query = f"{kw} recent shorts trending with 5M+ views"
        tool_input = f"{query}, {max_results_per_keyword}"

        raw = _youtube_search.run(tool_input)

        try:
            urls = json.loads(raw)
            if not isinstance(urls, list):
                urls = [urls]
        except Exception:
            urls = [raw]

        results.append(
            {
                "keyword": kw,
                "query": query,
                "urls": urls,
            }
        )

    return results



# -------------------------------------------------------------------
# Main agent pipeline
# -------------------------------------------------------------------

def run_ai_topic_agent(domain, region, niche, max_youtube_results=3):

    persona = Persona(domain=domain, region=region, niche=niche)
    llm = build_llm()
    keyword_chain = build_keyword_chain(llm)

    persona_description = persona.to_description()
    results = keyword_chain.invoke({"persona_description": persona_description})

    topics = results.get("topics", [])
    hashtags = results.get("hashtags", [])

    youtube_data = get_youtube_trends.invoke({
        "keywords": topics,
        "max_results_per_keyword": max_youtube_results,
    })

    return {
        "persona": persona_description,
        "topics": topics,
        "hashtags": hashtags,
        "youtube": youtube_data,
    }


# -------------------------------------------------------------------
# CLI Entry Point
# -------------------------------------------------------------------

if __name__ == "__main__":
    print("\n🧠 AI Trending Topic Finder\n")

    domain = input("Enter Persona Domain (e.g., AI Product Management): ")
    region = input("Enter Region (e.g., North America): ")
    niche = input("Enter Niche (e.g., AI agents for YouTubers): ")

    result = run_ai_topic_agent(domain, region, niche)

    print("\n=== Persona ===")
    print(result["persona"])

    print("\n=== Trending Topics ===")
    for i, t in enumerate(result["topics"], 1):
        print(f"{i}. {t}")

    print("\n=== Hashtags ===")
    print(", ".join(result["hashtags"]))

    print("\n=== YouTube Videos ===")
    for topic_section in result["youtube"]:
        print(f"\nTopic: {topic_section['keyword']}")
        for url in topic_section["urls"]:
            print(" -", url)

    print("\nDone! 🚀")
