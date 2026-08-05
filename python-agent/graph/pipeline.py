import os
import time
import random
from typing import TypedDict
from datetime import datetime, timezone
from dotenv import load_dotenv

# LangGraph imports
from langgraph.graph import StateGraph, END

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

# Tool Import
from langchain_community.tools import DuckDuckGoSearchRun

# Shared central logger module path to prevent loop crashes
from graph.logger_store import active_agent_logs

load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",  # best free model
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0
)


# ── Global run_id for logging ────────────────────────────────────
_current_run_id = None

def set_run_id(run_id: str):
    global _current_run_id
    _current_run_id = run_id

def log(message: str, level: str = "info"):
    global _current_run_id
    if _current_run_id:
        try:
            # Agar storage me yeh runId nahi hai, toh init karo
            if _current_run_id not in active_agent_logs:
                active_agent_logs[_current_run_id] = []
                
            active_agent_logs[_current_run_id].append({
                "time": datetime.now(timezone.utc).isoformat(),
                "message": message,
                "level": level
            })
        except Exception as e:
            print(f"[Logger Error Context]: {str(e)}")
            
    print(f"[Pipeline] {message}")

# ── Safe DuckDuckGo Search with Retry ────────────────────────────
def safe_duckduckgo_search(query: str, max_retries: int = 3) -> str:
    """DuckDuckGo search with rate-limit retry and backoff"""
    for attempt in range(max_retries):
        try:
            # Random delay to avoid rate limiting
            time.sleep(random.uniform(1, 3))
            result = DuckDuckGoSearchRun().run(query)
            return result
        except Exception as e:
            err_str = str(e)
            if "Ratelimit" in err_str or "202" in err_str:
                wait_time = (attempt + 1) * 5
                log(f"[Search] Rate limited (attempt {attempt+1}/{max_retries}). Retrying in {wait_time}s...", "warn")
                time.sleep(wait_time)
            else:
                log(f"[Search] Error: {err_str}", "error")
                if attempt == max_retries - 1:
                    return ""
                time.sleep(2)
    return ""

# ── State Schema Definition ──────────────────────────────────────
class AgentState(TypedDict):
    competitor_name: str
    competitor_url:  str
    blog_rss_url:    str
    scraped_data:    str
    news_data:       str
    relevance_score: float
    clean_data:      str
    analysis:        str
    summary:         str
    source:          str
    user_id:         str  


# ── Node 1: Scraper ──────────────────────────────────────────────
def scraper_node(state: AgentState) -> dict:
    from tools.scraper import scrape_website
    log(f"[Node: Scraper] Starting for {state['competitor_name']}")
    data = scrape_website(state["competitor_name"], state["competitor_url"])
    if "SCRAPE_ERROR" in data or "DYNAMIC_SCRAPE_ERROR" in data:
        log(f"[Node: Scraper] Failed — {data[:100]}", "warn")
    else:
        log(f"[Node: Scraper] Got {len(data)} chars of content", "info")
    return {"scraped_data": data}


# ── Node 2: News ─────────────────────────────────────────────────
def news_node(state: AgentState) -> dict:
    from tools.news import fetch_rss, fetch_blog_rss, fetch_newsapi
    log(f"[Node: News] Fetching RSS for {state['competitor_name']}")
    
    # Primary: Google News RSS
    news = fetch_rss(state["competitor_name"])
    
    # Supplement: NewsAPI (if RSS is thin or empty)
    newsapi_data = fetch_newsapi(state["competitor_name"])
    if newsapi_data:
        if news:
            news = news + "\n\nNEWSAPI_ARTICLES:\n" + newsapi_data
        else:
            news = newsapi_data
        log(f"[Node: News] NewsAPI supplement added", "info")
    
    article_count = news.count("TITLE:")
    log(f"[Node: News] Found {article_count} articles total", "info")
    
    blog = ""
    if state.get("blog_rss_url"):
        log(f"[Node: News] Fetching blog RSS...", "info")
        blog = fetch_blog_rss(state["blog_rss_url"])
    return {"news_data": news + "\n\nBLOG UPDATES:\n" + blog}


# ── Node 3: CRAG Grader ──────────────────────────────────────────
def crag_grader_node(state: AgentState) -> dict:
    log(f"[Node: CRAG Grader] Evaluating data quality...", "info")
    
    # Strip error messages before grading so they don't poison the score
    scraped = state["scraped_data"]
    news = state["news_data"]
    
    clean_scraped = "" if "ERROR" in scraped else scraped
    clean_news = "" if "ERROR" in news else news
    
    combined = clean_scraped + "\n\n" + clean_news
    
    # If absolutely no usable data, skip LLM call and force fallback
    if not combined.strip():
        log(f"[Node: CRAG Grader] No usable data found — forcing fallback", "error")
        return {
            "relevance_score": 0.3,
            "clean_data": state["scraped_data"] + "\n\n" + state["news_data"],
            "source": "scraped"
        }

    prompt = f"""You are a data quality evaluator for competitive intelligence.

Competitor being tracked: {state['competitor_name']}

Retrieved data (first 2000 chars):
{combined[:2000]}

Rate this data quality from 0.0 to 1.0 based on relevant rules.
Respond with ONLY a decimal number between 0.0 and 1.0. Nothing else."""

    try:
        response = llm.invoke([
            HumanMessage(content=prompt)
        ])
        score_text = response.content.strip()
        score = float(score_text)
        score = max(0.0, min(1.0, score))
    except Exception as e:
        log(f"[Node: CRAG Grader Warning] Parse failed ({str(e)}), setting default score.", "warn")
        score = 0.4

    level = "info" if score >= 0.8 else "warn" if score >= 0.5 else "error"
    status_label = "✅ Good" if score >= 0.8 else "⚠️ Ambiguous" if score >= 0.5 else "❌ Bad — triggering fallback"
    log(f"[Node: CRAG Grader] Score: {score:.2f} — {status_label}", level)

    return {
        "relevance_score": score,
        "clean_data": combined,
        "source": "scraped"
    }


# ── CRAG Router ──────────────────────────────────────────────────
def crag_router(state: AgentState) -> str:
    score = state["relevance_score"]
    if score >= 0.8:
        log(f"[CRAG Router] Direct to analyzer (good data)", "info")
        return "analyzer"
    elif score >= 0.5:
        log(f"[CRAG Router] Merging with web search (ambiguous)", "warn")
        return "merge"
    else:
        log(f"[CRAG Router] Full web search fallback (bad data)", "warn")
        return "web_search"


# ── Node 4a: Web Search Fallback ─────────────────────────────────
def web_search_node(state: AgentState) -> dict:
    current_year = datetime.now().year
    log(f"[Node: Web Search] Query: '{state['competitor_name']} latest news update {current_year}'", "info")
    result = safe_duckduckgo_search(f"{state['competitor_name']} latest news update {current_year}")
    
    if not result:
        log(f"[Node: Web Search] Search failed — keeping existing news data", "warn")
        # Don't overwrite clean_data with empty string; preserve whatever we had
        return {"clean_data": state["clean_data"], "source": "web_search_failed"}
    
    log(f"[Node: Web Search] Got {len(result)} chars from DuckDuckGo", "info")
    return {"clean_data": result, "source": "web_search"}


# ── Node 4b: Merge ───────────────────────────────────────────────
def merge_node(state: AgentState) -> dict:
    log(f"[Node: Merge] Supplementing scraped data with web search", "warn")
    web = safe_duckduckgo_search(f"{state['competitor_name']} latest news")
    
    if not web:
        log(f"[Node: Merge] Search failed — using existing data only", "warn")
        return {"clean_data": state["clean_data"], "source": "merge_failed"}
    
    log(f"[Node: Merge] Merged {len(state['clean_data'])} + {len(web)} chars", "info")
    merged = state["clean_data"] + "\n\n=== WEB SEARCH SUPPLEMENT ===\n" + web
    return {"clean_data": merged, "source": "merged"}


# ── Node 5: Analyzer ─────────────────────────────────────────────
def analyzer_node(state: AgentState) -> dict:
    from db.mongo_client import get_last_report
    log(f"[Node: Analyzer] Comparing with previous report...", "info")
    old_data = get_last_report(state["competitor_name"], state["user_id"])
    has_prev = "No previous" not in old_data
    log(f"[Node: Analyzer] Previous report: {'found' if has_prev else 'not found (first run)'}", "info")

    prompt = f"""You are a competitive intelligence analyst.
Competitor: {state['competitor_name']}
New data block matches template profiles. Extract strategic vectors or shifts.
Data: {state['clean_data'][:2500]}"""

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        analysis_content = response.content
    except Exception as e:
        analysis_content = f"Error during analysis generation: {str(e)}"
        log(f"[Node: Analyzer Error] {analysis_content}", "error")

    log(f"[Node: Analyzer] Analysis complete — {len(analysis_content)} chars", "info")
    return {"analysis": analysis_content}


# ── Node 6: Summarizer ───────────────────────────────────────────
def summarizer_node(state: AgentState) -> dict:
    log(f"[Node: Summarizer] Generating executive summary...", "info")

    prompt = f"""You are a sharp competitive intelligence analyst writing for busy startup founders.

Based on this analysis of {state['competitor_name']}, write 3 to 6 bullet points.

Analysis:
{state['analysis']}

STRICT RULES:
- Minimum 3, Maximum 6 bullet points
- Each bullet MUST feel like a breaking news headline — sharp, specific, intriguing
- Format: • [Strong action verb] [specific what] — [why it matters to the reader]
- Maximum 20 words per bullet
- Plain text only — NO markdown, NO asterisks, NO **, NO bold
- Start each bullet with •
- Use numbers/percentages where available (e.g. "raises ₹200Cr", "cuts prices 15%")
- End with something that makes the reader want to know MORE (hint at impact, don't fully explain)
- If nothing changed: • No significant moves detected — market position unchanged this cycle

GOOD EXAMPLE:
- Launches dark store network in 12 cities — signals aggressive quick-commerce pivot
- Cuts delivery fees by 30% — threatens Swiggy's price-sensitive customer base
- Onboards 500 cloud kitchen partners — quietly building parallel revenue stream

BAD EXAMPLE (do not do this):
- Company is doing well and expanding operations in various markets globally

Output ONLY the bullet points. Nothing else."""

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        summary_content = response.content
    except Exception as e:
        summary_content = "• No significant changes detected this cycle"
        log(f"[Node: Summarizer Error] {str(e)}", "error")

    log(f"[Node: Summarizer] Summary ready", "info")
    return {"summary": summary_content}


# ── Node 7: Store ────────────────────────────────────────────────
def store_node(state: AgentState) -> dict:
    from db.mongo_client import save_report
    log(f"[Node: Store] Saving report to MongoDB...", "info")
    save_report({
        "competitor":      state["competitor_name"],
        "summary":         state["summary"],
        "analysis":        state["analysis"],
        "relevance_score": state["relevance_score"],
        "source":          state.get("source", "scraped"),
        "url":             state["competitor_url"],
        "user_id":         state.get("user_id", ""),
        "timestamp":       datetime.now(timezone.utc)
    })
    log(f"[Node: Store] Report saved successfully", "info")
    return state


# ── Build Graph ──────────────────────────────────────────────────
def build_graph():
    g = StateGraph(AgentState)

    g.add_node("scraper",    scraper_node)
    g.add_node("news",       news_node)
    g.add_node("crag_grade", crag_grader_node)
    g.add_node("web_search", web_search_node)
    g.add_node("merge",      merge_node)
    g.add_node("analyzer",   analyzer_node)
    g.add_node("summarizer", summarizer_node)
    g.add_node("store",      store_node)

    g.set_entry_point("scraper")
    g.add_edge("scraper",    "news")
    g.add_edge("news",       "crag_grade")

    g.add_conditional_edges(
        "crag_grade",
        crag_router,
        {
            "analyzer":   "analyzer",
            "web_search": "web_search",
            "merge":      "merge"
        }
    )

    g.add_edge("web_search", "analyzer")
    g.add_edge("merge",      "analyzer")
    g.add_edge("analyzer",   "summarizer")
    g.add_edge("summarizer", "store")
    g.add_edge("store",      END)

    return g.compile()


# ── Run with logs ────────────────────────────────────────────────
def run_for_competitor_with_logs(competitor: dict, run_id: str) -> dict:
    set_run_id(run_id)
    log(f"[Scheduler] Starting run for {competitor['name']}", "info")
    graph = build_graph()
    result = graph.invoke({
        "competitor_name": competitor["name"],
        "competitor_url":  competitor["url"],
        "blog_rss_url":    competitor.get("blog_rss_url", ""),
        "user_id":         competitor.get("user_id", ""),
        "scraped_data":    "",
        "news_data":       "",
        "relevance_score": 0.0,
        "clean_data":      "",
        "analysis":        "",
        "summary":         "",
        "source":          ""
    })
    log(f"[Scheduler] Complete for {competitor['name']}", "info")
    return result


# ── Old function for backward compat ────────────────────────────
def run_for_competitor(competitor: dict) -> dict:
    return run_for_competitor_with_logs(competitor, "local-run")