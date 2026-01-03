import requests
import json
import re
import numpy as np
import pandas as pd
import streamlit as st
import time
from datetime import datetime
from dateutil.relativedelta import relativedelta
from dateutil.relativedelta import relativedelta
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import concurrent.futures

# --------------------------
# CONFIG
# --------------------------
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
TWITTER_API_KEY = st.secrets["TWITTER_API_KEY"]

GEMINI_MODEL = "models/gemini-2.5-flash"
GEMINI_EMBED_MODEL = "models/text-embedding-004"
POLY_API_URL = "https://gamma-api.polymarket.com/events"


TOP_EVENT_COUNT = 3
TWEET_MAX_PAGES = 5  # Increased to fetch more data
TWEET_MAX_COUNT = 100 # Increased cap


# ======================================================
# 0️⃣ HELPER: SCALAR DOT PRODUCT (Replaces torch.cosine_similarity)
# ======================================================
def cosine_similarity(v1, v2):
    # v1: (dim,)
    # v2: (N, dim) or (dim,)
    dot = np.dot(v2, v1)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2, axis=len(v2.shape)-1)
    return dot / (norm_v1 * norm_v2)

def get_gemini_embeddings(texts):
    if isinstance(texts, str):
        texts = [texts]
        
    # Batch limit is usually 100, so we chunk strictly
    BATCH_SIZE = 50 
    all_embeddings = []
    
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i+BATCH_SIZE]
        payload = {
            "requests": [
                {
                    "model": GEMINI_EMBED_MODEL,
                    "taskType": "RETRIEVAL_QUERY", 
                    "content": {"parts": [{"text": t}]}
                } 
                for t in batch
            ]
        }
        
        try:
            r = requests.post(
                f"https://generativelanguage.googleapis.com/v1beta/{GEMINI_EMBED_MODEL}:batchEmbedContents",
                params={"key": GEMINI_API_KEY},
                json=payload
            )
            if r.status_code != 200:
                st.error(f"Embedding API Error: {r.text}")
                return []
            
            # Extract 'values' from list of objects
            batch_embeds = [e["values"] for e in r.json().get("embeddings", [])]
            all_embeddings.extend(batch_embeds)
        except Exception as e:
            st.error(f"Embedding failed: {e}")
            return []
            
    return np.array(all_embeddings)


# ======================================================
# 1️⃣ FETCH POLYMARKET EVENTS
# ======================================================
def fetch_events():
    # We want to fetch ~3000 events to ensure we cover almost everything active.
    # API limit is 500 per page. So we need 6 pages (offsets 0, 500, ... 2500).
    offsets = [0, 500, 1000, 1500, 2000, 2500]
    all_events = []
    
    def fetch_page(offset):
        try:
            r = requests.get(
                POLY_API_URL, 
                params={"closed": "false", "active": "true", "limit": 500, "offset": offset}
            )
            if r.status_code == 200:
                return r.json()
        except:
            pass
        return []

    # Parallel Execution
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
        results = list(executor.map(fetch_page, offsets))
        
    for page in results:
        all_events.extend(page)
        
    if not all_events:
        st.error("Couldn't reach the prediction markets.")
        return []

    cleaned = []
    for ev in all_events:
        active = []
        for m in ev.get("markets", []):
            if m.get("active") and not m.get("closed"):
                # Safety check for missing keys
                raw_prices = m.get("outcomePrices")
                raw_outcomes = m.get("outcomes")
                
                if not raw_prices or not raw_outcomes:
                    continue
                    
                prices = json.loads(raw_prices) if isinstance(raw_prices, str) else raw_prices
                outcomes = json.loads(raw_outcomes) if isinstance(raw_outcomes, str) else raw_outcomes

                if outcomes and prices:
                    m["formatted_odds"] = " | ".join(
                        [f"{o}: ${float(p):.4f}" for o, p in zip(outcomes, prices)]
                    )
                    active.append(m)

        if active:
            ev["markets"] = active
            cleaned.append(ev)

    return cleaned


# ======================================================
# 2️⃣ FIND TOP MATCHING EVENTS (Similarity + Volume)
# ======================================================
def find_relevant_events(user_text, events):
    if not events:
        return []

    # 1. KEYWORD FILTERING (Strict Priority)
    # This prevents "Taylor Swift" query from showing "Hailey Bieber" markets
    keywords = [w.lower() for w in user_text.split() if len(w) > 2]
    keyword_matches = []
    
    for ev in events:
        title_lower = ev["title"].lower()
        # Check if ANY keyword is in the title
        if any(k in title_lower for k in keywords):
            # Calculate a simple "keyword score" (number of matches)
            match_count = sum(1 for k in keywords if k in title_lower)
            ev["match_count"] = match_count
            ev["total_volume"] = sum(float(m.get("volume", 0)) for m in ev.get("markets", []))
            # Set similarity to 1.0 (or high) since it's a direct text match
            ev["similarity"] = 0.99 
            keyword_matches.append(ev)
            
    if keyword_matches:
        # Sort by: 1. Number of keyword matches (desc), 2. Volume (desc)
        keyword_matches.sort(key=lambda x: (x["match_count"], x["total_volume"]), reverse=True)
        return keyword_matches[:TOP_EVENT_COUNT]

    # 2. FALLBACK: SEMANTIC SEARCH (If no text match found)
    # Get user embedding
    user_vec = get_gemini_embeddings(user_text)
    if len(user_vec) == 0:
        return []
    user_vec = user_vec[0] # Take first item

    # Get event embeddings (batch)
    titles = [e["title"] for e in events]
    title_vecs = get_gemini_embeddings(titles)
    
    if len(title_vecs) != len(titles):
        st.error("Embedding mismatch size")
        return []

    # Manual Cosine Similarity using Numpy
    scores = cosine_similarity(user_vec, title_vecs)

    for i, ev in enumerate(events):
        ev["similarity"] = float(scores[i])
        ev["total_volume"] = sum(float(m.get("volume", 0)) for m in ev.get("markets", []))

    sorted_events = sorted(
        events,
        key=lambda x: (x["similarity"], x["total_volume"]),
        reverse=True
    )

    return sorted_events[:TOP_EVENT_COUNT]


# ======================================================
# 2.5️⃣ HELPER: RENDER ODDS VISUAL
# ======================================================
def render_odds_visual(market):
    try:
        raw_outcomes = market.get("outcomes", "[]")
        raw_prices = market.get("outcomePrices", "[]")
        
        outcomes = json.loads(raw_outcomes) if isinstance(raw_outcomes, str) else raw_outcomes
        prices = json.loads(raw_prices) if isinstance(raw_prices, str) else raw_prices
        
        st.markdown(f"**{market.get('question', 'Market')}**")
        
        # Check for Binary Yes/No
        lower_outcomes = [str(o).lower() for o in outcomes]
        if set(lower_outcomes) == {"yes", "no"} and len(outcomes) == 2:
            # It's a binary market
            yes_idx = lower_outcomes.index("yes")
            yes_price = float(prices[yes_idx])
            no_idx = lower_outcomes.index("no")
            no_price = float(prices[no_idx])
            
            # Single Bar: Progress represents YES probability
            # Label: "Yes: 65% | No: 35%"
            st.progress(yes_price, text=f"Yes: {yes_price:.1%}  |  No: {no_price:.1%}")
            
        else:
            # Standard Multi-Choice Render
            for out, price in zip(outcomes, prices):
                p_val = float(price)
                st.progress(p_val, text=f"{out}: {p_val:.1%}")
            
    except Exception as e:
        st.error(f"Visual render error: {e}")


# ======================================================
# 3️⃣ GEMINI: GENERATE KEYWORDS FROM MARKETS
# ======================================================
def get_keywords_from_market(ev):
    questions = " | ".join([m["question"] for m in ev["markets"]])
    prompt = f"""
Extract 6–10 short, high-signal Twitter search keywords.
Return ONLY JSON array like ["keyword1","keyword2"].
Text: {questions}
"""
    r = requests.post(
        f"https://generativelanguage.googleapis.com/v1/{GEMINI_MODEL}:generateContent",
        params={"key": GEMINI_API_KEY},
        json={"contents": [{"parts": [{"text": prompt}]}]}
    )

    try:
        txt = r.json()["candidates"][0]["content"]["parts"][0]["text"]
        return json.loads(txt)
    except:
        return []


# ======================================================
# 4️⃣ HELPER: BUILD BETTER TWITTER QUERY
# ======================================================
def build_query(event_title):
    title = event_title.lower()
    title = re.sub(r'[^a-z0-9\s]', '', title)
    words = [w for w in title.split() if len(w) > 3]

    # OR join title words
    q = " OR ".join(words)

    # Add fallback finance / prediction keywords
    q += " OR crypto OR stocks OR election OR prediction OR odds OR betting"

    return q


# ======================================================
# 5️⃣ FETCH TWEETS (CAPPED & IMPROVED)
# ======================================================
# ======================================================
# 5️⃣ FETCH TWEETS (CAPPED & IMPROVED)
# ======================================================
# ======================================================
# 5️⃣ FETCH TWEETS (CAPPED & IMPROVED)
# ======================================================
# ======================================================
# 5️⃣ FETCH TWEETS (Refactored for Events OR Topics)
# ======================================================
def fetch_tweets(source, is_event=True, max_pages=TWEET_MAX_PAGES, max_tweets=TWEET_MAX_COUNT):
    # Determine Query
    if is_event:
        # STRATEGY 1: Smart Keywords from Gemini (for Events)
        keywords = get_keywords_from_market(source)
        if keywords:
            query = " OR ".join(keywords)
        else:
            query = build_query(source["title"])
        
        fallback_query = build_query(source["title"])
    else:
        # STRATEGY: Raw Topic Search
        # We append some general finance terms to ensure we get relevant context, 
        # but keep it broad enough for the main topic.
        clean_topic = re.sub(r'[^a-zA-Z0-9 ]', '', source)
        query = f'"{clean_topic}" OR {clean_topic}' 
        # Fallback is just the topic itself
        fallback_query = clean_topic

    st.write(f"Searching: `{query}`")

    def run_search(q):
        found = []
        cursor = ""
        pages = 0
        while pages < max_pages and len(found) < max_tweets:
            try:
                r = requests.get(
                    "https://api.twitterapi.io/twitter/tweet/advanced_search",
                    headers={"X-API-Key": TWITTER_API_KEY},
                    params={"query": q, "queryType": "Top", "cursor": cursor}
                )
                if r.status_code != 200:
                    st.error(f"API Error: {r.status_code} - {r.text}")
                    break
                
                data = r.json()
                batch = data.get("tweets", [])
                if not batch: break
                
                found.extend(batch)
                if not data.get("has_next_page"): break
                
                cursor = data.get("next_cursor", "")
                pages += 1
            except Exception as e:
                st.error(f"Request failed: {str(e)}")
                break
        return found

    tweets = run_search(query)

    # Retry Strategy
    if not tweets and query != fallback_query:
        st.warning(f"Too specific. Widening the search net...")
        st.write(f"Searching: `{fallback_query}`")
        tweets = run_search(fallback_query)

    st.write(f"Conversations found: {len(tweets)}")
    return tweets[:max_tweets]


# ======================================================
# 6️⃣ GEMINI SENTIMENT
# ======================================================
def analyze_with_gemini(question, tweets):
    if not tweets:
        return None

    text = "\n\n".join(
        [f"User: {t.get('author', {}).get('userName', t.get('user', 'unknown'))}\nLikes:{t.get('likeCount',0)}\n{t['text']}" for t in tweets]
    )

    prompt = f"""
Market: "{question}"

Give sentiment ONLY from tweets:
1) Public Sentiment
2) Does it imply YES or NO?
3) Rough probability (0-100%)
Be concise. Explain like I'm 5. No jargon. NO EMOJIS.
Tweets:
{text}
"""

    r = requests.post(
        f"https://generativelanguage.googleapis.com/v1/{GEMINI_MODEL}:generateContent",
        params={"key": GEMINI_API_KEY},
        json={"contents": [{"parts": [{"text": prompt}]}]}
    )

    try:
        return r.json()["candidates"][0]["content"]["parts"][0]["text"]
    except:
        return "Gemini failed"


# ======================================================
# 7️⃣ GEMINI: MASTER SUMMARY
# ======================================================
def generate_comparison(topic, general_analysis, market_insights, avg_score, confidence):
    """
    Generates a direct comparison between Public Sentiment and Market Odds.
    """
    market_text = ""
    for m in market_insights:
        market_text += f"- Market: {m['title']} | Odds: {m['odds']} | Sentiment: {m['sentiment']}\n"
        
    prompt = f"""
    Analyze the alignment between Public Sentiment (Twitter) and Prediction Markets (Polymarket) for '{topic}'.
    
    Data:
    [Public Sentiment]
    {general_analysis}
    Quantitative: Score {avg_score:.2f}, Confidence {confidence:.0%}
    
    [Prediction Markets]
    {market_text}
    
    Compare what the public is saying vs. how people are betting.
    Explain the gap between the vibe (Twitter) and the money (Polymarket).
    Write like a smart friend. Short sentences. NO EMOJIS.
    Constraint: You only have CURRENT market odds. Do NOT hallucinate trends/shifts.
    
    Format:
    ### Where they agree
    (Where do tweets and betting odds align?)
    
    ### Where they clash
    (Where is the public irrational vs the market? Or where is the market missing a signal?)
    """

    r = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}",
        headers={"Content-Type": "application/json"},
        json={"contents": [{"parts": [{"text": prompt}]}]}
    )
    
    if r.status_code != 200:
        return "Error generating comparison."
        
    return r.json()['candidates'][0]['content']['parts'][0]['text']


def generate_deep_insights(topic, narrative, comparison, history_summary, market_insights):
    """
    Generates a final 'Deep Insight' section looking for non-obvious patterns.
    """
    market_text = "\n".join([f"- {m['title']}: {m['odds']}" for m in market_insights])
    
    prompt = f"""
    Act as a smart, neutral analyst. You have just analyzed '{topic}' across social media (Twitter), prediction markets (Polymarket), and historical trends.
    
    [Full Context]
    1. Narrative: {narrative}
    2. Comparison (Public vs Market): {comparison}
    3. Historical Trend: {history_summary}
    4. Market Odds: {market_text}
    
    TASK:
    Using all the data above, generate a concise but high-value insights section.
    1. Identify the most significant divergences between public sentiment and market probabilities.
    2. Explain possible reasons.
    3. Highlight notable agreements.
    4. Tell me what really matters—what does this imply about future events?
    5. Avoid simply repeating odds or sentiment scores; synthesize the information into narrative insights.
    
    Format:
    ### Key Takeaways
    (1. Most Important Insight - Explain the divergence/agreement and WHY it matters. Keep it brief.)
    (2. ...)
    (3. ...)
    
    ### What this means
    (What does this state of perception vs reality imply for the future? Be concise.)

    CRITICAL: NO EMOJIS.
    Constraint: You only have CURRENT market odds. Do NOT hallucinate that they have "shifted" or "repriced" recently unless you have historical proof.
    STYLE GUIDE:
    - Be extremely concise.
    - Use short, punchy sentences.
    - No fluff.
    - Get straight to the point.
    """
    
    r = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}",
        headers={"Content-Type": "application/json"},
        json={"contents": [{"parts": [{"text": prompt}]}]}
    )
    
    if r.status_code != 200:
        return "Error generating insights."
        
    return r.json()['candidates'][0]['content']['parts'][0]['text']


# ======================================================
# 7.4️⃣ GEMINI: CROSS-PLATFORM COMPARISON
# ======================================================
def generate_cross_platform_comparison(topic, twitter_data, reddit_data):
    """
    Compares the narratives from Twitter and Reddit.
    """
    # Prepare raw text samples (top 50)
    tw_text = "\n".join([f"- {t['text']}" for t in twitter_data.get('data', [])[:50]])
    rd_text = "\n".join([f"- {p['text']}" for p in reddit_data.get('data', [])[:50]])

    prompt = f"""
    Compare the public sentiment and discussion around '{topic}' on two different platforms:
    
    [PLATFORM A: X]
    Narrative: {twitter_data['narrative']}
    Sentiment: {twitter_data['sentiment']:.2f} (Confidence: {twitter_data['confidence']:.0%})
    Raw Samples:
    {tw_text}
    
    [PLATFORM B: REDDIT]
    Narrative: {reddit_data['narrative']}
    Sentiment: {reddit_data['sentiment']:.2f} (Confidence: {reddit_data['confidence']:.0%})
    Raw Samples:
    {rd_text}
    
    Task:
    Compare the vibe on X vs Reddit.
    
    MAX 3 bullet points per section. Be extremely concise.
    
    Format:
    ### Agreements
    - (Point 1)
    - (Point 2)
    
    ### Divergences
    - (Point 1)
    - (Point 2)
    
    ### Who is surer?
    (Compare certainty levels. Does high confidence map to reality? Keep it to 1-2 sentences.)
    
    CRITICAL: NO EMOJIS.
    """
    
    r = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}",
        headers={"Content-Type": "application/json"},
        json={"contents": [{"parts": [{"text": prompt}]}]}
    )
    
    if r.status_code != 200:
        return "Error generating comparison."
        
    return r.json()['candidates'][0]['content']['parts'][0]['text']


# ======================================================
# 7.5️⃣ GEMINI: GENERAL TOPIC ANALYSIS (Qualitative)
# ======================================================
def analyze_general_topic(topic, tweets):
    if not tweets:
        return "The internet is silent on this."

    text = "\n".join([t['text'] for t in tweets[:50]]) # Analyze top 50 for speed

    prompt = f"""
    Topic: "{topic}"
    
    Recent Tweets:
    {text}
    
    TASK:
    Provide a summary in 2 distinct sections separated by "|||".
    CRITICAL: DO NOT USE EMOJIS. Use simple, direct language. ELI5. No jargon.
    
    Format:
    What's happening
    1. Current Status (1-2 sentences): Include price/level if applicable, describe what is happening NOW.
    2. Key Ongoing Events: List 3-5 concrete events being discussed.
    (Focus on RECENT developments only. Be concise, factual, and time-anchored.)
    |||
    What people are saying
    (3-4 short bullet points summarizing main narratives found in the provided tweets)
    """

    r = requests.post(
        f"https://generativelanguage.googleapis.com/v1/{GEMINI_MODEL}:generateContent",
        params={"key": GEMINI_API_KEY},
        json={"contents": [{"parts": [{"text": prompt}]}]}
    )

    try:
        return r.json()["candidates"][0]["content"]["parts"][0]["text"]
    except:
        return "Gemini analysis failed."


# ======================================================
# 7.6️⃣ VADER: QUANTITATIVE METRICS
# ======================================================
def calculate_sentiment_metrics(tweets):
    if not tweets:
        return 0.0, 0.0
        
    analyzer = SentimentIntensityAnalyzer()
    compound_scores = []
    
    for t in tweets:
        score = analyzer.polarity_scores(t['text'])
        compound_scores.append(score['compound'])
        
    avg_score = np.mean(compound_scores) # -1 to 1 (Polarity)
    confidence = np.mean([abs(s) for s in compound_scores]) # 0 to 1 (Strength/Confidence)
    
    return avg_score, confidence


# ======================================================
# 7.7️⃣ HISTORICAL TIME SERIES (12 Months)
# ======================================================
def fetch_historical_sentiment(topic, current_confidence=None, current_sentiment=None):
    history_data = []
    today = datetime.utcnow()
    
    # Progress UI
    progress_bar = st.progress(0)
    status_text = st.empty()
    status_text.text("⏳ Fetching 12 months history in parallel...")
    
    def fetch_month(i):
        # Calculate month window
        date_cursor = today - relativedelta(months=i)
        start_date = date_cursor.replace(day=1)
        # End date is start of next month
        end_date = start_date + relativedelta(months=1)
        
        start_str = start_date.strftime("%Y-%m-%d_00:00:00_UTC")
        end_str = end_date.strftime("%Y-%m-%d_00:00:00_UTC")
        label = start_date.strftime("%b %Y")
        
        # Optimization: Use current data for current month (i=0)
        if i == 0 and current_confidence is not None and current_sentiment is not None:
             return {"Month": label, "Date": start_date, "Confidence": current_confidence, "Sentiment": current_sentiment}
        
        # embed dates in query
        clean_topic = re.sub(r'[^a-zA-Z0-9 ]', '', topic)
        query = f'"{clean_topic}" since:{start_str} until:{end_str}'
        
        try:
            r = requests.get(
                "https://api.twitterapi.io/twitter/tweet/advanced_search",
                headers={"X-API-Key": TWITTER_API_KEY},
                params={"query": query, "queryType": "Top"} 
            )
            
            if r.status_code == 200:
                data = r.json()
                tweets = data.get("tweets", [])
                if tweets:
                    avg, conf = calculate_sentiment_metrics(tweets)
                    return {"Month": label, "Date": start_date, "Confidence": conf, "Sentiment": avg}
            
            return {"Month": label, "Date": start_date, "Confidence": 0.0, "Sentiment": 0.0}
                 
        except Exception:
             return {"Month": label, "Date": start_date, "Confidence": 0.0, "Sentiment": 0.0}

    # Parallel Execution
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        # Submit all tasks
        futures = {executor.submit(fetch_month, i): i for i in range(11, -1, -1)}
        
        completed_count = 0
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                history_data.append(result)
            except Exception as e:
                st.error(f"History thread failed: {e}")
            
            completed_count += 1
            progress_bar.progress(completed_count / 12)

    progress_bar.empty()
    status_text.empty()
    
    # Sort by date since parallel execution returns out of order
    df = pd.DataFrame(history_data)
    if not df.empty:
        df = df.sort_values("Date")
        
    return df


# ======================================================
# 7.8️⃣ GEMINI: HISTORY ANALYSIS
# ======================================================
def analyze_history_with_gemini(df, topic):
    if df.empty:
        return "Not enough history to show a trend."
        
    # Convert data to a simple string for the LLM
    data_str = df.to_csv(index=False)
    
    prompt = f"""
    TOPIC: "{topic}"
    HISTORICAL DATA (Last 12 Months):
    {data_str}
    
    TASK:
    Analyze the trends in "Sentiment" (Polarity: -1 Neg to +1 Pos) and "Confidence" (Strength: 0 to 1).
    - Identify any major shifts or anomalies.
    - Explain what the relationship between Sentiment and Confidence likely means.
    - Keep it concise (2-3 sentences max).
    - Explain like I'm 5. Avoid technical jargon.
    - CRITICAL: DO NOT USE EMOJIS.
    """

    r = requests.post(
        f"https://generativelanguage.googleapis.com/v1/{GEMINI_MODEL}:generateContent",
        params={"key": GEMINI_API_KEY},
        json={"contents": [{"parts": [{"text": prompt}]}]}
    )

    try:
        return r.json()["candidates"][0]["content"]["parts"][0]["text"]
    except:
        return "Gemini historical analysis failed."


# ======================================================
# 7.9️⃣ REDDIT FETCHING
# ======================================================
import xml.etree.ElementTree as ET

# ======================================================
# 7.9️⃣ REDDIT FETCHING (RSS/ATOM BACKEND)
# ======================================================
def fetch_reddit(query, limit=50):
    # Strategy: Use RSS Feed (Atom) - extremely robust against bot detection
    url = f"https://www.reddit.com/search.rss?q={query}&sort=relevance&t=month"
    
    # Minimal headers are still good practice
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
    }

    try:
        r = requests.get(url, headers=headers)
        if r.status_code != 200:
            st.error(f"Reddit RSS Error: {r.status_code}")
            return []
        
        # Parse Atom Feed
        posts = []
        root = ET.fromstring(r.content)
        
        # Namespace map for Atom
        ns = {'atom': 'http://www.w3.org/2005/Atom'}
        
        count = 0
        for entry in root.findall('atom:entry', ns):
            if count >= limit: break
            
            title = entry.find('atom:title', ns).text or ""
            # Content is technically HTML in RSS, but we can grab it. 
            # Sometimes it's just the title that matters for broad scanning.
            # Using title as the main text source is often safer/cleaner for sentiment.
            
            author_tag = entry.find('atom:author', ns)
            author = author_tag.find('atom:name', ns).text if author_tag is not None else "u/unknown"
            
            link = entry.find('atom:link', ns).attrib.get("href", "")
            
            posts.append({
                "text": title, # Main signal
                "user": author,
                "likes": 0, # RSS doesn't give live likes, default to 0
                "date": "Recent",
                "url": link
            })
            count += 1
            
        return posts

    except Exception as e:
        st.error(f"Reddit Parse Error: {e}")
        return []

# ======================================================
# 8️⃣ REUSABLE ANALYSIS FUNCTION
# ======================================================
def analyze_topic_data(topic, source_type):
    """
    Fetches data and performs Section 1 analysis for a given source.
    Returns a dict with results or None if failed.
    """
    if source_type == "X":
        fetch_func = lambda t: fetch_tweets(t, is_event=False)
        label = "X"
    else:
        fetch_func = fetch_reddit
        label = "Reddit"

    data = fetch_func(topic)
    
    if not data:
        return None

    # Analyze
    gen_analysis = analyze_general_topic(topic, data)
    avg_score, confidence = calculate_sentiment_metrics(data)
    
    # Parse Gemini Output
    parts = gen_analysis.split("|||")
    if len(parts) == 2:
        narrative = parts[0].strip()
        discussion = parts[1].strip()
    else:
        narrative = gen_analysis
        discussion = ""
        
    return {
        "source": label,
        "data": data,
        "narrative": narrative,
        "discussion": discussion,
        "sentiment": avg_score,
        "confidence": confidence,
        "count": len(data)
    }


# ======================================================
# 9️⃣ STREAMLIT APP LAYOUT
# ======================================================
st.set_page_config(layout="wide") # Use wide mode for specific multi-column layouts

# NAVIGATION STATE
if 'page' not in st.session_state:
    st.session_state['page'] = 'home'

def go_home():
    st.session_state['page'] = 'home'

def go_single():
    st.session_state['page'] = 'single'

def go_compare():
    st.session_state['page'] = 'compare'

# LANDING PAGE
if st.session_state['page'] == 'home':
    st.title("Sentiment vs. Reality")
    st.markdown("""
    See what the internet feels vs. where the money is betting. We compare social noise with market odds to show you the difference between opinion and reality.

    Select a path to begin:
    """)
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("### Deep Dive")
        st.write("Analyze one topic to see the gap between social noise and real betting odds.")
        st.caption("_Example: Enter 'Bitcoin' to see if Twitter is bullish while the market bets on a crash._")
        if st.button("Start Deep Dive", use_container_width=True):
            go_single()
            st.rerun()

    with col2:
        st.warning("### X vs. Reddit")
        st.write("Compare the vibe across different platforms.")
        st.caption("_Example: see if Reddit is panicking about 'Inflation' while X is ignoring it._")
        if st.button("Compare Platforms", use_container_width=True):
            go_compare()
            st.rerun()

else:
    # COMMON HEADER FOR SUB-PAGES
    c1, c2 = st.columns([1, 5])
    with c1:
        if st.button("← Back"):
            go_home()
            st.rerun()
    with c2:
        st.title("Sentiment vs. Reality")

    user_text = st.text_input("What topic are you curious about?", placeholder="e.g. Bitcoin, Election, Inflation")

    if st.button("Reveal the Truth") and user_text.strip():

        # ======================================================
        # PAGE 1: SINGLE SOURCE ANALYSIS
        # ======================================================
        if st.session_state['page'] == 'single':
            # Force X for deep analysis as per user request
            data_source = "X"
            
            with st.status(f"Reading the internet...", expanded=True) as status:
                result = analyze_topic_data(user_text, data_source)
                if not result:
                    st.error(f"Couldn't hear any chatter about {data_source}.")
                    st.stop()
                status.update(label="Done reading.", state="complete", expanded=False)

            # RENDER SECTION 1
            st.header("The Vibe")
            col1, col2, col3 = st.columns(3)
            col1.metric("Vibe Score", f"{result['sentiment']:.2f}")
            col2.metric("Conviction", f"{result['confidence']:.0%}")
            col3.metric("Chatter Volume", f"{result['count']}")
            
            st.subheader("What they're saying")
            st.write(result['narrative'])
            st.subheader("Key themes")
            st.markdown(result['discussion'])
            
            # HISTORICAL
            st.subheader("How the feeling changed (12mo)")
            # Always X
            df_history = fetch_historical_sentiment(user_text, result['confidence'], result['sentiment'])
                 
            if not df_history.empty:
                df_history.set_index("Date", inplace=True)
                st.line_chart(df_history[["Sentiment", "Confidence"]])
                
            st.divider()

            # RENDER SECTION 2 (Legacy Logic maintained for Single View)
            st.header("Where The Money Is")
            with st.spinner("Checking prediction markets..."):
                events = fetch_events()
                final_events = find_relevant_events(user_text, events)

            market_insights = []
            if final_events:
                st.write(f"Found {len(final_events)} relevant markets.")
                
                def process_market(ev):
                    try:
                        # Always fetch X for single view
                        tweets = fetch_tweets(ev, is_event=True)
                        
                        sentiment = analyze_with_gemini(ev["title"], tweets)
                        odds_str = " | ".join([m['formatted_odds'] for m in ev['markets']])
                        return {"event": ev, "title": ev['title'], "odds": odds_str, "sentiment": sentiment}
                    except Exception as e:
                        return {"event": ev, "title": ev['title'], "odds": "Error", "sentiment": f"Analysis failed: {e}"}

                with st.spinner("Crunching the numbers..."):
                    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                        processed_markets = list(executor.map(process_market, final_events))

                for pm in processed_markets:
                    market_insights.append(pm)
                    with st.expander(f"{pm['title']}", expanded=False):
                         for m in pm['event']["markets"]:
                            render_odds_visual(m)
                         st.code(pm['sentiment'], language=None)
            
            # Deep Insights
            st.header("The Reality Check")
            with st.spinner("Connecting the dots..."):
                comp_text = generate_comparison(user_text, result['narrative'], market_insights, result['sentiment'], result['confidence'])
                # We skip history summary generation for speed in this refactor, or we can add it back if critical
                deep = generate_deep_insights(user_text, result['narrative'], comp_text, "History skipped", market_insights)
                st.markdown(deep)


        # ======================================================
        # PAGE 2: SOURCE COMPARISON
        # ======================================================
        if st.session_state['page'] == 'compare':
            st.write("## X vs. Reddit: Who's Right?")
            
            col_twitter, col_reddit = st.columns(2)
        
            # Parallel Execution for both sources
            with st.status("Listening to both sides...", expanded=True):
                with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                    future_tw = executor.submit(analyze_topic_data, user_text, "X")
                    future_rd = executor.submit(analyze_topic_data, user_text, "Reddit")
                    
                    res_tw = future_tw.result()
                    res_rd = future_rd.result()
            
            # LEFT COLUMN: X
            with col_twitter:
                st.markdown("### X")
                if res_tw:
                    # Split metrics
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Vibe", f"{res_tw['sentiment']:.2f}", delta_color="normal", help="-1 to 1")
                    m2.metric("Conviction", f"{res_tw['confidence']:.0%}", help="How strong the opinions are")
                    m3.metric("Volume", f"{res_tw['count']}")
                    
                    st.info(res_tw['discussion'])
                else:
                    st.error("Crickets on X.")
                    
            # RIGHT COLUMN: REDDIT
            with col_reddit:
                st.markdown("### Reddit")
                if res_rd:
                    # Split metrics
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Vibe", f"{res_rd['sentiment']:.2f}", delta_color="normal", help="-1 to 1")
                    m2.metric("Conviction", f"{res_rd['confidence']:.0%}", help="How strong the opinions are")
                    m3.metric("Volume", f"{res_rd['count']}")
                    
                    st.info(res_rd['discussion'])
                else:
                    st.error("Radio silence on Reddit.")
    
            st.divider()
            
            # CROSS ANALYSIS
            if res_tw and res_rd:
                st.write("## The Verdict")
                with st.spinner("Reading between the lines..."):
                    comparison = generate_cross_platform_comparison(user_text, res_tw, res_rd)
                    st.markdown(comparison)
