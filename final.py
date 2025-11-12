import streamlit as st
import pandas as pd
import requests
import re
import os
import time
import io
import csv
from urllib.parse import urljoin
from bs4 import BeautifulSoup
import plotly.express as px
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache

# -------------------------
# Setup
# -------------------------
load_dotenv()
FACT_API_KEY = os.getenv("GOOGLE_FACT_CHECK_API")
BASE_URL = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
CACHE_TTL = 24 * 60 * 60  # cache for 1 day
DATA_FILE = "politifact_data.csv"

# -------------------------
# Utility: Text Cleaning
# -------------------------
def clean_text(text: str) -> str:
    """Clean input statements for better query matching."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r'[“”"\'.,!?]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text[:300]  # limit for API efficiency

# -------------------------
# Google Fact Check (Cached + Optimized)
# -------------------------
@lru_cache(maxsize=500)
def google_fact_check(statement: str):
    """Query Google Fact Check API for a given statement."""
    if not FACT_API_KEY:
        return {"verdict": "API key missing", "publisher": None, "rating": None, "url": None}

    query = clean_text(statement)
    if not query:
        return {"verdict": "Unverified", "publisher": None, "rating": None, "url": None}

    params = {"query": query, "key": FACT_API_KEY}
    try:
        res = requests.get(BASE_URL, params=params, timeout=8)
        res.raise_for_status()
        data = res.json()
        claims = data.get("claims", [])
        if not claims:
            return {"verdict": "Unverified", "publisher": None, "rating": None, "url": None}

        for claim in claims:
            for review in claim.get("claimReview", []):
                rating = review.get("textualRating", "").lower()
                publisher = review.get("publisher", {}).get("name", "Unknown")
                url = review.get("url", "")
                if any(x in rating for x in ["false", "pants", "incorrect", "misleading"]):
                    return {"verdict": "False", "publisher": publisher, "rating": rating, "url": url}
                if any(x in rating for x in ["true", "accurate", "correct", "mostly true"]):
                    return {"verdict": "True", "publisher": publisher, "rating": rating, "url": url}
        return {"verdict": "Unverified", "publisher": None, "rating": None, "url": None}
    except requests.RequestException as e:
        return {"verdict": "API Error", "publisher": None, "rating": str(e), "url": None}

# -------------------------
# PolitiFact Scraper
# -------------------------
def fetch_politifact_claims(start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> pd.DataFrame:
    """Scrape PolitiFact claims between two dates."""
    base_url = "https://www.politifact.com/factchecks/list/"
    current_url = base_url
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["author", "statement", "source", "date", "label"])

    scraped_rows_count = 0
    page_count = 0
    status_slot = st.empty()
    session = requests.Session()

    while current_url and page_count < 50:
        page_count += 1
        status_slot.text(f"Fetching page {page_count}... ({scraped_rows_count} rows so far)")
        try:
            response = session.get(current_url, timeout=15)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "lxml")
        except requests.exceptions.RequestException as e:
            st.warning(f"Network Error: {e}")
            break

        rows_to_add = []
        for card in soup.find_all("li", class_="o-listicle__item"):
            date_div = card.find("div", class_="m-statement__desc")
            date_text = date_div.get_text(strip=True) if date_div else None
            claim_date = None
            if date_text:
                match = re.search(r"stated on ([A-Za-z]+\s+\d{1,2},\s+\d{4})", date_text)
                if match:
                    try:
                        claim_date = pd.to_datetime(match.group(1), format='%B %d, %Y')
                    except ValueError:
                        continue

            if claim_date:
                if start_ts <= claim_date <= end_ts:
                    statement_block = card.find("div", class_="m-statement__quote")
                    statement = statement_block.find("a", href=True).get_text(strip=True) if statement_block else None
                    source_a = card.find("a", class_="m-statement__name")
                    source = source_a.get_text(strip=True) if source_a else None
                    footer = card.find("footer", class_="m-statement__footer")
                    author = None
                    if footer:
                        author_match = re.search(r"By\s+([^•]+)", footer.get_text(strip=True))
                        if author_match:
                            author = author_match.group(1).strip()

                    label_img = card.find("img", alt=True)
                    label = label_img['alt'].replace('-', ' ').title() if label_img else None
                    rows_to_add.append([author, statement, source, claim_date.strftime('%Y-%m-%d'), label])
                elif claim_date < start_ts:
                    current_url = None
                    break

        if not rows_to_add:
            break

        writer.writerows(rows_to_add)
        scraped_rows_count += len(rows_to_add)
        next_link = soup.find("a", class_="c-button c-button--hollow", string=re.compile(r"Next", re.I))
        current_url = urljoin(base_url, next_link['href']) if next_link else None
        time.sleep(0.5)

    output.seek(0)
    df = pd.read_csv(output)
    df = df.dropna(subset=['statement'])
    df.to_csv(DATA_FILE, index=False)
    return df

# -------------------------
# Verify Statements (FAST)
# -------------------------
def verify_statements(df: pd.DataFrame):
    """Verify all statements using Google Fact Check API concurrently."""
    st.info("Verifying statements via Google Fact Check API... ⚡ Please wait...")
    progress = st.progress(0)
    statements = df["statement"].tolist()
    results = [None] * len(statements)

    def process_statement(idx, stmt):
        if not isinstance(stmt, str) or not stmt.strip():
            return idx, {"verdict": "Invalid", "publisher": None, "rating": None, "url": None}
        return idx, google_fact_check(stmt)

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(process_statement, i, s) for i, s in enumerate(statements)]
        for j, future in enumerate(as_completed(futures)):
            idx, result = future.result()
            results[idx] = result
            progress.progress((j + 1) / len(statements))

    df["google_verdict"] = [r["verdict"] for r in results]
    df["publisher"] = [r["publisher"] for r in results]
    df["google_rating_text"] = [r["rating"] for r in results]
    df["google_source_url"] = [r["url"] for r in results]
    return df

# -------------------------
# Visualization
# -------------------------
def show_summary(df):
    st.markdown("### 📊 Google Verdict Summary")
    summary = df["google_verdict"].value_counts(normalize=True).mul(100).round(2)
    st.write(summary)
    fig = px.bar(
        summary,
        x=summary.index,
        y=summary.values,
        text=summary.values,
        title="Google Fact Check Verdict Distribution (%)",
        labels={"x": "Verdict", "y": "Percentage"},
        color=summary.index,
    )
    st.plotly_chart(fig, use_container_width=True)

# -------------------------
# Streamlit UI
# -------------------------
def run_app():
    st.set_page_config(page_title="PolitiFact + Google Fact Check", layout="wide", page_icon="📰")
    st.title("📰 PolitiFact Scraper + Google Fact Check Verifier (FAST ⚡)")

    st.sidebar.header("Settings")
    start_date = st.sidebar.date_input("Start Date", pd.Timestamp.now() - pd.Timedelta(days=30))
    end_date = st.sidebar.date_input("End Date", pd.Timestamp.now())

    if "df" not in st.session_state:
        st.session_state.df = pd.DataFrame()

    if st.button("🔍 Scrape PolitiFact Data"):
        df = fetch_politifact_claims(pd.to_datetime(start_date), pd.to_datetime(end_date))
        if df.empty:
            st.warning("No data scraped for the selected date range.")
        else:
            st.session_state.df = df
            st.success(f"✅ Scraped {len(df)} statements from PolitiFact.")
            st.dataframe(df.head(), use_container_width=True)
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("⬇ Download Scraped Data (CSV)", csv, "politifact_scraped.csv", "text/csv")

    if not st.session_state.df.empty:
        if st.button("🚀 Verify with Google Fact Check API"):
            verified_df = verify_statements(st.session_state.df)
            st.success("✅ Verification Complete!")
            show_summary(verified_df)
            st.dataframe(verified_df, use_container_width=True)
            st.download_button(
                "⬇ Download Verified Results (CSV)",
                verified_df.to_csv(index=False).encode("utf-8"),
                file_name="verified_results.csv",
                mime="text/csv",
            )

# -------------------------
if __name__ == "__main__":
    run_app()
