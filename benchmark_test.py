import asyncio
import time
import httpx
import sys
import os
import json
from datetime import datetime

# Configuration
RECORD_ID = 38
BASE_URL = "http://127.0.0.1:8000"
LOG_DIR = "logs"

def setup_logger():
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(LOG_DIR, f"benchmark_{timestamp}.log")
    return log_file

def log_msg(file_path, msg, to_console=True):
    timestamp = datetime.now().strftime("[%H:%M:%S]")
    full_msg = f"{timestamp} {msg}"
    
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(full_msg + "\n")
    
    if to_console:
        print(full_msg)

async def check_server(client):
    """Checks if the backend is reachable."""
    try:
        resp = await client.get(f"{BASE_URL}/docs")
        return resp.status_code == 200
    except:
        return False

async def benchmark_workflow():
    log_file = setup_logger()
    
    log_msg(log_file, f"🚀 Starting Client-Side Benchmark (Record ID: {RECORD_ID})")
    log_msg(log_file, f"Target: {BASE_URL}")
    log_msg(log_file, f"Log File: {log_file}")
    log_msg(log_file, "-" * 60)
    
    async with httpx.AsyncClient(timeout=300.0) as client:
        # 0. Server Health Check
        if not await check_server(client):
            log_msg(log_file, f"❌ Cannot connect to {BASE_URL}")
            log_msg(log_file, "👉 Please run the server first: uvicorn app.main:app --reload")
            return

        # --- PHASE 1: DATA PREPARATION (Supabase Query & Aggregation) ---
        log_msg(log_file, f"1️⃣  Measuring Data Preparation (Supabase Fetch & Aggregate)...")
        start_prep = time.perf_counter()
        prep_time = 0
        try:
            url = f"{BASE_URL}/api/prepare-input/{RECORD_ID}"
            log_msg(log_file, f"   GET {url}", to_console=False)
            
            prep_response = await client.get(url)
            
            if prep_response.status_code != 200:
                log_msg(log_file, f"   ❌ Failed: {prep_response.status_code} - {prep_response.text}")
                return
            
            data_content = prep_response.text
            data_size = len(data_content)
            log_msg(log_file, f"   Response Preview (first 500 chars):\n{data_content[:500]}...", to_console=False)
            
        except Exception as e:
            log_msg(log_file, f"   ❌ Exception: {e}")
            return
        
        end_prep = time.perf_counter()
        prep_time = end_prep - start_prep
        log_msg(log_file, f"   ✅ Done. Data Payload Size: {data_size} chars")
        log_msg(log_file, f"   ⏱️  Time: {prep_time:.4f} s")

        # --- PHASE 2: SUMMARIZATION (AI Generation) ---
        log_msg(log_file, f"\n2️⃣  Measuring AI Summarization (Streaming)...")
        start_summary = time.perf_counter()
        summary_time = 0
        summary_text = ""
        try:
            url = f"{BASE_URL}/api/summarize-stream/{RECORD_ID}"
            log_msg(log_file, f"   GET {url}", to_console=False)
            
            async with client.stream("GET", url) as response:
                if response.status_code != 200:
                    log_msg(log_file, f"   ❌ Failed: {response.status_code}")
                    return

                print(f"[{datetime.now().strftime('%H:%M:%S')}]    Stream receiving", end="", flush=True)
                async for chunk in response.aiter_text():
                    summary_text += chunk
                    # print(".", end="", flush=True) 
                print(" Done.")
                
            log_msg(log_file, f"   Full Generated Summary:\n{summary_text}", to_console=False)
                
        except Exception as e:
            log_msg(log_file, f"   ❌ Exception: {e}")
            return
            
        end_summary = time.perf_counter()
        summary_time = end_summary - start_summary
        
        if not summary_text:
            log_msg(log_file, "   ❌ Empty summary received.")
            return

        log_msg(log_file, f"   ✅ Summary Length: {len(summary_text)} chars")
        log_msg(log_file, f"   ⏱️  Time: {summary_time:.4f} s")

        # --- PHASE 3: EXPLANATION (RAG / Metrics) ---
        log_msg(log_file, f"\n3️⃣  Measuring Explanation (Source Retrieval)...")
        start_explain = time.perf_counter()
        explain_time = 0
        try:
            url = f"{BASE_URL}/api/explain/{RECORD_ID}"
            explain_payload = {"summary": summary_text}
            log_msg(log_file, f"   POST {url}", to_console=False)
            
            explain_response = await client.post(url, json=explain_payload)
            
            if explain_response.status_code != 200:
                log_msg(log_file, f"   ❌ Failed: {explain_response.status_code}")
                return
            
            explanation_data = explain_response.json()
            citations = explanation_data.get("citations", [])
            
            log_msg(log_file, f"   Full Explanation Response:\n{json.dumps(explanation_data, ensure_ascii=False, indent=2)}", to_console=False)
            
        except Exception as e:
            log_msg(log_file, f"   ❌ Exception: {e}")
            return
            
        end_explain = time.perf_counter()
        explain_time = end_explain - start_explain
        log_msg(log_file, f"   ✅ Citations Found: {len(citations)}")
        log_msg(log_file, f"   ⏱️  Time: {explain_time:.4f} s")

    # --- FINAL REPORT ---
    log_msg(log_file, "\n" + "="*60)
    log_msg(log_file, "📊 BENCHMARK RESULTS")
    log_msg(log_file, "="*60)
    log_msg(log_file, f"1. Data Prep (DB) : {prep_time:.4f} s")
    log_msg(log_file, f"2. Summarize (LLM): {summary_time:.4f} s")
    log_msg(log_file, f"3. Explain (RAG)  : {explain_time:.4f} s")
    log_msg(log_file, "-" * 60)
    log_msg(log_file, f"TOTAL TIME        : {prep_time + summary_time + explain_time:.4f} s")
    log_msg(log_file, "="*60)

if __name__ == "__main__":
    asyncio.run(benchmark_workflow())

