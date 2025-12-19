import requests
import time
import json
import os
from datetime import datetime

# Configuration
API_URL = "http://127.0.0.1:8000/api/summarize/38"
PARAMS = {"measure_metrics": "true"}
OUTPUT_DIR = "test_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def main():
    print("Waiting for server logic (if restarting)...")
    time.sleep(2) 

    print(f"Calling API: {API_URL} with params {PARAMS}")
    start_time = time.time()
    
    try:
        response = requests.get(API_URL, params=PARAMS, timeout=300) 
        total_time = time.time() - start_time
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{OUTPUT_DIR}/summary_result_38_{timestamp}.txt"

        if response.status_code == 200:
            data = response.json()
            
            with open(filename, "w", encoding="utf-8") as f:
                f.write("="*80 + "\n")
                f.write(f"TEST RESULT FOR RECORD ID: {data.get('id_ho_so')}\n")
                f.write(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("="*80 + "\n\n")

                # 1. Source (Prompt)
                source = data.get('source', '')
                f.write("-" * 30 + "\n")
                f.write("INPUT SOURCE (PROMPT):\n")
                f.write("-" * 30 + "\n")
                f.write(source + "\n\n")

                # 2. Summary
                summary = data.get('summary', '')
                f.write("-" * 30 + "\n")
                f.write("GENERATED SUMMARY:\n")
                f.write("-" * 30 + "\n")
                f.write(summary + "\n\n")

                # 3. Metrics
                metrics = data.get('metrics')
                f.write("-" * 30 + "\n")
                f.write("EVALUATION METRICS:\n")
                f.write("-" * 30 + "\n")
                
                f.write(f"Total Request Time: {total_time:.2f}s\n")
                if metrics:
                    f.write(f"Generation Duration : {metrics.get('duration', 0):.2f}s\n")
                    f.write(f"Token Counts        : Input={metrics.get('input_tokens')}, Output={metrics.get('output_tokens')}\n")
                    f.write(f"Compression Ratio   : {metrics.get('ratio', 0):.4f}\n")
                    f.write(f"Similarity Score    : {metrics.get('similarity_score', 0):.4f}\n\n")
                    
                    f.write("ROUGE Scores (F1 / Precision / Recall):\n")
                    r1 = metrics.get('rouge1', {})
                    r2 = metrics.get('rouge2', {})
                    rl = metrics.get('rougeL', {})
                    
                    f.write(f"  ROUGE-1: {r1.get('f1',0):.4f} / {r1.get('p',0):.4f} / {r1.get('r',0):.4f}\n")
                    f.write(f"  ROUGE-2: {r2.get('f1',0):.4f} / {r2.get('p',0):.4f} / {r2.get('r',0):.4f}\n")
                    f.write(f"  ROUGE-L: {rl.get('f1',0):.4f} / {rl.get('p',0):.4f} / {rl.get('r',0):.4f}\n")
                else:
                    f.write("[WARNING] No metrics returned in response.\n")
            
            print(f"\n[SUCCESS] Results saved to: {filename}")
            print(f"Summary preview: {summary[:100]}...")

        else:
            print(f"\n[ERROR] Status Code: {response.status_code}")
            print(response.text)

    except requests.exceptions.ConnectionError:
        print("\n[ERROR] Connection refused. Is the FastAPI server running?")
    except requests.exceptions.Timeout:
        print("\n[ERROR] Request timed out.")
    except Exception as e:
        print(f"\n[ERROR] An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
