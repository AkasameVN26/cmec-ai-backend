import httpx
import asyncio
import json

async def test_summarize_38():
    record_id = 38
    url = f"http://127.0.0.1:8000/api/summarize-stream/{record_id}"
    
    print(f"--- Đang bắt đầu tóm tắt bệnh án {record_id} ---")
    print(f"URL: {url}")
    print("-" * 50)

    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            async with client.stream("GET", url) as response:
                if response.status_code != 200:
                    print(f"Lỗi API: {response.status_code}")
                    return

                async for chunk in response.aiter_text():
                    if chunk:
                        print(chunk, end="", flush=True)
                
                print("\n" + "-" * 50)
                print("--- Hoàn thành tóm tắt ---")
    except Exception as e:
        print(f"\nLỗi kết nối: {e}")
        print("Đảm bảo backend FastAPI đang chạy trên cổng 8000.")

if __name__ == "__main__":
    asyncio.run(test_summarize_38())

