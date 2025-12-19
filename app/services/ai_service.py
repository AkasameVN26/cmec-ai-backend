import httpx
import time
import json
from typing import AsyncGenerator
from fastapi import HTTPException
from app.core.config import settings
from app.models.schemas import SummaryResponse, EvaluationMetrics
from app.services.metric_service import metric_service

SYSTEM_PROMPT = """Bạn là trợ lý y tế chuyên tóm tắt bệnh án chính xác và trung thực tuyệt đối. 
Nhiệm vụ: Tóm tắt bệnh án dựa hoàn toàn vào văn bản được cung cấp.
Quy tắc bắt buộc:
- Chỉ sử dụng thông tin CÓ TRONG VĂN BẢN. Tuyệt đối KHÔNG sử dụng kiến thức bên ngoài hay tự suy luận. Nếu thông tin bị thiếu, hãy bỏ qua.
- Trích xuất tối đa các chi tiết lâm sàng có giá trị NẾU CHÚNG CÓ TRONG VĂN BẢN. BỎ QUA CÁC DẤU HIỆU BÌNH THƯỜNG.
- Chỉ trình bày bản tóm tắt **ngắn gọn**, không đưa ra lời khuyên hay nhận xét cá nhân."""

BASE_INSTRUCTION = """CẤU TRÚC TÓM TẮT:
1. Thông tin chung: 
2. Lý do nhập viện: 
3. Tiền căn: (tóm tắt tiền sử gia đình, xã hội)
4. Tóm tắt bệnh sử & Diễn biến bệnh: 
5. Khám Lâm sàng: (tóm tắt các dấu hiệu bất thường)
6. Kết luận của các kết quả Cận lâm sàng: 
7. Chẩn đoán & Kế hoạch điều trị: 

Dựa vào văn bản dưới đây, hãy hoàn thành bản tóm tắt:"""

OPTIONS = {
    "repeat_penalty": 1.05,
    "temperature": 0.4,
    "top_k": 40,
    "min_p": 0.05,
    "top_p": 1.0,
    "seed": 0
}

class AIService:
    async def summarize_stream(self, prompt_content: str) -> AsyncGenerator[str, None]:
        full_user_content = BASE_INSTRUCTION + prompt_content
        
        ollama_payload = {
            "model": settings.OLLAMA_MODEL, 
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": full_user_content}
            ],
            "stream": True,
            "options": OPTIONS
        }
        
        headers = {"Content-Type": "application/json"}
        print(f"[AIService] Starting stream for model: {settings.OLLAMA_MODEL}")
        
        try:
            async with httpx.AsyncClient(timeout=300.0) as client:
                async with client.stream(
                    "POST",
                    f"{settings.OLLAMA_BASE_URL}/api/chat", 
                    json=ollama_payload, 
                    headers=headers
                ) as response:
                    if response.status_code != 200:
                        error_text = await response.aread()
                        print(f"[AIService] Ollama Error {response.status_code}: {error_text}")
                        yield f"Lỗi từ AI Server: {response.status_code}"
                        return

                    async for line in response.aiter_lines():
                        if line:
                            try:
                                json_data = json.loads(line)
                                content_chunk = json_data.get("message", {}).get("content", "")
                                if content_chunk:
                                    yield content_chunk
                            except json.JSONDecodeError:
                                print(f"[AIService] JSON Decode Error: {line}")
                                continue
        except httpx.ConnectError:
             print("[AIService] Connect Error: Could not connect to Ollama")
             yield "Lỗi: Không thể kết nối đến Ollama server."
        except Exception as e:
            print(f"[AIService] Exception: {str(e)}")
            yield f"Lỗi: {str(e)}"

    async def summarize(self, id_ho_so: int, prompt_content: str, evaluate: bool = False) -> SummaryResponse:
        full_user_content = BASE_INSTRUCTION + prompt_content
        
        ollama_payload = {
            "model": settings.OLLAMA_MODEL, 
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": full_user_content}
            ],
            "stream": False,
            "options": OPTIONS
        }
        
        headers = {"Content-Type": "application/json"}
        
        print(f"[AIService] Sending request to Ollama ({settings.OLLAMA_MODEL})...")
        start_time = time.time()
        
        try:
            async with httpx.AsyncClient(timeout=300.0) as client:
                response = await client.post(
                    f"{settings.OLLAMA_BASE_URL}/api/chat", 
                    json=ollama_payload, 
                    headers=headers
                )
                response.raise_for_status()
                result_json = response.json()

            summary_text = result_json.get("message", {}).get("content", "Không thể tóm tắt.")
            
            ollama_duration = time.time() - start_time
            print(f"[AIService] Ollama Generation took: {ollama_duration:.2f}s")

            metrics = None
            if evaluate:
                # Note: Metric calculation is CPU/GPU intensive and synchronous.
                # Ideally, this should run in a thread pool executor if it blocks too long,
                # but for now we keep it simple to return in the same response.
                metrics_data = await metric_service.calculate_metrics(prompt_content, summary_text, ollama_duration)
                metrics = EvaluationMetrics(**metrics_data)

            return SummaryResponse(
                id_ho_so=id_ho_so,
                source=full_user_content,
                summary=summary_text,
                metrics=metrics
            )

        except httpx.ConnectError:
             raise HTTPException(status_code=503, detail=f"Không thể kết nối đến Ollama server tại {settings.OLLAMA_BASE_URL}. Vui lòng đảm bảo Ollama đang chạy.")
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=500, detail=f"Lỗi từ Ollama API: {e.response.text}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

ai_service = AIService()
