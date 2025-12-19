from fastapi import APIRouter, HTTPException, Depends, Query
from fastapi.responses import StreamingResponse
from app.services.supabase_service import supabase_service, SupabaseService
from app.services.ai_service import ai_service, AIService
from app.services.metric_service import metric_service
from app.models.schemas import PreparedInput, SummaryResponse, ExplainRequest, ExplainResponse

router = APIRouter()

@router.post("/explain/{id_ho_so}", response_model=ExplainResponse)
async def explain_summary_endpoint(id_ho_so: int, request: ExplainRequest):
    try:
        # Step 1: Get original source data
        prepared_data = await supabase_service.prepare_input(id_ho_so)
        # source_text = prepared_data.prompt_content # No longer needed
        source_segments = prepared_data.source_segments
        
        # Step 2: Calculate explanation
        # Note: top_k defaults to 2 in service, can be made parameterized if needed
        explanation = await metric_service.explain_summary(source_segments, request.summary)
        
        if "error" in explanation:
             raise HTTPException(status_code=500, detail=explanation["error"])
             
        return explanation
        
    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"[Endpoint] Explain Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/prepare-input/{id_ho_so}", response_model=PreparedInput)
async def prepare_input(id_ho_so: int):
    try:
        return await supabase_service.prepare_input(id_ho_so)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/summarize-stream/{id_ho_so}")
async def summarize_medical_record_stream(id_ho_so: int):
    try:
        print(f"[Endpoint] Summarize Stream Request for ID: {id_ho_so}")
        
        # Step 1: Get data from Supabase
        try:
            prepared_data = await supabase_service.prepare_input(id_ho_so)
            print(f"[Endpoint] Data prepared successfully. Prompt length: {len(prepared_data.prompt_content)}")
        except Exception as e:
            print(f"[Endpoint] Error preparing input: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Database Error: {str(e)}")
        
        # Step 2: Call AI Service in streaming mode
        return StreamingResponse(
            ai_service.summarize_stream(prepared_data.prompt_content),
            media_type="text/plain"
        )
        
    except Exception as e:
        print(f"[Endpoint] General Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/summarize/{id_ho_so}", response_model=SummaryResponse)
async def summarize_medical_record(
    id_ho_so: int, 
    measure_metrics: bool = Query(False, description="Tính toán các chỉ số đánh giá (ROUGE, Token, v.v.)")
):
    try:
        # Step 1: Get data from Supabase
        prepared_data = await supabase_service.prepare_input(id_ho_so)
        
        # Step 2: Call AI Service
        return await ai_service.summarize(id_ho_so, prepared_data.prompt_content, evaluate=measure_metrics)
        
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))