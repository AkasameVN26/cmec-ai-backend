from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from app.api.endpoints import router
from app.services.metric_service import metric_service
import asyncio
import sys

if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load models on startup
    print("--- STARTUP: Server starting... Loading models... ---")
    metric_service.load_models() 
    print("--- STARTUP: Models Loaded ---")
    yield
    # Clean up on shutdown (if needed)
    print("--- SHUTDOWN ---")

app = FastAPI(title="CMEC AI Backend", lifespan=lifespan)

# Add CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

app.include_router(router, prefix="/api")

@app.get("/")
def read_root():
    return {"message": "CMEC AI Backend is running. Use /api/prepare-input/{id_ho_so} to aggregate data or /api/summarize/{id_ho_so} to get an AI summary."}