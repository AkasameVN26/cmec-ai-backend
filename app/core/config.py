import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    SUPABASE_URL: str = os.getenv("SUPABASE_URL", "")
    SUPABASE_KEY: str = os.getenv("SUPABASE_KEY", "")
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "vyl-medsum-demo")

    def validate(self):
        if not self.SUPABASE_URL or not self.SUPABASE_KEY:
            raise ValueError("Missing SUPABASE_URL or SUPABASE_KEY in .env file")

settings = Settings()
settings.validate()
