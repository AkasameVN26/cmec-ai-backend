import asyncio
import os
import sys
from dotenv import load_dotenv

# Ensure the app module can be found
sys.path.append(os.getcwd())

load_dotenv()

from app.services.supabase_service import supabase_service

async def main():
    # Use a default ID or take from command line
    id_ho_so = 38
    if len(sys.argv) > 1:
        try:
            id_ho_so = int(sys.argv[1])
        except ValueError:
            print("Invalid ID provided, using default 38")

    print(f"--- Generating Prompt for Record ID: {id_ho_so} ---")
    
    try:
        prepared_input = await supabase_service.prepare_input(id_ho_so)
        
        print("\n" + "="*40)
        print("GENERATED PROMPT CONTENT")
        print("="*40 + "\n")
        print(prepared_input.prompt_content)
        print("\n" + "="*40)
        
        print(f"\nTotal Segments: {len(prepared_input.source_segments)}")
        print(f"Raw Notes Count: {prepared_input.raw_notes_count}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
