import os
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

print("--- Finding id_ho_so with data in both lich_kham and ghi_chu_y_te ---")

try:
    # Get id_ho_so from lich_kham
    lich_kham_ids_res = supabase.table("lich_kham").select("id_ho_so").execute()
    lich_kham_ids = {item["id_ho_so"] for item in lich_kham_ids_res.data}
    print(f"Found {len(lich_kham_ids)} unique id_ho_so in lich_kham.")

    # Get id_ho_so from ghi_chu_y_te
    ghi_chu_ids_res = supabase.table("ghi_chu_y_te").select("id_ho_so").execute()
    ghi_chu_ids = {item["id_ho_so"] for item in ghi_chu_ids_res.data}
    print(f"Found {len(ghi_chu_ids)} unique id_ho_so in ghi_chu_y_te.")

    # Find common ids
    common_ids = list(lich_kham_ids.intersection(ghi_chu_ids))
    
    if common_ids:
        print(f"\nFound {len(common_ids)} common id_ho_so with data in both tables.")
        print("Here are up to 5 examples:")
        for i, doc_id in enumerate(common_ids[:5]):
            print(f"- {doc_id}")
    else:
        print("\nNo common id_ho_so found with data in both lich_kham and ghi_chu_y_te.")
        print("Consider checking if there's any data in these tables.")

except Exception as e:
    print(f"An error occurred: {e}")
