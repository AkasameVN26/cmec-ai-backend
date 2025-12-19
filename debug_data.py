import os
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

id_ho_so = 38

print(f"--- Checking data for ID: {id_ho_so} ---")

# Check Lich Kham
res_lich = supabase.table("lich_kham").select("*").eq("id_ho_so", id_ho_so).execute()
print(f"Lich Kham Count: {len(res_lich.data)}")
if res_lich.data:
    print(f"First Lich Kham Reason: {res_lich.data[0].get('ly_do_kham')}")

# Check Ghi Chu Y Te
res_notes = supabase.table("ghi_chu_y_te").select("*").eq("id_ho_so", id_ho_so).execute()
print(f"Total Notes: {len(res_notes.data)}")

# Check Note Types to see send_to_ai status
if res_notes.data:
    type_ids = list(set([n['id_loai_ghi_chu'] for n in res_notes.data]))
    res_types = supabase.table("loai_ghi_chu").select("*").in_("id_loai_ghi_chu", type_ids).execute()
    print("Note Types found:")
    for t in res_types.data:
        print(f"ID: {t['id_loai_ghi_chu']}, Name: {t['ten_loai_ghi_chu']}, Send AI: {t['send_to_ai']}")
