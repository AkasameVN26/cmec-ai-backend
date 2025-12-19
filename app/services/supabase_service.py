from supabase import create_client, Client
from app.core.config import settings
from app.models.schemas import PreparedInput, SourceSegment
from datetime import datetime
import asyncio
from typing import List, Dict, Any

class SupabaseService:
    def __init__(self):
        self.client: Client = create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)

    def _fetch_patient_info(self, id_ho_so: int) -> SourceSegment:
        tuoi_str = "Không rõ"
        gioi_tinh = "Không rõ"
        
        try:
            ho_so_res = self.client.table("ho_so_benh_an") \
                .select("id_benh_nhan, benh_nhan(ngay_sinh, gioi_tinh)") \
                .eq("id_ho_so", id_ho_so) \
                .maybe_single() \
                .execute()
            
            if ho_so_res.data:
                benh_nhan_data = ho_so_res.data.get("benh_nhan", {}) or {}
                ngay_sinh_str = benh_nhan_data.get("ngay_sinh")
                gioi_tinh = benh_nhan_data.get("gioi_tinh", "Không rõ")
                
                if ngay_sinh_str:
                    try:
                        ns = datetime.strptime(ngay_sinh_str, "%Y-%m-%d")
                        tuoi = datetime.now().year - ns.year
                        tuoi_str = str(tuoi)
                    except ValueError:
                        pass
        except Exception as e:
            print(f"[SupabaseService] Error fetching patient info: {e}")
        
        return SourceSegment(
            content=f"- Tuổi: {tuoi_str}.\n- Giới tính: {gioi_tinh}.",
            source_type="THÔNG TIN CHUNG", 
            source_id=id_ho_so
        )

    def _fetch_allergies(self, id_ho_so: int) -> SourceSegment:
        di_ung_res = self.client.table("ghi_chu_y_te") \
            .select("id_ghi_chu, noi_dung_ghi_chu, loai_ghi_chu!inner(ten_loai_ghi_chu)") \
            .eq("id_ho_so", id_ho_so) \
            .eq("loai_ghi_chu.ten_loai_ghi_chu", "Thông tin dị ứng") \
            .execute()
        
        if di_ung_res.data and len(di_ung_res.data) > 0:
            item = di_ung_res.data[0]
            content = item['noi_dung_ghi_chu'].strip()
            if not content.endswith('.'):
                content += '.'
            return SourceSegment(
                content=f"- Dị ứng: {content}",
                source_type="THÔNG TIN CHUNG", 
                source_id=item['id_ghi_chu']
            )
        else:
            return SourceSegment(
                content="- Dị ứng: Bệnh nhân chưa ghi nhận tiền sử dị ứng thuốc.",
                source_type="THÔNG TIN CHUNG",
                source_id=id_ho_so
            )

    def _fetch_visits(self, id_ho_so: int) -> tuple[List[int], List[SourceSegment]]:
        lich_kham_res = self.client.table("lich_kham") \
            .select("id_lich_kham, ly_do_kham, thoi_gian_kham") \
            .eq("id_ho_so", id_ho_so) \
            .order("thoi_gian_tao") \
            .execute()
        
        visit_ids = [item["id_lich_kham"] for item in lich_kham_res.data]
        segments = []
        
        if lich_kham_res.data:
            first_visit = lich_kham_res.data[0]
            ly_do = first_visit.get("ly_do_kham", "")
            
            # Format visit date for header
            visit_date_str = "N/A"
            if first_visit.get("thoi_gian_kham"):
                try:
                    dt = datetime.fromisoformat(first_visit["thoi_gian_kham"].replace('Z', '+00:00'))
                    visit_date_str = dt.strftime("%d/%m/%Y")
                except:
                    pass

            if ly_do:
                ly_do_content = ly_do.strip()
                if not ly_do_content.endswith('.'):
                    ly_do_content += '.'
                segments.append(SourceSegment(
                    content=f"LÝ DO KHÁM: {ly_do_content}",
                    source_type=f"Khám ngày {visit_date_str}",
                    source_id=first_visit['id_lich_kham']
                ))
        
        return visit_ids, segments

    def _fetch_diagnoses(self, id_ho_so: int) -> List[SourceSegment]:
        chan_doan_res = self.client.table("chan_doan") \
            .select("id_benh, loai_chan_doan, benh(ten_benh)") \
            .eq("id_ho_so", id_ho_so) \
            .execute()
        
        segments = []
        if chan_doan_res.data:
            main_diseases = []
            other_diseases = []
            
            for item in chan_doan_res.data:
                ten_benh = item.get("benh", {}).get("ten_benh", "Không rõ")
                loai = item.get("loai_chan_doan", "")
                
                if loai == "Bệnh chính":
                    main_diseases.append(ten_benh)
                else:
                    other_diseases.append(ten_benh)
            
            if main_diseases:
                content = f"- Chẩn đoán chính: {', '.join(main_diseases)}"
                if not content.endswith('.'):
                    content += '.'
                segments.append(SourceSegment(
                    content=content,
                    source_type="CHẨN ĐOÁN",
                    source_id=None
                ))
            
            if other_diseases:
                content = f"- Bệnh kèm theo: {', '.join(other_diseases)}"
                if not content.endswith('.'):
                    content += '.'
                segments.append(SourceSegment(
                    content=content,
                    source_type="CHẨN ĐOÁN",
                    source_id=None
                ))
        return segments

    def _fetch_notes(self, id_ho_so: int) -> tuple[List[Dict[str, Any]], List[SourceSegment]]:
        ghi_chu_res = self.client.table("ghi_chu_y_te") \
            .select("id_ghi_chu, noi_dung_ghi_chu, loai_ghi_chu!inner(ten_loai_ghi_chu, thu_tu_uu_tien, send_to_ai)") \
            .eq("id_ho_so", id_ho_so) \
            .eq("loai_ghi_chu.send_to_ai", True) \
            .neq("loai_ghi_chu.ten_loai_ghi_chu", "Thông tin dị ứng") \
            .execute()
        
        def get_priority(item):
            try: return item["loai_ghi_chu"]["thu_tu_uu_tien"]
            except: return 9999
        sorted_notes = sorted(ghi_chu_res.data, key=get_priority)

        segments = []
        if sorted_notes:
            for note in sorted_notes:
                ten_loai = note["loai_ghi_chu"]["ten_loai_ghi_chu"]
                noi_dung = note["noi_dung_ghi_chu"]
                segments.append(SourceSegment(
                    content=noi_dung, # Content only contains the note body
                    source_type=ten_loai, # Use the actual note type as source_type
                    source_id=note['id_ghi_chu']
                ))
        return sorted_notes, segments

    def _fetch_lab_results(self, visit_ids: List[int]) -> List[SourceSegment]:
        if not visit_ids:
            return []
            
        chi_dinh_res = self.client.table("chi_dinh_cls") \
            .select("id_chi_dinh, dich_vu_cls(ten_dich_vu), ket_qua_cls(ket_luan, chi_so_xet_nghiem)") \
            .in_("id_lich_kham", visit_ids) \
            .execute()
        
        segments = []
        for item in chi_dinh_res.data:
            ten_dv = item.get("dich_vu_cls", {}).get("ten_dich_vu", "Dịch vụ ?")
            kq_list = item.get("ket_qua_cls", [])
            kq_data = kq_list[0] if isinstance(kq_list, list) and len(kq_list) > 0 else kq_list if isinstance(kq_list, dict) else None

            if kq_data:
                ket_luan = kq_data.get("ket_luan", "")
                chi_so = kq_data.get("chi_so_xet_nghiem")
                
                res_str = f"- {ten_dv}: {ket_luan}"
                if chi_so:
                        res_str += f" (Chỉ số: {chi_so})"
                
                segments.append(SourceSegment(
                    content=res_str,
                    source_type="CẬN LÂM SÀNG",
                    source_id=item['id_chi_dinh']
                ))
        return segments

    def _fetch_prescriptions(self, visit_ids: List[int]) -> List[SourceSegment]:
        if not visit_ids:
            return []

        don_thuoc_res = self.client.table("don_thuoc") \
            .select("id_don_thuoc, thoi_gian_ke_don") \
            .in_("id_lich_kham", visit_ids) \
            .execute()
        
        don_thuoc_ids = [dt["id_don_thuoc"] for dt in don_thuoc_res.data]
        segments = []

        if don_thuoc_ids:
            chi_tiet_res = self.client.table("chi_tiet_don_thuoc") \
                .select("id_don_thuoc, so_luong, lieu_dung, thuoc(ten_thuoc, don_vi_tinh)") \
                .in_("id_don_thuoc", don_thuoc_ids) \
                .execute()
            
            if chi_tiet_res.data:
                for item in chi_tiet_res.data:
                    ten_thuoc = item.get("thuoc", {}).get("ten_thuoc", "Thuốc ?")
                    dvt = item.get("thuoc", {}).get("don_vi_tinh", "")
                    sl = item.get("so_luong", 0)
                    lieu = item.get("lieu_dung", "")
                    
                    segments.append(SourceSegment(
                        content=f"- {ten_thuoc} ({sl} {dvt}): {lieu}",
                        source_type="ĐƠN THUỐC",
                        source_id=item.get('id_don_thuoc')
                    ))
        return segments

    async def prepare_input(self, id_ho_so: int) -> PreparedInput:
        segments = []

        # --- Phase 1: Parallel Independent Queries ---
        # Run independent queries concurrently in threads
        
        # Define tasks
        task_patient = asyncio.to_thread(self._fetch_patient_info, id_ho_so)
        task_allergies = asyncio.to_thread(self._fetch_allergies, id_ho_so)
        task_visits = asyncio.to_thread(self._fetch_visits, id_ho_so)
        task_diagnoses = asyncio.to_thread(self._fetch_diagnoses, id_ho_so)
        task_notes = asyncio.to_thread(self._fetch_notes, id_ho_so)

        # Execute Phase 1
        res_patient, res_allergies, res_visits, res_diagnoses, res_notes = await asyncio.gather(
            task_patient, task_allergies, task_visits, task_diagnoses, task_notes
        )

        # Unpack results
        visit_ids, visit_segments = res_visits
        sorted_notes, note_segments = res_notes

        # Add initial segments in order
        segments.append(res_patient)
        segments.append(res_allergies)
        segments.extend(visit_segments)
        
        # Check diagnoses and notes for fallback
        has_formal_diagnosis = len(res_diagnoses) > 0
        has_note_diagnosis = any(n["loai_ghi_chu"]["ten_loai_ghi_chu"] == "Chẩn đoán xác định" for n in sorted_notes)

        if not has_formal_diagnosis and not has_note_diagnosis:
             segments.append(SourceSegment(
                content="- Chưa có kết luận, đang chờ kết quả cận lâm sàng.",
                source_type="CHẨN ĐOÁN",
                source_id=None
            ))
        else:
            segments.extend(res_diagnoses)

        # --- Phase 2: Dependent Queries (if visits exist) ---
        if visit_ids:
             task_labs = asyncio.to_thread(self._fetch_lab_results, visit_ids)
             task_prescriptions = asyncio.to_thread(self._fetch_prescriptions, visit_ids)
             
             res_labs, res_prescriptions = await asyncio.gather(task_labs, task_prescriptions)
             
             segments.extend(res_labs)
             segments.extend(note_segments) # Notes usually come before or after labs depending on style, keeping original order: after labs
             segments.extend(res_prescriptions)
        else:
            segments.extend(note_segments)

        # --- 6. Final Aggregate with Smart Headers ---
        prompt_parts = []
        last_group_type = None

        for s in segments:
            chunk = ""
            stype = s.source_type
            
            # Grouping logic for lists
            if stype in ["CHẨN ĐOÁN", "CẬN LÂM SÀNG", "ĐƠN THUỐC"]:
                if last_group_type != stype:
                    if stype == "CẬN LÂM SÀNG":
                        chunk += "**KẾT QUẢ CẬN LÂM SÀNG:**\n"
                    else:
                        chunk += f"**{stype}:**\n"
                chunk += s.content
                last_group_type = stype
            
            # Single items logic
            else:
                last_group_type = None # Reset group
                
                if stype == "THÔNG TIN CHUNG": # Handles patient info and allergies
                    chunk += s.content
                elif stype.startswith("Khám ngày"):
                    # These already have the "header" as part of their content, just add bold if not already bold.
                    # Ensure it starts with bold if it doesn't already (e.g. from _fetch_allergies or _fetch_visits)
                    if not s.content.strip().startswith('**'):
                         if ':' in s.content: # If there's a colon, assume format like "HEADER: content"
                            parts = s.content.split(':', 1)
                            chunk += f"**{parts[0].strip()}:**{parts[1]}"
                         else: # Otherwise, assume the entire content is the header, and add a colon inside bold
                            chunk += f"**{s.content.strip()}:**"
                    else:
                        chunk += s.content # Already bolded, or content itself is the header
                else:
                    # Generic Notes
                    chunk += f"**{stype.upper()}:**\n{s.content}"
            
            prompt_parts.append(chunk)

        prompt_content = "\n\n".join(prompt_parts)

        return PreparedInput(
            id_ho_so=id_ho_so,
            prompt_content=prompt_content,
            raw_notes_count=len(sorted_notes),
            source_segments=segments
        )

supabase_service = SupabaseService()
