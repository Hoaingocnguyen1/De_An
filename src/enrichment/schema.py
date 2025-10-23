from typing import List
from pydantic import BaseModel, Field

class EnrichmentOutput(BaseModel):
    """
    Cấu trúc dữ liệu chuẩn cho kết quả làm giàu nội dung.
    """
    summary: str = Field(description="Một bản tóm tắt súc tích, có ý nghĩa về nội dung đầu vào.")
    keywords: List[str] = Field(description="Một danh sách các từ khóa kỹ thuật hoặc các khái niệm chính.")