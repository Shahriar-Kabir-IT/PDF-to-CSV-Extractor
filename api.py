import os
from typing import Optional, Tuple

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse

from llm_pdf_extractor import (
    extract_table_with_llm,
    save_csv,
)


app = FastAPI(title="PDF Table Extractor API", version="1.0.0")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


def _parse_bbox(bbox: Optional[str]) -> Optional[Tuple[float, float, float, float]]:
    if not bbox:
        return None
    parts = [p.strip() for p in bbox.split(",")]
    if len(parts) != 4:
        raise HTTPException(status_code=400, detail="bbox must be 'x0,y0,x1,y1'")
    try:
        return tuple(float(p) for p in parts)  # type: ignore
    except Exception:
        raise HTTPException(status_code=400, detail="bbox values must be numeric")


@app.post("/extract")
async def extract(
    file: UploadFile = File(...),
    bbox: Optional[str] = None,
    scale: float = 3.0,
    save_csv_flag: bool = True,
    use_ollama: bool = False,
    model: str = "gpt-4o-mini",
    ollama_model: str = "llava:latest",
    api_key: Optional[str] = None,
):
    try:
        pdf_bytes = await file.read()
        bbox_tuple = _parse_bbox(bbox)
        rows = extract_table_with_llm(
            pdf_bytes,
            bbox=bbox_tuple,
            scale=scale,
            use_ollama=use_ollama,
            openai_model=model,
            openai_api_key=api_key or os.getenv("OPENAI_API_KEY"),
            ollama_model=ollama_model,
        )

        saved_path = None
        if save_csv_flag:
            out_dir = os.path.join(os.getcwd(), "updated_excel_files")
            os.makedirs(out_dir, exist_ok=True)
            base = os.path.splitext(os.path.basename(file.filename or "extracted"))[0]
            saved_path = save_csv(rows, os.path.join(out_dir, f"{base}_extracted_table_data.csv"))

        return JSONResponse(
            {
                "status": "success",
                "rows_count": len(rows),
                "rows": rows,
                "csv_path": saved_path,
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)