from pdf2image import convert_from_path
import pytesseract, re
import os
import tempfile
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel


pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Users\neha.mandwal\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"
)

POPLER_PATH = r"C:\poppler-23.05.0\poppler-24.08.0\Library\bin"

class KFTFeatures(BaseModel):
    Creatinine : float
    Urea : float
    Potassium : float
    Sodium : float
    Phosphorus : float

app = FastAPI(title="Step1- KFT preprocessing")

def extract_kft_from_pdf(path: str)-> dict:
    try:
        pages = convert_from_path(path, dpi=300, poppler_path=POPLER_PATH)
    except Exception as e:
        raise RuntimeError(f"PDF to image failed :{e}")
    
    text=[]
    for page in pages:
        text.append(pytesseract.image_to_string(page, lang='eng'))
    full_text= "\n".join(text)


    patterns= {
        'Creatinine':  r'Creatinine\s*[:=]?\s*([\d.]+)',
        'Urea':        r'(?:BUN|Urea)\s*[:=]?\s*([\d.]+)',
        'Potassium':   r'Potassium\s*[:=]?\s*([\d.]+)',
        'Sodium':      r'Sodium\s*[:=]?\s*([\d.]+)',
        'Phosphorus':  r'Phosphorus\s*[:=]?\s*([\d.]+)',
    }

    result = {}
    for name, pat in patterns.items():
        m = re.search(pat, full_text, re.IGNORECASE)
        if m:
            try:
                result[name] = float(m.group(1))
            except ValueError:
                result[name] = 0.0
        else:
            result[name] = 0.0
    return result




@app.post("/upload-kft", response_model=KFTFeatures)
async def upload_kft(report: UploadFile = File(...)):
    """
    1) Accepts a PDF KFT report upload
    2) Runs OCR + regex extraction
    3) Returns structured KFT features as JSON
    """
    # 3. Validate file type
    if not report.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF reports are supported.")

    # 4. Save to temporary file
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            contents = await report.read()
            tmp.write(contents)
            tmp_path = tmp.name
    except Exception as e:
        raise HTTPException(500, f"Could not save upload: {e}")

    # 5. Extract KFT
    try:
        features = extract_kft_from_pdf(tmp_path)
    except Exception as e:
        raise HTTPException(500, str(e))
    finally:
        os.unlink(tmp_path)

    # 6. Return features for frontend display
    return features
