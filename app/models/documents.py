from pydantic import BaseModel

class DocumentUpload(BaseModel):
    filename: str
    content: str
    

class DocumentResponse(BaseModel):
    id: str
    filename: str
    status: str