# app/data_schema.py
from pydantic import BaseModel, Field

# Defines the expected input format for the /predict endpoint
class DiabetesFeatures(BaseModel):
    age: float = Field(..., description="Age in normalized units", example=0.02)
    sex: float = Field(..., description="Sex in normalized units", example=-0.044)
    bmi: float = Field(..., description="Body mass index in normalized units", example=0.06)
    bp: float = Field(..., description="Average blood pressure in normalized units", example=-0.03)
    s1: float = Field(..., description="TC, total cholesterol in normalized units", example=-0.02)
    s2: float = Field(..., description="LDL, low-density lipoproteins in normalized units", example=0.03)
    s3: float = Field(..., description="HDL, high-density lipoproteins in normalized units", example=-0.02)
    s4: float = Field(..., description="TCH, thyroid stimulating hormone in normalized units", example=0.02)
    s5: float = Field(..., description="LOG, lamotrigine in normalized units", example=0.02)
    s6: float = Field(..., description="GLU, blood sugar in normalized units", example=-0.001)

    class Config:
        schema_extra = {
            "example": {
                "age": 0.02, "sex": -0.044, "bmi": 0.06, "bp": -0.03, "s1": -0.02,
                "s2": 0.03, "s3": -0.02, "s4": 0.02, "s5": 0.02, "s6": -0.001
            }
        }