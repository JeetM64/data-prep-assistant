"""sample_datasets.py — built-in sample datasets for quick demo"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
import pandas as pd
import io

router = APIRouter()

class SampleRequest(BaseModel):
    target_column: Optional[str] = None

DATASETS = {
    "iris": {
        "csv": """sepal_length,sepal_width,petal_length,petal_width,species
5.1,3.5,1.4,0.2,setosa
4.9,3.0,1.4,0.2,setosa
4.7,3.2,1.3,0.2,setosa
7.0,3.2,4.7,1.4,versicolor
6.4,3.2,4.5,1.5,versicolor
6.9,3.1,4.9,1.5,versicolor
5.5,2.3,4.0,1.3,versicolor
6.3,3.3,6.0,2.5,virginica
5.8,2.7,5.1,1.9,virginica
7.1,3.0,5.9,2.1,virginica
6.3,2.9,5.6,1.8,virginica
5.1,3.5,1.4,0.2,setosa
4.6,3.1,1.5,0.2,setosa
5.0,3.6,1.4,0.2,setosa
5.4,3.9,1.7,0.4,setosa
4.6,3.4,1.4,0.3,setosa
5.0,3.4,1.5,0.2,setosa
4.4,2.9,1.4,0.2,setosa
4.9,3.1,1.5,0.1,setosa
5.4,3.7,1.5,0.2,setosa
6.5,2.8,4.6,1.5,versicolor
5.7,2.8,4.5,1.3,versicolor
6.3,3.3,4.7,1.6,versicolor
4.9,2.4,3.3,1.0,versicolor
6.6,2.9,4.6,1.3,versicolor
7.6,3.0,6.6,2.1,virginica
4.9,2.5,4.5,1.7,virginica
7.3,2.9,6.3,1.8,virginica
6.7,2.5,5.8,1.8,virginica
7.2,3.6,6.1,2.5,virginica""",
        "target": "species"
    },
    "titanic": {
        "csv": """survived,pclass,sex,age,sibsp,parch,fare,embarked
0,3,male,22.0,1,0,7.25,S
1,1,female,38.0,1,0,71.28,C
1,3,female,26.0,0,0,7.93,S
1,1,female,35.0,1,0,53.1,S
0,3,male,35.0,0,0,8.05,S
0,3,male,,0,0,8.46,Q
0,1,male,54.0,0,0,51.86,S
0,3,male,2.0,3,1,21.08,S
1,3,female,27.0,0,2,11.13,S
1,2,female,14.0,1,0,30.07,C
1,3,female,4.0,1,1,16.7,S
1,1,female,58.0,0,0,26.55,S
0,3,male,20.0,0,0,8.05,S
0,3,male,39.0,1,5,31.28,S
0,3,female,14.0,0,0,7.85,S
1,2,female,55.0,0,0,16.0,S
0,3,male,2.0,4,1,29.13,Q
1,1,male,28.0,0,0,35.5,S
0,2,male,40.0,0,0,13.0,S
1,3,male,21.0,0,0,8.05,S
0,1,male,45.0,1,0,83.48,S
1,2,female,29.0,1,0,26.0,S
0,2,male,28.5,0,0,13.0,S
1,2,female,5.0,1,2,27.75,S
0,3,male,11.0,5,2,46.9,S
0,3,male,22.0,0,0,7.23,C
1,2,female,38.0,0,0,80.0,S
0,1,male,45.0,1,0,83.48,S
1,3,female,8.0,3,1,21.08,S
0,3,male,19.0,0,0,7.90,S""",
        "target": "survived"
    },
    "wine": {
        "csv": """fixed_acidity,volatile_acidity,citric_acid,residual_sugar,chlorides,free_sulfur_dioxide,total_sulfur_dioxide,density,pH,sulphates,alcohol,quality
7.4,0.7,0.0,1.9,0.076,11.0,34.0,0.9978,3.51,0.56,9.4,5
7.8,0.88,0.0,2.6,0.098,25.0,67.0,0.9968,3.2,0.68,9.8,5
11.2,0.28,0.56,1.9,0.075,17.0,60.0,0.998,3.16,0.58,9.8,6
7.4,0.7,0.0,1.9,0.076,11.0,34.0,0.9978,3.51,0.56,9.4,5
7.9,0.6,0.06,1.6,0.069,15.0,59.0,0.9964,3.3,0.46,9.4,5
7.3,0.65,0.0,1.2,0.065,15.0,21.0,0.9946,3.39,0.47,10.0,7
7.8,0.58,0.02,2.0,0.073,9.0,18.0,0.9968,3.36,0.57,9.5,7
7.5,0.5,0.36,6.1,0.071,17.0,102.0,0.9978,3.35,0.8,10.5,5
8.9,0.62,0.18,3.8,0.176,52.0,145.0,0.9986,3.16,0.88,9.2,5
8.5,0.28,0.56,1.8,0.092,35.0,103.0,0.9969,3.3,0.75,10.5,7
7.0,0.735,0.05,2.0,0.086,5.0,14.0,0.9962,3.37,0.48,9.5,5
8.6,0.49,0.28,1.9,0.11,20.0,136.0,0.9972,2.93,1.95,9.9,6
7.6,0.41,0.37,1.8,0.097,31.0,47.0,0.9966,3.24,0.64,9.6,5
6.9,0.4,0.14,2.4,0.085,21.0,40.0,0.9968,3.43,0.63,9.7,6
6.3,0.39,0.16,1.4,0.08,11.0,23.0,0.9955,3.34,0.56,9.3,5
7.2,0.39,0.31,2.8,0.086,25.0,113.0,0.9968,3.21,0.71,9.9,6
8.3,0.42,0.62,19.25,0.04,41.0,172.0,1.0002,2.98,0.67,9.7,5
7.1,0.71,0.0,1.9,0.08,14.0,35.0,0.9972,3.47,0.55,9.4,5
5.6,0.615,0.0,1.6,0.089,16.0,59.0,0.9943,3.58,0.52,9.9,5
7.8,0.645,0.0,2.0,0.082,8.0,16.0,0.9964,3.38,0.59,9.8,6
6.9,0.685,0.0,2.5,0.105,22.0,37.0,0.9966,3.46,0.57,9.2,5
7.6,0.41,0.37,1.8,0.097,31.0,47.0,0.9966,3.24,0.64,9.6,5
8.9,0.22,0.48,1.8,0.077,29.0,60.0,0.9968,3.39,0.53,9.4,6
6.5,0.36,0.29,1.6,0.021,24.0,85.0,0.9923,3.41,0.61,11.4,6
7.2,0.39,0.31,2.8,0.086,25.0,113.0,0.9968,3.21,0.71,9.9,6
8.1,0.56,0.28,1.7,0.368,16.0,56.0,0.9968,3.11,1.28,9.3,5
7.4,0.59,0.08,4.4,0.086,6.0,29.0,0.9974,3.38,0.5,9.0,4
7.9,0.32,0.51,1.8,0.341,17.0,56.0,0.9969,3.04,1.08,9.2,6
8.6,0.38,0.36,3.0,0.081,30.0,119.0,0.9974,3.29,0.89,9.4,6
7.3,0.45,0.36,5.9,0.074,12.0,87.0,0.9978,3.33,0.83,10.5,5""",
        "target": "quality"
    }
}

@router.post("/sample/{dataset_name}")
async def analyze_sample(dataset_name: str, request: SampleRequest = None):
    if dataset_name not in DATASETS:
        raise HTTPException(404, f"Dataset '{dataset_name}' not found. Available: {list(DATASETS.keys())}")

    from app.main import _run_full_analysis
    try:
        ds = DATASETS[dataset_name]
        df = pd.read_csv(io.StringIO(ds["csv"]))
        target = (request.target_column if request and request.target_column else ds["target"])
        return _run_full_analysis(df, target_column=target)
    except Exception as e:
        raise HTTPException(500, f"Sample analysis failed: {e}")