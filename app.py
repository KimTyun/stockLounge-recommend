import os
from dotenv import load_dotenv
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import coo_matrix
from implicit.als import AlternatingLeastSquares
from sqlalchemy import create_engine

load_dotenv()

DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")

engine = create_engine(f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}")

query = """ 
SELECT 
    x.user_id,
    x.board_id,
    SUM(x.interaction) AS interaction
FROM (
    -- 1) 좋아요 이벤트 (가중치 = 1)
    SELECT 
        bl.user_id,
        bl.board_id,
        1 * (1 / (1 + b.report_count)) AS interaction
    FROM board_likes bl
    JOIN boards b ON b.id = bl.board_id
    WHERE bl.user_id <> b.user_id

    UNION ALL

    -- 2) 댓글 이벤트 (가중치 = 0.7)
    SELECT
        c.user_id,
        c.board_id,
        0.7 * (1 / (1 + (c.report_count + b.report_count))) AS interaction
    FROM comments c
    JOIN boards b ON b.id = c.board_id
    WHERE c.user_id <> b.user_id
) x
GROUP BY x.user_id, x.board_id;
"""

data = pd.read_sql(query, engine)
print(data)

app = FastAPI()

origins = [
    os.getenv("FRONTEND_APP_URL"),
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,          
    allow_credentials=True,         
    allow_methods=["*"],            
    allow_headers=["*"],          
)

# 인코딩
user_enc = LabelEncoder()
item_enc = LabelEncoder()
data['user_idx'] = user_enc.fit_transform(data['user_id'])
data['item_idx'] = item_enc.fit_transform(data['board_id'])

# sparse matrix
matrix = coo_matrix(
    (data['interaction'], (data['user_idx'], data['item_idx']))
)
user_item_matrix = matrix.tocsr()

# ALS 모델 학습
model = AlternatingLeastSquares(factors=10, iterations=15)
model.fit(user_item_matrix)

# boards 테이블에서 id, title 불러오기
boards_df = pd.read_sql("SELECT id AS board_id, title FROM boards", engine)


# 추천 API
@app.get("/recommend")
def recommend(user_id: int = Query(..., description="원본 user_id 입력 (예: 3)"), top_n: int = 5):
    if user_id not in data['user_id'].values:
        raise HTTPException(status_code=404, detail="해당 user_id는 데이터에 없습니다.")
    
    # user_id → index
    user_idx = user_enc.transform([user_id])[0]

    # ALS 추천
    item_indices, scores = model.recommend(
        userid=user_idx,
        user_items=user_item_matrix[user_idx],
        N=top_n
    )

    # index → board_id
    item_ids = item_enc.inverse_transform(item_indices)

    # board_id → title 매핑
    result = []
    for board_id, score in zip(item_ids, scores):
        title_row = boards_df.loc[boards_df['board_id'] == board_id, 'title']
        title = title_row.values[0] if not title_row.empty else ""
        result.append({"board_id": int(board_id), "title": title, "score": round(float(score), 5)})

    return result
