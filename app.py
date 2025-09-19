import os
from dotenv import load_dotenv
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import coo_matrix, csr_matrix
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
    b.id AS board_id,
    b.user_id AS author_id,
    b.title,
    b.content,
    b.category,
    CASE WHEN bl.user_id IS NOT NULL THEN 1 ELSE 0 END AS liked_by_user,
    CASE WHEN c.user_id IS NOT NULL THEN 1 ELSE 0 END AS commented_by_user
FROM boards b
-- 특정 유저가 추천했는지 체크
LEFT JOIN board_likes bl 
    ON b.id = bl.board_id AND bl.user_id = 3
-- 특정 유저가 댓글 달았는지 체크
LEFT JOIN comments c 
    ON b.id = c.board_id AND c.user_id = 3
ORDER BY b.id;

"""

data = pd.read_sql(query, engine)
print(data)

# app = FastAPI()

# origins = [
#     os.getenv("FRONTEND_APP_URL"),
# ]

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,          
#     allow_credentials=True,         
#     allow_methods=["*"],            
#     allow_headers=["*"],          
# )
# user_enc = LabelEncoder()
# item_enc = LabelEncoder()
# data['user_idx'] = user_enc.fit_transform(data['user_id'])
# data['item_idx'] = item_enc.fit_transform(data['item_id'])
# matrix = coo_matrix((data['carts_count'], (data['user_idx'], data['item_idx'])))
# user_item_matrix = matrix.tocsr()
# model = AlternatingLeastSquares(factors=10, iterations=15)
# model.fit(user_item_matrix)

# @app.get("/recommend")
# def recommend(user_id: int = Query(..., description="원본 user_id 입력 (예: 3)"), top_n: int = 3):
#     if user_id not in data['user_id'].values:
#         raise HTTPException(status_code=404, detail="해당 user_id는 데이터에 없습니다.")
#     user_idx = user_enc.transform([user_id])[0]
#     user_vector = csr_matrix(user_item_matrix[user_idx])
#     item_indices, scores = model.recommend(
#         userid=user_idx, 
#         user_items=user_vector,
#         N=top_n
#     )
#     item_ids = item_enc.inverse_transform(item_indices)

#     result = [
#         {"id": int(item_id), "score": round(float(score), 5)}
#         for item_id, score in zip(item_ids, scores)
#     ]
#     return result