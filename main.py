import os
import re
import json
import hmac
import hashlib
import secrets
import sqlite3
from typing import Optional

import pandas as pd
import uvicorn

from dotenv import load_dotenv
from openai import OpenAI

from fastapi import FastAPI, Request, Form, UploadFile, File
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware


# ============================================================
# 기본 설정
# ============================================================

load_dotenv()

app = FastAPI()

app.add_middleware(
    SessionMiddleware,
    secret_key=os.getenv("SESSION_SECRET_KEY", "finanfit-secret-key-1439"),
)

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

DB_PATH = "users.db"
CSV_PATH = os.path.join("static", "예금목록.csv")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None


# ============================================================
# SQLite DB
# ============================================================

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db()

    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            salt TEXT NOT NULL,
            name TEXT NOT NULL,
            email TEXT NOT NULL,
            age INTEGER NOT NULL,
            gender TEXT NOT NULL,
            investment_type TEXT NOT NULL,
            job TEXT NOT NULL,
            income INTEGER NOT NULL,
            profile_image TEXT
        )
    """)

    conn.commit()
    conn.close()


init_db()


# ============================================================
# 비밀번호 처리
# ============================================================

def hash_password(password: str, salt: Optional[str] = None):
    if salt is None:
        salt = secrets.token_hex(16)

    hashed = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        100000
    ).hex()

    return hashed, salt


def verify_password(password: str, hashed_password: str, salt: str):
    new_hash, _ = hash_password(password, salt)

    return hmac.compare_digest(new_hash, hashed_password)


# ============================================================
# CSV 불러오기
# ============================================================

def load_products():
    try:
        df = pd.read_csv(CSV_PATH)
    except UnicodeDecodeError:
        df = pd.read_csv(CSV_PATH, encoding="cp949")

    df = df.fillna("")

    numeric_columns = ["최고금리_숫자", "기본금리_숫자"]

    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    return df


# ============================================================
# 추천 알고리즘
# ============================================================

def extract_min_amount(text):
    """
    가입금액 문자열에서 최소 가입금액을 대략 추출
    """

    if not text:
        return 0

    text = str(text).replace(",", "")

    patterns = [
        (r"(\d+)\s*억원", 100000000),
        (r"(\d+)\s*천만원", 10000000),
        (r"(\d+)\s*백만원", 1000000),
        (r"(\d+)\s*만원", 10000),
        (r"(\d+)\s*천원", 1000),
    ]

    for pattern, unit in patterns:
        match = re.search(pattern, text)

        if match:
            return int(match.group(1)) * unit

    return 0


def recommend_products(user, count=3):
    df = load_products()

    scores = []

    age = int(user["age"])
    income = int(user["income"])
    investment_type = user["investment_type"]
    job = user["job"]

    for index, row in df.iterrows():

        score = 0.0

        max_rate = float(row.get("최고금리_숫자", 0))
        base_rate = float(row.get("기본금리_숫자", 0))

        term = str(row.get("가입기간", ""))
        method = str(row.get("가입방법", ""))
        target = str(row.get("가입대상", ""))
        amount_text = str(row.get("가입금액", ""))

        min_amount = extract_min_amount(amount_text)

        # ----------------------------------------------------
        # 기본 금리 점수
        # ----------------------------------------------------

        score += max_rate * 8
        score += base_rate * 4

        # ----------------------------------------------------
        # 투자성향
        # ----------------------------------------------------

        if investment_type == "안정형":
            score += base_rate * 8

            if "정기예금" in str(row.get("상품명", "")):
                score += 5

            if "1년" in term or "12개월" in term:
                score += 3

        elif investment_type == "안정추구형":
            score += base_rate * 6
            score += max_rate * 3

            if "12개월" in term:
                score += 3

        elif investment_type == "위험중립형":
            score += max_rate * 6

            if "스마트폰" in method or "인터넷" in method:
                score += 3

        elif investment_type == "적극투자형":
            score += max_rate * 9

            if max_rate >= 3:
                score += 6

        elif investment_type == "공격투자형":
            score += max_rate * 11

            if max_rate >= 3.5:
                score += 10

        # ----------------------------------------------------
        # 나이
        # ----------------------------------------------------

        if age < 30:

            if "스마트폰" in method or "인터넷" in method:
                score += 6

            if "1년" in term or "12개월" in term:
                score += 3

        elif age < 50:

            if "12개월" in term or "1년" in term:
                score += 4

            if base_rate >= 2.5:
                score += 3

        else:

            if "연금" in str(row.get("상품명", "")):
                score += 10

            if "장기" in term or "년" in term:
                score += 4

        # ----------------------------------------------------
        # 직업
        # ----------------------------------------------------

        if job in ["직장인", "공무원"]:

            if "스마트폰" in method or "인터넷" in method:
                score += 4

        if job in ["자영업자", "사업자"]:

            if "개인사업자" in target:
                score += 7

        if job == "학생":

            if min_amount <= 1000000:
                score += 8

        # ----------------------------------------------------
        # 소득
        # ----------------------------------------------------

        if income < 30000000:

            if min_amount <= 1000000:
                score += 7

        elif income < 60000000:

            if min_amount <= 5000000:
                score += 5

        else:

            if max_rate >= 3:
                score += 5

        scores.append((index, score))

    scores.sort(key=lambda x: x[1], reverse=True)

    selected_indexes = [x[0] for x in scores[:count]]

    results = []

    for idx in selected_indexes:
        row = df.loc[idx]

        results.append({
            "bank": row.get("금융사", ""),
            "name": row.get("상품명", ""),
            "max_rate": row.get("최고금리", ""),
            "base_rate": row.get("기본금리", ""),
            "period": row.get("가입기간", ""),
            "amount": row.get("가입금액", ""),
            "method": row.get("가입방법", ""),
            "target": row.get("가입대상", ""),
            "url": row.get("상세URL", ""),
        })

    return results


# ============================================================
# 로그인 여부
# ============================================================

def get_current_user(request: Request):

    user_id = request.session.get("user_id")

    if not user_id:
        return None

    conn = get_db()

    user = conn.execute(
        "SELECT * FROM users WHERE id = ?",
        (user_id,)
    ).fetchone()

    conn.close()

    return user


# ============================================================
# 로그인 페이지
# ============================================================

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):

    if request.session.get("user_id"):
        return RedirectResponse("/main", status_code=302)

    return RedirectResponse("/login", status_code=302)


@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):

    return templates.TemplateResponse(
        request=request,
        name="login.html",
        context={
            "error": None
        }
    )


@app.post("/login", response_class=HTMLResponse)
async def login(
    request: Request,
    username: str = Form(...),
    password: str = Form(...)
):

    conn = get_db()

    user = conn.execute(
        "SELECT * FROM users WHERE username = ?",
        (username,)
    ).fetchone()

    conn.close()

    if not user:

        return templates.TemplateResponse(
            request=request,
            name="login.html",
            context={
                "error": "아이디 또는 비밀번호가 올바르지 않습니다."
            }
        )

    if not verify_password(
        password,
        user["password"],
        user["salt"]
    ):

        return templates.TemplateResponse(
            request=request,
            name="login.html",
            context={
                "error": "아이디 또는 비밀번호가 올바르지 않습니다."
            }
        )

    request.session["user_id"] = user["id"]

    return RedirectResponse("/main", status_code=303)


# ============================================================
# 회원가입
# ============================================================

@app.get("/signup", response_class=HTMLResponse)
async def signup_page(request: Request):

    return templates.TemplateResponse(
        request=request,
        name="signup.html",
        context={
            "error": None
        }
    )


@app.post("/signup", response_class=HTMLResponse)
async def signup(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
    name: str = Form(...),
    email: str = Form(...),
    age: int = Form(...),
    gender: str = Form(...),
    investment_type: str = Form(...),
    job: str = Form(...),
    income: int = Form(...),
    profile_image: UploadFile = File(None),
):

    conn = get_db()

    exists = conn.execute(
        "SELECT id FROM users WHERE username = ?",
        (username,)
    ).fetchone()

    if exists:

        conn.close()

        return templates.TemplateResponse(
            request=request,
            name="signup.html",
            context={
                "error": "이미 사용 중인 아이디입니다."
            }
        )

    password_hash, salt = hash_password(password)

    profile_path = None

    if profile_image and profile_image.filename:

        os.makedirs(
            os.path.join("static", "profiles"),
            exist_ok=True
        )

        extension = os.path.splitext(profile_image.filename)[1]

        file_name = f"{secrets.token_hex(12)}{extension}"

        save_path = os.path.join(
            "static",
            "profiles",
            file_name
        )

        content = await profile_image.read()

        with open(save_path, "wb") as f:
            f.write(content)

        profile_path = f"/static/profiles/{file_name}"

    try:

        conn.execute("""
            INSERT INTO users (
                username,
                password,
                salt,
                name,
                email,
                age,
                gender,
                investment_type,
                job,
                income,
                profile_image
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            username,
            password_hash,
            salt,
            name,
            email,
            age,
            gender,
            investment_type,
            job,
            income,
            profile_path
        ))

        conn.commit()

    except Exception as e:

        conn.close()

        return templates.TemplateResponse(
            request=request,
            name="signup.html",
            context={
                "error": str(e)
            }
        )

    conn.close()

    return RedirectResponse("/login", status_code=303)


# ============================================================
# 메인 화면
# ============================================================

@app.get("/main", response_class=HTMLResponse)
async def main_page(request: Request):

    user = get_current_user(request)

    if not user:
        return RedirectResponse("/login", status_code=302)

    products = recommend_products(user, 3)

    return templates.TemplateResponse(
        request=request,
        name="main.html",
        context={
            "user": user,
            "products": products
        }
    )


# ============================================================
# 챗봇 화면
# ============================================================

@app.get("/chatbot", response_class=HTMLResponse)
async def chatbot_page(request: Request):

    user = get_current_user(request)

    if not user:
        return RedirectResponse("/login", status_code=302)

    return templates.TemplateResponse(
        request=request,
        name="chatbot.html",
        context={
            "user": user
        }
    )


# ============================================================
# 챗봇 API
# ============================================================

@app.post("/api/chat")
async def chatbot_api(request: Request):

    user = get_current_user(request)

    if not user:

        return JSONResponse(
            {
                "answer": "로그인이 필요합니다."
            },
            status_code=401
        )

    data = await request.json()

    message = data.get("message", "").strip()

    if not message:

        return {
            "answer": "질문을 입력해주세요."
        }

    products = recommend_products(user, 5)

    product_text = json.dumps(
        products,
        ensure_ascii=False,
        indent=2
    )

    system_prompt = f"""
당신은 FinanFit 금융상품 상담 챗봇입니다.

사용자 정보:
이름: {user["name"]}
나이: {user["age"]}
투자성향: {user["investment_type"]}
직업: {user["job"]}
연소득: {user["income"]}

현재 추천 가능한 예금상품 데이터:
{product_text}

규칙:
1. 한국어로 답변합니다.
2. 제공된 금융상품 정보를 우선 사용합니다.
3. 금리, 가입기간, 가입금액, 가입방법을 구체적으로 설명합니다.
4. 사용자의 나이, 투자성향, 직업, 소득을 고려합니다.
5. 없는 상품 정보를 만들어내지 않습니다.
6. 투자 수익을 보장한다고 표현하지 않습니다.
7. 최종 가입 조건과 금리는 금융사 공식 페이지에서 다시 확인하도록 안내합니다.
8. 답변은 이해하기 쉽게 작성합니다.
"""

    try:

        if client is None:

            return {
                "answer": "OPENAI_API_KEY가 설정되어 있지 않습니다."
            }

        response = client.responses.create(
            model="gpt-4o-mini",
            instructions=system_prompt,
            input=message
        )

        answer = response.output_text

        return {
            "answer": answer
        }

    except Exception as e:

        return {
            "answer": f"챗봇 처리 중 오류가 발생했습니다.\n{str(e)}"
        }


# ============================================================
# 로그아웃
# ============================================================

@app.get("/logout")
async def logout(request: Request):

    request.session.clear()

    return RedirectResponse("/login", status_code=302)


# ============================================================
# 실행
# ============================================================

if __name__ == "__main__":

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=1439,
        reload=True
    )