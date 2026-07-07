# DB 서버 세팅 가이드

전체 구성: GPU 서버 1대에 PostgreSQL + db_server.py를 올리고, 여러 검사/리뷰 PC에서 접속.

```
[검사 PC 1] app_front ──┐
[검사 PC 2] app_front ──┤──→ [GPU 서버] db_server.py:8001 → PostgreSQL:5432
[리뷰 PC 1] app_back  ──┤
[리뷰 PC 2] app_back  ──┘
```

---

## 1. GPU 서버 — PostgreSQL 설치

### Ubuntu / Debian

```bash
sudo apt update
sudo apt install postgresql postgresql-contrib -y
sudo systemctl start postgresql
sudo systemctl enable postgresql
```

### macOS (로컬 테스트용)

```bash
brew install postgresql@15
brew services start postgresql@15
echo 'export PATH="/opt/homebrew/opt/postgresql@15/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
```

---

## 2. GPU 서버 — DB / 유저 생성

```bash
sudo -u postgres psql <<EOF
CREATE USER pcb WITH PASSWORD 'pcb1234';
CREATE DATABASE pcb_inspection OWNER pcb;
GRANT ALL PRIVILEGES ON DATABASE pcb_inspection TO pcb;
EOF
```

접속 확인:

```bash
psql -U pcb -d pcb_inspection -h localhost
# 접속되면 \q 로 종료
```

---

## 3. GPU 서버 — 외부 접속 허용 (다른 PC에서 쓸 경우)

### postgresql.conf 수정

```bash
sudo nano /etc/postgresql/*/main/postgresql.conf
```

아래 줄을 찾아 수정:

```
listen_addresses = '*'
```

### pg_hba.conf 수정

```bash
sudo nano /etc/postgresql/*/main/pg_hba.conf
```

파일 맨 아래에 추가:

```
host  all  all  0.0.0.0/0  md5
```

### 재시작

```bash
sudo systemctl restart postgresql
```

> macOS는 `brew services restart postgresql@15`

---

## 4. GPU 서버 — 프로젝트 세팅

```bash
pip install psycopg2-binary
```

`config.yaml` 확인 (기본값 그대로 사용):

```yaml
database:
  url: postgresql://pcb:pcb1234@localhost:5432/pcb_inspection
```

테이블 생성:

```bash
python -m db.init_db
```

성공 시 출력:

```
[DB] 초기화 완료 → postgresql://pcb:***@localhost:5432/pcb_inspection
```

---

## 5. GPU 서버 — DB API 서버 실행

```bash
python -m db.db_server
```

출력:

```
INFO:     Started server process
INFO:     Uvicorn running on http://0.0.0.0:8001
```

방화벽이 있으면 포트 허용:

```bash
sudo ufw allow 8001
```

---

## 6. 검사 / 리뷰 PC — config.yaml 수정

각 PC의 프로젝트 `config.yaml`에서 GPU 서버 IP로 변경:

```yaml
database:
  url: postgresql://pcb:pcb1234@<GPU서버IP>:5432/pcb_inspection

db_server:
  client_url: http://<GPU서버IP>:8001
```

패키지 설치:

```bash
pip install psycopg2-binary
```

---

## 7. 실행 순서

| 순서 | PC | 명령 |
|------|-----|------|
| 1 | GPU 서버 | `python -m db.db_server` |
| 2 | 검사 PC | `python app_front/main.py` |
| 3 | 리뷰 PC | `python app_back/main.py` → Ctrl+D |

---

## 8. 같은 PC에서 테스트할 경우

모든 앱을 한 PC에서 실행할 때는 `config.yaml`을 기본값(localhost)으로 두고:

```bash
# 터미널 1
python -m db.db_server

# 터미널 2
python app_front/main.py

# 터미널 3
python app_back/main.py
```

---

## 9. 환경변수로 DB URL 지정 (선택)

`config.yaml` 대신 환경변수로 오버라이드 가능:

```bash
export DATABASE_URL="postgresql://pcb:pcb1234@192.168.1.10:5432/pcb_inspection"
python -m db.db_server
```

---

## 10. DB 접속 방법

### psql CLI (터미널)

```bash
# 로컬 접속
psql -U pcb -d pcb_inspection -h localhost

# 원격 접속
psql -U pcb -d pcb_inspection -h <GPU서버IP>
```

유용한 psql 명령:

```sql
\dt                          -- 테이블 목록
\d inspection_sessions       -- 테이블 컬럼 구조

SELECT * FROM inspection_sessions ORDER BY started_at DESC LIMIT 5;
SELECT verdict, COUNT(*) FROM tile_inspections GROUP BY verdict;
SELECT * FROM reviews ORDER BY reviewed_at DESC LIMIT 10;
```

---

### pgAdmin 4 (GUI — Windows/macOS)

1. [pgAdmin 공식 사이트](https://www.pgadmin.org/download/)에서 설치
2. 실행 후 **Add New Server**
3. 입력값:

| 항목 | 값 |
|------|-----|
| Name | pcb-inspection (임의) |
| Host | GPU 서버 IP |
| Port | 5432 |
| Database | pcb_inspection |
| Username | pcb |
| Password | pcb1234 |

---

### DBeaver (GUI — 무료, 범용)

1. [DBeaver 다운로드](https://dbeaver.io/download/)
2. **New Database Connection** → PostgreSQL 선택
3. 위 pgAdmin과 동일한 정보 입력
4. **Test Connection** → 성공 확인 후 저장

---

### DB API 서버 헬스체크 (브라우저 / curl)

db_server.py가 실행 중인지 확인:

```bash
# 로컬
curl http://localhost:8001/health

# 원격
curl http://<GPU서버IP>:8001/health
```

응답: `{"status": "ok"}`

브라우저에서 `http://<GPU서버IP>:8001/docs` 접속하면 Swagger UI로 API 전체 확인 가능.

---

### Python에서 직접 조회

```python
from db.api_client import client

# 서버 URL 변경 (필요 시)
client.set_server_url("http://<GPU서버IP>:8001")

# FAIL/REVIEW 타일 조회
tiles = client.fetch_new_tiles(verdict="FAIL,REVIEW")
for t in tiles:
    print(t["verdict"], t["row"], t["col"], t["detections"])
```

---

## 문제 해결

| 증상 | 원인 | 해결 |
|------|------|------|
| `could not connect to server` | PostgreSQL 미실행 | `sudo systemctl start postgresql` |
| `password authentication failed` | 비밀번호 불일치 | pg_hba.conf 확인, `ALTER USER pcb PASSWORD 'pcb1234';` |
| `connection refused` 외부에서 | listen_addresses 미설정 | 3번 항목 재확인 후 재시작 |
| `psycopg2` import 오류 | 드라이버 미설치 | `pip install psycopg2-binary` |
