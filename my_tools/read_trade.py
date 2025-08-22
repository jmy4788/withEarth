import requests
import json
import csv
from pathlib import Path


# 실제 GAE URL 입력
url = "https://vaulted-scholar-466013-r5.appspot.com/api/trades?limit=2000"

response = requests.get(url)
if response.status_code != 200:
    print(f"API 호출 실패: {response.text}")
    exit(1)

data = response.json()
rows = data.get('rows', [])

# CSV 필드 (코드에서 정의된 순서)
fields = ['timestamp', 'symbol', 'side', 'qty', 'entry', 'tp', 'sl', 'exit', 'pnl', 'status', 'id']
output_dir = Path('repo_trades')  # 자유롭게 지정 (클라이언트 측)
output_dir.mkdir(exist_ok=True)
csv_path = output_dir / 'trades_export.csv'
# 로컬 파일 저장
with open(csv_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)

print(f"CSV 저장 성공: trades_export.csv (행 수: {len(rows)})")