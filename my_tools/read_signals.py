import requests
import json
import csv
from pathlib import Path

# 실제 GAE URL 입력
url = "https://vaulted-scholar-466013-r5.appspot.com/api/signals"

response = requests.get(url)
if response.status_code != 200:
    print(f"API 호출 실패: {response.text}")
    exit(1)

data = response.json()
items = data.get('items', [])

# 옵션 1: 전체 JSON 저장 (디렉토리 지정 가능)
output_dir = Path('repo_signals')  # 자유롭게 지정 (클라이언트 측)
output_dir.mkdir(exist_ok=True)
json_path = output_dir / 'signals_export.json'
with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=4, ensure_ascii=False)
print(f"JSON 저장 성공: {json_path} (항목 수: {len(items)})")

# 옵션 2: CSV로 변환 저장 (decision 플래튼 예시)
if items:
    fields = ['file', 'direction', 'entry', 'tp', 'sl', 'prob', 'risk_ok', 'rr']  # decision 키 기반 (코드에서 확인)
    rows = []
    for item in items:
        dec = item.get('decision', {})
        rows.append({
            'file': item.get('file', ''),
            'direction': dec.get('direction', ''),
            'entry': dec.get('entry', 0),
            'tp': dec.get('tp', 0),
            'sl': dec.get('sl', 0),
            'prob': dec.get('prob', 0.0),
            'risk_ok': dec.get('risk_ok', False),
            'rr': dec.get('rr', 0.0)
        })
    csv_path = output_dir / 'signals_export.csv'
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV 저장 성공: {csv_path}")