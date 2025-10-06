import requests, json
payload = {
  "closes": [float(i) for i in range(200)],
  "horizon_steps": 4,
  "dt_sec": 900,
  "freq": 0,
  "bracket": {"entry": 12345.0, "long": {"tp": 12360.0, "sl": 12310.0}, "short": {"tp": 12300.0, "sl": 12380.0}},
  "n_paths": 2000
}
headers = {"x-api-key": "7752dd5b7276827df423c0f454eaabcc89d629ab442dbf960cbb7cee9a4d8cfb"}
resp = requests.post(
    "https://tsfm-svc-185927152147.asia-northeast3.run.app/v1/prob_gate",
    json=payload,
    headers=headers,
    timeout=120,
)
print(resp.status_code)
print(resp.text[:1000])
