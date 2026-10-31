import json,sys
print(f"{'name':12s} {'rand med':>9s} {'p95':>7s} {'lg p95':>7s} | {'bub med':>8s} {'p95':>7s} {'lg p95':>7s} {'sec':>6s}")
for l in open('B_results.jsonl'):
    d=json.loads(l); r,b=d['random'],d['bubble']; print(f"{d['name']:12s} {r['l2_med']:9.4f} {r['l2_p95']:7.3f} {r['lg_p95']:7.2f} | {b['l2_med']:8.4f} {b['l2_p95']:7.3f} {b['lg_p95']:7.2f} {d['seconds']:6.0f}")
