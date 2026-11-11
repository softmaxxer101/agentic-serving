import time, sys, urllib.request, csv, re
url, out = sys.argv[1], sys.argv[2]
w = csv.writer(open(out, 'w', buffering=1)); w.writerow(['ts', 'name', 'labels', 'value'])
while True:
    try: txt = urllib.request.urlopen(url, timeout=2).read().decode()
    except Exception:
        time.sleep(1); continue
    ts = time.time()
    for l in txt.splitlines():
        if l.startswith('vllm:') and '_bucket' not in l:
            m = re.match(r'([^{ ]+)(\{.*\})? (\S+)$', l)
            if m: w.writerow([ts, m.group(1), m.group(2) or '', m.group(3)])
    time.sleep(1)
