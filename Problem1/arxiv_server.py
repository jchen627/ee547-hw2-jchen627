#!/usr/bin/env python3
import http.server
import urllib.parse
import json
import re
import sys
import os
import datetime

# 加载数据
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "sample_data", "papers.json")
STATS_FILE = os.path.join(BASE_DIR, "sample_data", "corpus_analysis.json")

try:
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        PAPERS = json.load(f)
except Exception:
    PAPERS = []

try:
    with open(STATS_FILE, "r", encoding="utf-8") as f:
        STATS = json.load(f)
except Exception:
    STATS = {}

# 根据 ID 查找论文
def find_paper(arxiv_id):
    for p in PAPERS:
        if p.get("arxiv_id") == arxiv_id:
            return p
    return None

# 搜索功能
def search_papers(query):
    terms = re.findall(r"\w+", query.lower())
    results = []
    for paper in PAPERS:
        title = paper.get("title", "").lower()
        abstract = paper.get("abstract", "").lower()
        score = 0
        matches_in = []
        for term in terms:
            if term in title:
                score += title.count(term)
                matches_in.append("title")
            if term in abstract:
                score += abstract.count(term)
                matches_in.append("abstract")
        if score > 0:
            results.append({
                "arxiv_id": paper.get("arxiv_id"),
                "title": paper.get("title"),
                "match_score": score,
                "matches_in": list(set(matches_in))
            })
    return {"query": query, "results": results}

# 自定义 HTTP 处理类
class ArxivHandler(http.server.BaseHTTPRequestHandler):
    def _send_json(self, data, status=200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data, indent=2).encode("utf-8"))

        # 日志输出
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{now}] {self.command} {self.path} - {status} {self.responses[status][0]}")

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path_parts = parsed.path.strip("/").split("/")
        query = urllib.parse.parse_qs(parsed.query)

        try:
            if self.path == "/papers":
                self._send_json(PAPERS)

            elif len(path_parts) == 2 and path_parts[0] == "papers":
                paper = find_paper(path_parts[1])
                if paper:
                    self._send_json(paper)
                else:
                    self._send_json({"error": "Paper not found"}, status=404)

            elif path_parts[0] == "search":
                if "q" not in query:
                    self._send_json({"error": "Missing query"}, status=400)
                else:
                    result = search_papers(query["q"][0])
                    self._send_json(result)

            elif self.path == "/stats":
                self._send_json(STATS)

            else:
                self._send_json({"error": "Invalid endpoint"}, status=404)

        except Exception as e:
            self._send_json({"error": str(e)}, status=500)

# 启动服务
if __name__ == "__main__":
    port = 8080
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            print("Error: Port must be numeric")
            sys.exit(1)

    server = http.server.HTTPServer(("0.0.0.0", port), ArxivHandler)
    print(f"Server running on http://localhost:{port}")
    server.serve_forever()
