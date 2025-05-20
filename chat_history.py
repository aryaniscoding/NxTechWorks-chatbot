import sqlite3
from datetime import datetime

DB = "chat_history.db"

def init_history_db():
    conn = sqlite3.connect(DB)
    conn.execute('''
      CREATE TABLE IF NOT EXISTS history (
        id INTEGER PRIMARY KEY,
        session_id TEXT, timestamp TEXT,
        role TEXT, message TEXT
      )''')
    conn.commit(); conn.close()

def save_message(sid, role, msg):
    conn = sqlite3.connect(DB)
    conn.execute(
      "INSERT INTO history(session_id,timestamp,role,message) VALUES(?,?,?,?)",
      (sid, datetime.utcnow().isoformat(), role, msg)
    )
    conn.commit(); conn.close()

def load_history(sid):
    conn = sqlite3.connect(DB)
    cur = conn.execute(
      "SELECT timestamp,role,message FROM history WHERE session_id=? ORDER BY timestamp",
      (sid,)
    )
    rows = cur.fetchall(); conn.close()
    return [(t,r,m) for t,r,m in rows]
