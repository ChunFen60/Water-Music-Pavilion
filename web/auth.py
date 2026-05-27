import os
import hashlib
import sqlite3
from datetime import datetime, timezone, timedelta

DB_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "database"
)
DB_PATH = os.path.join(DB_DIR, "users.db")

# 北京时间
TZ = timezone(timedelta(hours=8))


def _get_conn():
    os.makedirs(DB_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS users ("
        "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
        "  username TEXT UNIQUE NOT NULL,"
        "  password_hash TEXT NOT NULL,"
        "  role TEXT NOT NULL DEFAULT 'user',"
        "  created_at TEXT NOT NULL DEFAULT (datetime('now'))"
        ")"
    )
    # 迁移：给旧表补上缺失的列
    cols = [r[1] for r in conn.execute("PRAGMA table_info(users)").fetchall()]
    if "role" not in cols:
        conn.execute("ALTER TABLE users ADD COLUMN role TEXT NOT NULL DEFAULT 'user'")
    if "created_at" not in cols:
        conn.execute("ALTER TABLE users ADD COLUMN created_at TEXT NOT NULL DEFAULT ''")
    conn.execute(
        "CREATE TABLE IF NOT EXISTS page_visits ("
        "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
        "  username TEXT DEFAULT NULL,"
        "  visited_at TEXT NOT NULL DEFAULT (datetime('now'))"
        ")"
    )
    conn.commit()
    return conn


def _hash_password(password, salt=None):
    if salt is None:
        salt = os.urandom(32)
    key = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        100000
    )
    return salt, key


def create_user(username, password):
    """Returns True on success, False if username already exists.
    如果用户名匹配环境变量 ADMIN_USERNAME，自动设为管理员。"""
    conn = _get_conn()
    try:
        salt, key = _hash_password(password)
        stored = salt.hex() + "$" + key.hex()
        admin_name = os.getenv("ADMIN_USERNAME", "")
        role = "admin" if admin_name and username == admin_name else "user"
        conn.execute(
            "INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)",
            (username, stored, role)
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()


def verify_user(username, password):
    """Returns True if credentials are valid."""
    conn = _get_conn()
    row = conn.execute(
        "SELECT password_hash FROM users WHERE username = ?",
        (username,)
    ).fetchone()
    conn.close()

    if row is None:
        return False

    stored = row[0]
    salt_hex, key_hex = stored.split("$")
    salt = bytes.fromhex(salt_hex)
    _, expected_key = _hash_password(password, salt)
    return expected_key.hex() == key_hex


def user_exists(username):
    conn = _get_conn()
    row = conn.execute(
        "SELECT 1 FROM users WHERE username = ?",
        (username,)
    ).fetchone()
    conn.close()
    return row is not None


# =========================
# 访问统计
# =========================

def log_visit(username=None):
    """记录一次页面访问"""
    conn = _get_conn()
    conn.execute(
        "INSERT INTO page_visits (username) VALUES (?)",
        (username,)
    )
    conn.commit()
    conn.close()


def get_stats():
    """返回统计数据"""
    conn = _get_conn()
    total_visits = conn.execute(
        "SELECT COUNT(*) FROM page_visits"
    ).fetchone()[0]
    total_users = conn.execute(
        "SELECT COUNT(*) FROM users"
    ).fetchone()[0]
    today = datetime.now(TZ).strftime("%Y-%m-%d")
    today_visits = conn.execute(
        "SELECT COUNT(*) FROM page_visits WHERE date(visited_at) = ?",
        (today,)
    ).fetchone()[0]
    logged_visits = conn.execute(
        "SELECT COUNT(*) FROM page_visits WHERE username IS NOT NULL"
    ).fetchone()[0]
    guest_visits = conn.execute(
        "SELECT COUNT(*) FROM page_visits WHERE username IS NULL"
    ).fetchone()[0]
    conn.close()
    return {
        "total_users": total_users,
        "total_visits": total_visits,
        "today_visits": today_visits,
        "logged_visits": logged_visits,
        "guest_visits": guest_visits,
    }


# =========================
# 管理员功能
# =========================

def is_admin(username):
    """检查用户是否为管理员"""
    conn = _get_conn()
    row = conn.execute(
        "SELECT role FROM users WHERE username = ?",
        (username,)
    ).fetchone()
    conn.close()
    return row is not None and row[0] == "admin"


def get_all_users():
    """返回所有注册用户列表 [(id, username, role, created_at), ...]"""
    conn = _get_conn()
    rows = conn.execute(
        "SELECT id, username, role, created_at FROM users ORDER BY id"
    ).fetchall()
    conn.close()
    return rows


def get_visit_logs(limit=100):
    """返回最近访问记录 [(id, username, visited_at), ...]"""
    conn = _get_conn()
    rows = conn.execute(
        "SELECT id, username, visited_at FROM page_visits "
        "ORDER BY id DESC LIMIT ?",
        (limit,)
    ).fetchall()
    conn.close()
    return rows


def get_user_visit_counts():
    """返回每个用户的访问次数 [(username, count), ...]"""
    conn = _get_conn()
    rows = conn.execute(
        "SELECT username, COUNT(*) as cnt FROM page_visits "
        "WHERE username IS NOT NULL "
        "GROUP BY username ORDER BY cnt DESC"
    ).fetchall()
    conn.close()
    return rows


def promote_to_admin(username):
    """将指定用户提升为管理员"""
    conn = _get_conn()
    conn.execute(
        "UPDATE users SET role = 'admin' WHERE username = ?",
        (username,)
    )
    conn.commit()
    conn.close()
