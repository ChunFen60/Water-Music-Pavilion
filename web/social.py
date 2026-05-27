import os
import base64
import sqlite3
import random
from datetime import datetime, timezone, timedelta

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_DIR = os.path.join(BASE_DIR, "database")
DB_PATH = os.path.join(DB_DIR, "social.db")
TZ = timezone(timedelta(hours=8))
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB


def _get_conn():
    os.makedirs(DB_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS friend_requests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            from_user TEXT NOT NULL,
            to_user TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        );
        CREATE TABLE IF NOT EXISTS friendships (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user1 TEXT NOT NULL,
            user2 TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        );
        CREATE TABLE IF NOT EXISTS private_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            from_user TEXT NOT NULL,
            to_user TEXT NOT NULL,
            content TEXT NOT NULL,
            sent_at TEXT NOT NULL DEFAULT (datetime('now'))
        );
        CREATE TABLE IF NOT EXISTS chat_groups (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            group_code TEXT UNIQUE NOT NULL,
            group_name TEXT NOT NULL,
            created_by TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        );
        CREATE TABLE IF NOT EXISTS group_members (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            group_code TEXT NOT NULL,
            username TEXT NOT NULL,
            joined_at TEXT NOT NULL DEFAULT (datetime('now'))
        );
        CREATE TABLE IF NOT EXISTS group_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            group_code TEXT NOT NULL,
            from_user TEXT NOT NULL,
            content TEXT NOT NULL,
            sent_at TEXT NOT NULL DEFAULT (datetime('now'))
        );
        CREATE TABLE IF NOT EXISTS shared_files (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            group_code TEXT NOT NULL,
            from_user TEXT NOT NULL,
            file_name TEXT NOT NULL,
            file_type TEXT NOT NULL,
            file_size INTEGER NOT NULL DEFAULT 0,
            file_data TEXT NOT NULL,
            uploaded_at TEXT NOT NULL DEFAULT (datetime('now'))
        );
    """)
    conn.commit()
    return conn


# =========================
# Friend Requests
# =========================

def send_friend_request(from_user, to_user):
    if from_user == to_user:
        return False, "You cannot add yourself as a friend."

    conn = _get_conn()
    existing = conn.execute(
        "SELECT 1 FROM friendships WHERE (user1=? AND user2=?) OR (user1=? AND user2=?)",
        (from_user, to_user, to_user, from_user)
    ).fetchone()
    if existing:
        conn.close()
        return False, "You are already friends."

    pending = conn.execute(
        "SELECT id FROM friend_requests WHERE from_user=? AND to_user=? AND status='pending'",
        (from_user, to_user)
    ).fetchone()
    if pending:
        conn.close()
        return False, "A pending friend request already exists."

    reverse = conn.execute(
        "SELECT id FROM friend_requests WHERE from_user=? AND to_user=? AND status='pending'",
        (to_user, from_user)
    ).fetchone()
    if reverse:
        conn.close()
        return False, "This user already sent you a friend request. Check your pending requests."

    conn.execute(
        "INSERT INTO friend_requests (from_user, to_user) VALUES (?, ?)",
        (from_user, to_user)
    )
    conn.commit()
    conn.close()
    return True, "Friend request sent!"


def get_pending_requests(username):
    conn = _get_conn()
    rows = conn.execute(
        "SELECT id, from_user, created_at FROM friend_requests WHERE to_user=? AND status='pending' ORDER BY id DESC",
        (username,)
    ).fetchall()
    conn.close()
    return rows


def get_sent_requests(username):
    conn = _get_conn()
    rows = conn.execute(
        "SELECT id, to_user, status, created_at FROM friend_requests WHERE from_user=? ORDER BY id DESC LIMIT 30",
        (username,)
    ).fetchall()
    conn.close()
    return rows


def accept_friend_request(request_id, username):
    conn = _get_conn()
    req = conn.execute(
        "SELECT from_user, to_user FROM friend_requests WHERE id=? AND status='pending'",
        (request_id,)
    ).fetchone()
    if not req:
        conn.close()
        return False, "Request not found or already processed."
    if req[1] != username:
        conn.close()
        return False, "You cannot accept this request."

    conn.execute("UPDATE friend_requests SET status='accepted' WHERE id=?", (request_id,))
    u1, u2 = sorted([req[0], req[1]])
    conn.execute("INSERT INTO friendships (user1, user2) VALUES (?, ?)", (u1, u2))
    conn.commit()
    conn.close()
    return True, f"You are now friends with {req[0]}!"


def reject_friend_request(request_id, username):
    conn = _get_conn()
    req = conn.execute(
        "SELECT to_user FROM friend_requests WHERE id=? AND status='pending'",
        (request_id,)
    ).fetchone()
    if not req:
        conn.close()
        return False, "Request not found or already processed."
    if req[0] != username:
        conn.close()
        return False, "You cannot reject this request."

    conn.execute("UPDATE friend_requests SET status='rejected' WHERE id=?", (request_id,))
    conn.commit()
    conn.close()
    return True, "Friend request rejected."


# =========================
# Friends
# =========================

def get_friends(username):
    conn = _get_conn()
    rows = conn.execute(
        "SELECT user1, user2 FROM friendships WHERE user1=? OR user2=?",
        (username, username)
    ).fetchall()
    conn.close()
    friends = []
    for u1, u2 in rows:
        friend = u2 if u1 == username else u1
        friends.append(friend)
    return sorted(friends)


def are_friends(user1, user2):
    conn = _get_conn()
    u1, u2 = sorted([user1, user2])
    row = conn.execute(
        "SELECT 1 FROM friendships WHERE user1=? AND user2=?",
        (u1, u2)
    ).fetchone()
    conn.close()
    return row is not None


# =========================
# Private Messages
# =========================

def send_private_message(from_user, to_user, content):
    if not content.strip():
        return False
    conn = _get_conn()
    conn.execute(
        "INSERT INTO private_messages (from_user, to_user, content) VALUES (?, ?, ?)",
        (from_user, to_user, content.strip())
    )
    conn.commit()
    conn.close()
    return True


def get_private_messages(user1, user2, limit=50):
    conn = _get_conn()
    rows = conn.execute(
        """SELECT from_user, content, sent_at
           FROM private_messages
           WHERE (from_user=? AND to_user=?) OR (from_user=? AND to_user=?)
           ORDER BY id ASC LIMIT ?""",
        (user1, user2, user2, user1, limit)
    ).fetchall()
    conn.close()
    return rows


# =========================
# Groups
# =========================

def _generate_group_code():
    conn = _get_conn()
    for _ in range(100):
        code = str(random.randint(10000, 99999))
        exists = conn.execute("SELECT 1 FROM chat_groups WHERE group_code=?", (code,)).fetchone()
        if not exists:
            conn.close()
            return code
    conn.close()
    return None


def create_group(group_name, created_by):
    if not group_name.strip():
        return None, "Group name cannot be empty."

    code = _generate_group_code()
    if code is None:
        return None, "Failed to generate group code. Please try again."

    conn = _get_conn()
    conn.execute(
        "INSERT INTO chat_groups (group_code, group_name, created_by) VALUES (?, ?, ?)",
        (code, group_name.strip(), created_by)
    )
    conn.execute(
        "INSERT INTO group_members (group_code, username) VALUES (?, ?)",
        (code, created_by)
    )
    conn.commit()
    conn.close()
    return code, f"Group '{group_name.strip()}' created! Share this code: {code}"


def join_group(group_code, username):
    conn = _get_conn()
    group = conn.execute(
        "SELECT group_name FROM chat_groups WHERE group_code=?",
        (group_code,)
    ).fetchone()
    if not group:
        conn.close()
        return False, "Group not found. Check the code and try again."

    member = conn.execute(
        "SELECT 1 FROM group_members WHERE group_code=? AND username=?",
        (group_code, username)
    ).fetchone()
    if member:
        conn.close()
        return False, "You are already a member of this group."

    conn.execute(
        "INSERT INTO group_members (group_code, username) VALUES (?, ?)",
        (group_code, username)
    )
    conn.commit()
    conn.close()
    return True, f"Joined group '{group[0]}'!"


def is_group_member(group_code, username):
    conn = _get_conn()
    row = conn.execute(
        "SELECT 1 FROM group_members WHERE group_code=? AND username=?",
        (group_code, username)
    ).fetchone()
    conn.close()
    return row is not None


def get_my_groups(username):
    conn = _get_conn()
    rows = conn.execute(
        """SELECT g.group_code, g.group_name, g.created_by, g.created_at
           FROM chat_groups g
           JOIN group_members m ON g.group_code = m.group_code
           WHERE m.username = ?
           ORDER BY g.id DESC""",
        (username,)
    ).fetchall()
    conn.close()
    return rows


def get_group_info(group_code):
    conn = _get_conn()
    row = conn.execute(
        "SELECT group_code, group_name, created_by, created_at FROM chat_groups WHERE group_code=?",
        (group_code,)
    ).fetchone()
    conn.close()
    return row


def get_group_members(group_code):
    conn = _get_conn()
    rows = conn.execute(
        "SELECT username, joined_at FROM group_members WHERE group_code=? ORDER BY id ASC",
        (group_code,)
    ).fetchall()
    conn.close()
    return rows


def search_groups_by_name(keyword):
    conn = _get_conn()
    rows = conn.execute(
        "SELECT group_code, group_name, created_by FROM chat_groups WHERE group_name LIKE ? ORDER BY id DESC LIMIT 20",
        (f"%{keyword}%",)
    ).fetchall()
    conn.close()
    return rows


# =========================
# Group Messages
# =========================

def send_group_message(group_code, from_user, content):
    if not content.strip():
        return False
    conn = _get_conn()
    conn.execute(
        "INSERT INTO group_messages (group_code, from_user, content) VALUES (?, ?, ?)",
        (group_code, from_user, content.strip())
    )
    conn.commit()
    conn.close()
    return True


def get_group_messages(group_code, limit=50):
    conn = _get_conn()
    rows = conn.execute(
        "SELECT from_user, content, sent_at FROM group_messages WHERE group_code=? ORDER BY id ASC LIMIT ?",
        (group_code, limit)
    ).fetchall()
    conn.close()
    return rows


# =========================
# File Sharing
# =========================

def upload_file(group_code, from_user, file_name, file_type, file_bytes):
    if len(file_bytes) > MAX_FILE_SIZE:
        return False, "File size exceeds 5MB limit."

    file_data_b64 = base64.b64encode(file_bytes).decode("utf-8")
    conn = _get_conn()
    conn.execute(
        "INSERT INTO shared_files (group_code, from_user, file_name, file_type, file_size, file_data) VALUES (?, ?, ?, ?, ?, ?)",
        (group_code, from_user, file_name, file_type, len(file_bytes), file_data_b64)
    )
    conn.commit()
    conn.close()
    return True, f"File '{file_name}' uploaded successfully!"


def get_shared_files(group_code):
    conn = _get_conn()
    rows = conn.execute(
        "SELECT id, from_user, file_name, file_type, file_size, uploaded_at FROM shared_files WHERE group_code=? ORDER BY id DESC",
        (group_code,)
    ).fetchall()
    conn.close()
    return rows


def get_file(file_id):
    conn = _get_conn()
    row = conn.execute(
        "SELECT file_name, file_type, file_data FROM shared_files WHERE id=?",
        (file_id,)
    ).fetchone()
    conn.close()
    if not row:
        return None
    return row[0], row[1], base64.b64decode(row[2])
