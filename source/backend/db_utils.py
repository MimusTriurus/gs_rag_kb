import sqlite3
import os

def init_db(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query TEXT NOT NULL,
            liked BOOLEAN NOT NULL
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS not_found_queries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

def insert_feedback(db_path, query, liked):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('INSERT INTO feedback (query, liked) VALUES (?, ?)', (query, liked))
    conn.commit()
    conn.close()

def insert_not_found_query(db_path, query):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('INSERT INTO not_found_queries (query) VALUES (?)', (query,))
    conn.commit()
    conn.close()

def get_all_feedback(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('SELECT query, liked FROM feedback')
    data = cursor.fetchall()
    conn.close()
    return data

def get_all_not_found_queries(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('SELECT query FROM not_found_queries')
    data = cursor.fetchall()
    conn.close()
    return data
