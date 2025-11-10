"""
====================================================================
AI Chatbot for Database Interaction 
Author: PALANIVEL M
Model: Gemini 2.5 Flash Lite  (models/gemini-2.5-flash-lite) Future Will Be Update New Models
Purpose:
    - Natural-language questions → auto-generated SQL → executed safely
    - Works with a demo SQLite DB or user-provided MSSQL credentials
    - Adds clear LLM interpretation and equivalent queries (MySQL/Oracle)
====================================================================
"""

# ===============================================================
# AI Chatbot for Database — Imports & Global Setup
# ===============================================================
# 📌 Purpose:
#   Centralized imports used across the app. We keep imports explicit for
#   readability and predictable packaging. Nothing dynamic here.

import os
import gc
import re
import torch
import streamlit as st
import google.api_core.exceptions as google_exceptions

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.exc import ProgrammingError

# LangChain & friends
from langchain.chains import create_sql_query_chain
from langchain_community.utilities import SQLDatabase
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.prompts import SemanticSimilarityExampleSelector
from langchain.prompts import FewShotPromptTemplate
from langchain.prompts.prompt import PromptTemplate
from langchain.chains.sql_database.prompt import PROMPT_SUFFIX

# Extra UI (kept for parity)
from streamlit_extras.stylable_container import stylable_container


# ===============================================================
# Environment & Page Setup
# ===============================================================
# 📌 Purpose:
#   - Load .env (for admin key or optional MSSQL env defaults).
#   - Configure Streamlit page meta.
#   - Read initial MSSQL vars (used as defaults when "Connect your Own Database").

load_dotenv()

st.set_page_config(
    page_title="AI Chatbot for Database 😍",
    page_icon="🗣️",
    layout="wide"
)

# ===============================================================
# Gemini Connection — Safe, Lazy Initialization
# ===============================================================
# 📌 Purpose:
#   - Provide two ways to initialize Gemini:
#       1) Admin key from environment
#       2) User key from sidebar
#   - Never block the app if the model isn't connected yet.
#   - Track connection state in st.session_state without leaking secrets.

gemini_model = "models/gemini-2.5-flash-lite"  # fixed model

def _try_init_gemini(api_key: str):
    """
    Try to initialize the Google Generative AI client with the provided key.

    Returns:
        (llm, err)
    """
    try:
        from langchain_google_genai import GoogleGenerativeAI
        import google.api_core.exceptions as google_exceptions

        # ✅ Explicitly set the API key environment (needed in Streamlit Cloud)
        os.environ["GOOGLE_API_KEY"] = api_key

        # Initialize model
        _llm = GoogleGenerativeAI(
            model=gemini_model,
            google_api_key=api_key,
            temperature=0.1,
            max_output_tokens=1024,
        )

        # ✅ Test request to confirm it works
        try:
            _ = _llm.invoke("ping")
        except Exception as test_err:
            if "API key not valid" in str(test_err):
                return None, "invalid"
            raise test_err

        return _llm, None

    except google_exceptions.ResourceExhausted:
        return None, "rate_limit"
    except google_exceptions.Unauthenticated:
        return None, "invalid"
    except google_exceptions.ServiceUnavailable:
        return None, "unavailable"
    except Exception as e:
        st.sidebar.error(f"Gemini init error: {str(e)}")  # ✅ show the real reason
        if "API key not valid" in str(e) or "INVALID_ARGUMENT" in str(e):
            return None, "invalid"
        return None, "other"


# ===============================================================
# Session State Initialization
# ===============================================================
# 📌 Purpose:
#   Keep runtime state across reruns without writing to disk.
#   We only store ephemeral, non-sensitive flags (or user-provided key in memory).

if "gemini_llm" not in st.session_state:
    st.session_state.gemini_llm = None   # holds the active LLM object
if "llm_source" not in st.session_state:
    st.session_state.llm_source = None   # one of {"admin", "user", None}
if "gemini_status" not in st.session_state:
    st.session_state.gemini_status = "disconnected"  # {"connected", "error", "disconnected"}


# ===============================================================
# Sidebar — Admin Key Connect
# ===============================================================
# 📌 Purpose:
#   - Optionally initialize Gemini using an admin key from .env.
#   - Non-blocking. If absent or invalid, the user can still use their key.

st.sidebar.subheader("🧠 LLM Configuration (admin key)")

try:
    admin_key = st.secrets.get("GOOGLE_API_KEY", "").strip()
except Exception:
    admin_key = os.getenv("GOOGLE_API_KEY", "").strip()

def connect_admin():
    """
    Attempt to connect Gemini using the admin key from the environment.
    On success:
        - st.session_state.gemini_llm is set
        - st.session_state.llm_source = "admin"
        - st.session_state.gemini_status = "connected"
    On failure:
        - gemini_status becomes "error" and a contextual message is shown.
    """
    if not admin_key:
        st.sidebar.warning("No admin key found in environment.")
        st.session_state.gemini_status = "error"
        return

    llm_try, err = _try_init_gemini(admin_key)
    if llm_try:
        st.session_state.gemini_llm = llm_try
        st.session_state.llm_source = "admin"
        st.session_state.gemini_status = "connected"
    else:
        st.session_state.gemini_status = "error"
        if err == "rate_limit":
            st.sidebar.warning("⏳ Model reached free-tier limit. Please wait.")
        elif err == "invalid":
            st.sidebar.error("❌ Admin API key invalid.")
        elif err == "unavailable":
            st.sidebar.error("⚠️ Gemini service temporarily unavailable.")
        else:
            st.sidebar.error("⚠️ Could not initialize Gemini.")

# Button to connect with admin key
if st.sidebar.button("▶️ Connect (Admin Key)"):
    connect_admin()

# Show small persistent label if admin key is active
if st.session_state.llm_source == "admin" and st.session_state.gemini_llm:
    with st.sidebar.expander("✅ Admin Key Active", expanded=True):
        st.markdown(f"**🧩 Using model:** `{gemini_model}`")
        st.info("💡 You can choose alternative models with your own Gemini API key using the below connection.")

# ===============================================================
# App Title & Subtitle — (UI only, no logic)
# ===============================================================
# 📌 Purpose:
#   Professional first impression + concise tagline.
#   Pure CSS/HTML for visual polish; safe and isolated from logic.

st.markdown("""
    <style>
        .main-title {
            text-align: center;
            font-size: 2.5rem;
            font-weight: 800;
            background: linear-gradient(90deg, #a5b4fc, #818cf8, #6366f1);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-top: 0;
            margin-bottom: 0.5rem;
            letter-spacing: 0.5px;
        }
        .subtitle {
            text-align: center;
            color: #cbd5e1;
            font-size: 1.1rem;
            font-weight: 400;
            margin-bottom: 2rem;
        }
        hr.custom-line {
            border: none;
            height: 1px;
            background: linear-gradient(90deg, rgba(99,102,241,0.2), rgba(255,255,255,0.1), rgba(99,102,241,0.2));
            margin-bottom: 2rem;
        }
    </style>

    <h1 class="main-title">🧠 Dynamic Database Interaction AI Chatbot</h1>
    <p class="subtitle">Query, analyze, and explore your database using everyday English</p>
    <hr class="custom-line">
""", unsafe_allow_html=True)


# ===============================================================
# Sidebar — User Key Connect / Clear
# ===============================================================
# 📌 Purpose:
#   - Allow user-provided Gemini key (temporary, in-memory only)
#   - Automatically reconnect if model is changed
#   - "Clear Keys" resets everything cleanly

st.sidebar.subheader("🧠 LLM Configuration (user key)")

# Input field for Gemini API key
user_key = st.sidebar.text_input(
    "Enter your Google Gemini API Key:",
    type="password",
    key="__user_gemini_key",
    placeholder="Paste your Gemini API key here...",
)

# --- Connect with user key ---
if st.sidebar.button("▶️ Connect (My Key)"):
    if not user_key.strip():
        st.sidebar.warning("Please enter your Gemini API key first.")
    else:
        llm_try, err = _try_init_gemini(user_key.strip())
        if llm_try:
            st.session_state.gemini_llm = llm_try
            st.session_state.llm_source = "user"
            st.session_state.gemini_status = "connected"
            st.session_state.user_key_connected = True
            st.session_state.user_gemini_api_key = user_key.strip()
            
        else:
            st.session_state.gemini_status = "error"
            st.session_state.user_key_connected = False
            if err == "rate_limit":
                st.sidebar.warning("⏳ Free-tier limit reached. Please wait.")
            elif err == "invalid":
                st.sidebar.error("❌ Invalid key. Please check and try again.")
            elif err == "unavailable":
                st.sidebar.error("⚠️ Gemini temporarily unavailable.")
            else:
                st.sidebar.error("⚠️ Could not initialize with your key.")


# --- Model dropdown (visible only when connected successfully) ---
if st.session_state.get("user_key_connected", False):
    st.markdown("<br>", unsafe_allow_html=True)
    st.sidebar.markdown("#### 🧠 Choose Gemini Model")

    model_options = [
        "models/gemini-2.5-pro",
        "models/gemini-2.5-flash",
        "models/gemini-2.5-flash-lite",
    ]

    selected_model = st.sidebar.selectbox(
        "Choose Gemini Model",
        model_options,
        index=model_options.index(
            st.session_state.get("selected_gemini_model", "models/gemini-2.5-flash-lite")
        ) if st.session_state.get("selected_gemini_model") in model_options else 2,
        help="Pick an advanced Gemini model to use with your own key.",
        key="user_model_selector",
    )

    # Auto-reconnect if model changes
    if (
        "selected_gemini_model" not in st.session_state
        or st.session_state["selected_gemini_model"] != selected_model
    ):
        st.session_state["selected_gemini_model"] = selected_model

        # Reconnect automatically with the selected model
        try:
            from langchain_google_genai import GoogleGenerativeAI
            llm_new = GoogleGenerativeAI(
                model=selected_model,
                google_api_key=st.session_state.user_gemini_api_key,
                temperature=0.1,
                max_output_tokens=1024,
            )
            # Ping to verify connection
            _ = llm_new.invoke("ping")
            st.session_state.gemini_llm = llm_new
        except Exception as e:
            st.sidebar.error(f"⚠️ Could not switch model: {e}")

    st.sidebar.info(f"🧩 Currently using: `{st.session_state['selected_gemini_model']}`")

# ===============================================================
# Fallback Display (Non-blocking)
# ===============================================================
# 📌 Purpose:
#   If Gemini is not connected yet, keep the app usable and tell the user how
#   to proceed. We do NOT stop the app here (non-blocking by design).

if st.session_state.gemini_llm is None:
    if st.session_state.gemini_status == "error":
        st.warning("Please connect Gemini using a valid API key (admin or your own).")
    else:
        st.info("▶️ Click 'Connect' in the sidebar to initialize Gemini.")

# Expose LLM object for downstream usage
llm = st.session_state.gemini_llm

if llm is None:
    st.sidebar.warning("⚠️ Gemini not connected yet — please connect using Admin or User key.")
else:
    st.sidebar.success("✅ Gemini is active and ready.")

# --- Single Clean Reset ---
if st.sidebar.button("🗑️ Clear Keys"):
    for key in list(st.session_state.keys()):
        if "gemini" in key or "__user_gemini" in key or "llm" in key:
            del st.session_state[key]
    st.session_state["__user_gemini_key"] = ""
    st.session_state.user_key_connected = False
    st.sidebar.success("✅ All Gemini connections and keys cleared. Please reconnect using Admin or User key.")
    st.rerun()


# ===============================================================
# Database Connection: Demo (SQLite), MSSQL, or MySQL
# ===============================================================
# 📌 Purpose:
#   - Allow switching between a safe bundled demo, MSSQL, or MySQL instance.
#   - Demo DB is created locally if not present.
#   - User credentials are stored only in session_state (ephemeral).

db = None
st.sidebar.subheader("🗄️ Database Connection Options")

connection_mode = st.sidebar.radio(
    "Choose database connection mode:",
    ("Use Demo Database", "Connect your Own Databases(MySQL/MSSQL)"),
)

# ===============================================================
# 🧠 Auto-Detect Database Type (SQLite / MSSQL / MySQL)
# ===============================================================
if connection_mode == "Use Demo Database":
    db_type = "SQLite"
else:
    db_choice = st.sidebar.selectbox(
        "🧩 Select your database type:",
        ("MSSQL", "MySQL"),
        help="Choose which type of database you're connecting to. The LLM prompt and SQL syntax will adapt automatically."
    )
    db_type = db_choice

# Show the detected or chosen type in sidebar
st.sidebar.caption(f"🧠 Prompt optimized for: `{db_type}` SQL syntax")

# ===============================================================
# DEMO DATABASE (SQLite)
# ===============================================================
if connection_mode == "Use Demo Database":
    st.sidebar.info("✅ Connected to built-in demo database (no credentials required).")

    import sqlite3
    demo_db_path = "demo_database.db"

    if not os.path.exists(demo_db_path):
        conn = sqlite3.connect(demo_db_path)
        cursor = conn.cursor()

        # Create schema
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS employees (
            employee_id INTEGER PRIMARY KEY,
            staff_name TEXT,
            department INTEGER,
            salary REAL,
            coundry TEXT,
            Location_ID INTEGER
        );
        """)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS department (
            depart_id INTEGER PRIMARY KEY,
            depart_name TEXT
        );
        """)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS LOCATION (
            Location_ID INTEGER PRIMARY KEY,
            City TEXT
        );
        """)

        # Seed sample data
        cursor.executescript("""
        INSERT INTO department VALUES 
            (1, 'data_scientist'), (2, 'data_analyst'), (3, 'data_enginer');
        INSERT INTO LOCATION VALUES 
            (122, 'Salem'), (123, 'Dallas'), (124, 'Chicago');
        INSERT INTO employees VALUES 
            (1, 'suresh', 1, 18000, 'INDIA', 122),
            (2, 'ajith', 2, 19000, 'USA', 123),
            (3, 'arul', 3, 25000, 'CANADA', 124),
            (4, 'rohesh', 1, 10000, 'USA', 123),
            (5, 'dinesh', 1, 15000, 'INDIA', 122),
            (6, 'rahul', 2, 11000, 'USA', 167);
        """)
        conn.commit()
        conn.close()

    engine = create_engine(f"sqlite:///{demo_db_path}")
    db = SQLDatabase(engine, sample_rows_in_table_info=3)

    st.markdown("### 🧩 Demo Database Overview")
    st.info("""
        This demo database helps you understand how the chatbot works.
        It contains **employees**, **department**, and **location** tables linked through foreign keys.

        You can ask natural language questions like:
        - List all employees and their departments.
        - Who earns the highest salary?
        - Show all employees working outside India.
        - Find the average salary of data analysts.
        - List employees with salary above 15000 and their cities.
    """)

    with st.expander("📊 View Table Details"):
        import pandas as pd
        conn = sqlite3.connect(demo_db_path)
        for table in ["employees", "department", "LOCATION"]:
            st.markdown(f"#### {table}")
            df = pd.read_sql_query(f"SELECT * FROM {table} LIMIT 5;", conn)
            st.dataframe(df)
        conn.close()

# ===============================================================
# USER DATABASE (MSSQL / MySQL)
# ===============================================================
else:
    st.sidebar.info(f"🔐 Enter your {db_type} credentials below:")

    # Initialize session_state keys
    for key in ["db_host", "db_name", "db_user", "db_password", "db_connected", "db_error"]:
        if key not in st.session_state:
            st.session_state[key] = "" if not key.endswith("_connected") else False

    st.session_state.db_host = st.sidebar.text_input("🖥️ Host / Server Name", value=st.session_state.db_host)
    st.session_state.db_name = st.sidebar.text_input("📂 Database Name", value=st.session_state.db_name)
    st.session_state.db_user = st.sidebar.text_input("👤 Username", value=st.session_state.db_user)
    st.session_state.db_password = st.sidebar.text_input("🔑 Password", value=st.session_state.db_password, type="password")

    if st.sidebar.button("⚡ Connect to Database"):
        try:
            if db_type == "MSSQL":
                engine = create_engine(
                    f"mssql+pyodbc://{st.session_state.db_user}:{st.session_state.db_password}"
                    f"@{st.session_state.db_host}/{st.session_state.db_name}?driver=ODBC+Driver+17+for+SQL+Server"
                )
            elif db_type == "MySQL":
                engine = create_engine(
                    f"mysql+pymysql://{st.session_state.db_user}:{st.session_state.db_password}"
                    f"@{st.session_state.db_host}:3306/{st.session_state.db_name}"
                )
            else:
                raise ValueError("Unsupported database type.")

            conn = engine.connect()
            conn.close()
            st.session_state.db_connected = True
            st.sidebar.success(f"✅ Connected to your {db_type} database successfully.")
            db = SQLDatabase(engine, sample_rows_in_table_info=3)

        except Exception as e:
            st.session_state.db_connected = False
            st.session_state.db_error = str(e)

            if "timeout" in str(e).lower() or "could not connect" in str(e).lower():
                st.sidebar.info(
                    f"🔒 To connect your local {db_type} database, please run this app on your own computer.\n\n"
                    "👉 Clone this project from GitHub and launch locally for secure access."
                )
            else:
                st.sidebar.info(
                    f"🔒 To connect your local {db_type} database, please run this app on your own computer.\n\n"
                    "👉 Clone this project from GitHub and launch locally for secure access."

    # ✅ Keep connection alive after rerun
    if st.session_state.db_connected:
        st.sidebar.success(f"🟢 {db_type} connection is active.")
        try:
            if db_type == "MSSQL":
                engine = create_engine(
                    f"mssql+pyodbc://{st.session_state.db_user}:{st.session_state.db_password}"
                    f"@{st.session_state.db_host}/{st.session_state.db_name}?driver=ODBC+Driver+17+for+SQL+Server"
                )
            elif db_type == "MySQL":
                engine = create_engine(
                    f"mysql+pymysql://{st.session_state.db_user}:{st.session_state.db_password}"
                    f"@{st.session_state.db_host}:3306/{st.session_state.db_name}"
                )
            db = SQLDatabase(engine, sample_rows_in_table_info=3)
        except Exception as e:
            st.sidebar.error(f"⚠️ Error reinitializing {db_type} database: {e}")
            db = None
    else:
        db = None


    # Clear credentials
    if st.sidebar.button("🗑️ Clear DB Credentials"):
        for key in ["db_host", "db_name", "db_user", "db_password", "db_connected", "db_error"]:
            st.session_state[key] = "" if not key.endswith("_connected") and not key.endswith("_error") else False
        st.sidebar.success("✅ Database credentials cleared.")
        st.rerun()



# ===============================================================
# Few-Shot Examples (for better SQL generation)
# ===============================================================
# 📌 Purpose:
#   Seed examples are embedded and indexed, improving LLM reliability by
#   retrieving relevant examples (via semantic similarity) per question.

few_shots = []

if db_type == "SQLite":
    few_shots = [
    {'Question': "Show all employee names and their salaries.",
     'SQLQuery': "SELECT staff_name, salary FROM employees;",
     'SQLResult': "suresh - 20000, ajith - 15000, arul - 18000",
     'Answer': "Displays all employees and their salaries."},

    {'Question': "List all department names.",
     'SQLQuery': "SELECT depart_name FROM department;",
     'SQLResult': "data_scientist, data_analyst, data_enginer",
     'Answer': "There are three departments."},

    {'Question': "Show employees who work in the data scientist department.",
     'SQLQuery': """SELECT e.staff_name, d.depart_name
FROM employees AS e
JOIN department AS d ON e.department = d.depart_id
WHERE d.depart_name = 'data_scientist';""",
     'SQLResult': "suresh - data_scientist",
     'Answer': "Suresh works in the Data Scientist department."},

    {'Question': "Find the employee with the highest salary.",
     'SQLQuery': """SELECT staff_name, salary
FROM employees
ORDER BY salary DESC
LIMIT 1;""",
     'SQLResult': "suresh - 20000",
     'Answer': "Suresh earns the highest salary."},

    {'Question': "Find the employee with the lowest salary.",
     'SQLQuery': """SELECT staff_name, salary
FROM employees
ORDER BY salary ASC
LIMIT 1;""",
     'SQLResult': "ajith - 15000",
     'Answer': "Ajith earns the lowest salary."},

    {'Question': "List employees with their department and city.",
     'SQLQuery': """SELECT e.staff_name, d.depart_name, l.City
FROM employees AS e
JOIN department AS d ON e.department = d.depart_id
JOIN LOCATION AS l ON e.Location_ID = l.Location_ID;""",
     'SQLResult': "suresh - data_scientist - Salem, ajith - data_analyst - Dallas, arul - data_enginer - Chicago",
     'Answer': "Displays employees with department and city."},

    {'Question': "Show employees working outside India.",
     'SQLQuery': """SELECT staff_name, coundry
FROM employees
WHERE coundry != 'INDIA';""",
     'SQLResult': "ajith - USA, arul - CANADA",
     'Answer': "Ajith and Arul work outside India."},

    {'Question': "Count employees per department.",
     'SQLQuery': """SELECT d.depart_name, COUNT(e.employee_id)
FROM employees AS e
JOIN department AS d ON e.department = d.depart_id
GROUP BY d.depart_name;""",
     'SQLResult': "data_scientist - 1, data_analyst - 1, data_enginer - 1",
     'Answer': "Each department has one employee."},

    {'Question': "Calculate average salary of all employees.",
     'SQLQuery': "SELECT AVG(salary) FROM employees;",
     'SQLResult': "17666.67",
     'Answer': "Average salary is ₹17,666.67."},

    {'Question': "Show employees earning more than ₹15,000.",
     'SQLQuery': """SELECT e.staff_name, e.salary, d.depart_name
FROM employees AS e
JOIN department AS d ON e.department = d.depart_id
WHERE e.salary > 15000;""",
     'SQLResult': "suresh - 20000 - data_scientist, arul - 18000 - data_enginer",
     'Answer': "Suresh and Arul earn above ₹15,000."},

    {'Question': "Find departments with higher-than-average company salary.",
     'SQLQuery': """SELECT d.depart_name
FROM department AS d
JOIN employees AS e ON e.department = d.depart_id
GROUP BY d.depart_name
HAVING AVG(e.salary) > (SELECT AVG(salary) FROM employees);""",
     'SQLResult': "data_enginer",
     'Answer': "Data Engineer department has higher salary."},

    {'Question': "Which city contributes the most to total payroll?",
     'SQLQuery': """SELECT l.City, SUM(e.salary) AS TotalSalary
FROM employees AS e
JOIN LOCATION AS l ON e.Location_ID = l.Location_ID
GROUP BY l.City
ORDER BY TotalSalary DESC
LIMIT 1;""",
     'SQLResult': "Chicago - 25000",
     'Answer': "Chicago contributes the most to payroll."},

    {'Question': "List top 3 employees earning 20% above their department average.",
     'SQLQuery': """WITH DeptAvg AS (
  SELECT department, AVG(salary) AS avg_salary
  FROM employees
  GROUP BY department)
SELECT e.staff_name, e.salary, d.depart_name
FROM employees AS e
JOIN department AS d ON e.department = d.depart_id
JOIN DeptAvg AS a ON e.department = a.department
WHERE e.salary > a.avg_salary * 1.2
ORDER BY e.salary DESC
LIMIT 3;""",
     'SQLResult': "arul - 25000 - data_enginer",
     'Answer': "Arul earns more than 20% above department average."}
]


elif db_type == "MySQL":
    few_shots = [
    {'Question': "Show all employee names and their corresponding salaries.",
     'SQLQuery': "SELECT staff_name, salary FROM employees;",
     'SQLResult': "suresh - 20000.00, ajith - 15000.00, arul - 18000.00",
     'Answer': "There are three employees — Suresh earns ₹20,000, Ajith earns ₹15,000, and Arul earns ₹18,000."},

    {'Question': "List all department names.",
     'SQLQuery': "SELECT depart_name FROM department;",
     'SQLResult': "data_scientist, data_analyst, data_enginer",
     'Answer': "The company has three departments — Data Scientist, Data Analyst, and Data Engineer."},

    {'Question': "Show all employees who work in the data scientist department.",
     'SQLQuery': """SELECT e.staff_name, d.depart_name
FROM employees AS e
JOIN department AS d ON e.department = d.depart_id
WHERE d.depart_name = 'data_scientist';""",
     'SQLResult': "suresh - data_scientist",
     'Answer': "Suresh is the only employee working in the Data Scientist department."},

    {'Question': "Which employee has the highest salary?",
     'SQLQuery': """SELECT staff_name, salary
FROM employees
ORDER BY salary DESC
LIMIT 1;""",
     'SQLResult': "suresh - 20000.00",
     'Answer': "Suresh earns the highest salary — ₹20,000."},

    {'Question': "Which employee earns the lowest salary?",
     'SQLQuery': """SELECT staff_name, salary
FROM employees
ORDER BY salary ASC
LIMIT 1;""",
     'SQLResult': "ajith - 15000.00",
     'Answer': "Ajith earns the lowest salary — ₹15,000."},

    {'Question': "List all employees along with their department and city.",
     'SQLQuery': """SELECT e.staff_name, d.depart_name, l.City
FROM employees AS e
JOIN department AS d ON e.department = d.depart_id
JOIN LOCATION AS l ON e.Location_ID = l.Location_ID;""",
     'SQLResult': "suresh - data_scientist - Salem, ajith - data_analyst - Dallas, arul - data_enginer - Chicago",
     'Answer': "Suresh works in Salem as a Data Scientist, Ajith in Dallas as a Data Analyst, and Arul in Chicago as a Data Engineer."},

    {'Question': "Show all employees working outside India.",
     'SQLQuery': """SELECT staff_name, coundry
FROM employees
WHERE coundry <> 'INDIA';""",
     'SQLResult': "ajith - USA, arul - CANADA",
     'Answer': "Ajith works in the USA and Arul in Canada — both outside India."},

    {'Question': "Count how many employees are in each department.",
     'SQLQuery': """SELECT d.depart_name, COUNT(e.employee_id) AS employee_count
FROM employees AS e
JOIN department AS d ON e.department = d.depart_id
GROUP BY d.depart_name;""",
     'SQLResult': "data_scientist - 1, data_analyst - 1, data_enginer - 1",
     'Answer': "Each department currently has one employee."},

    {'Question': "Find the average salary of all employees.",
     'SQLQuery': "SELECT AVG(salary) AS average_salary FROM employees;",
     'SQLResult': "17666.67",
     'Answer': "The average employee salary is ₹17,666.67."},

    {'Question': "Show all employees and their department where salary is above ₹15,000.",
     'SQLQuery': """SELECT e.staff_name, e.salary, d.depart_name
FROM employees AS e
JOIN department AS d ON e.department = d.depart_id
WHERE e.salary > 15000;""",
     'SQLResult': "suresh - 20000.00 - data_scientist, arul - 18000.00 - data_enginer",
     'Answer': "Suresh and Arul earn above ₹15,000 — in Data Scientist and Data Engineer departments, respectively."},

    {'Question': "Find the department(s) whose average salary is higher than the company average.",
     'SQLQuery': """SELECT d.depart_name
FROM department AS d
JOIN employees AS e ON d.depart_id = e.department
GROUP BY d.depart_name
HAVING AVG(e.salary) > (SELECT AVG(salary) FROM employees);""",
     'SQLResult': "data_enginer",
     'Answer': "The Data Engineer department has a higher-than-average company salary."},

    {'Question': "Which city contributes the most to the total payroll?",
     'SQLQuery': """SELECT l.City, SUM(e.salary) AS TotalSalary
FROM employees AS e
JOIN LOCATION AS l ON e.Location_ID = l.Location_ID
GROUP BY l.City
ORDER BY TotalSalary DESC
LIMIT 1;""",
     'SQLResult': "Chicago - 25000",
     'Answer': "Chicago contributes the most to the total payroll."},

    {'Question': "List top 3 employees who earn more than 20% above their department’s average salary.",
     'SQLQuery': """WITH DeptAvg AS (
  SELECT department, AVG(salary) AS avg_salary
  FROM employees
  GROUP BY department)
SELECT e.staff_name, e.salary, d.depart_name
FROM employees AS e
JOIN department AS d ON e.department = d.depart_id
JOIN DeptAvg AS a ON e.department = a.department
WHERE e.salary > a.avg_salary * 1.2
ORDER BY e.salary DESC
LIMIT 3;""",
     'SQLResult': "arul - 25000 - data_enginer",
     'Answer': "Arul earns more than 20% above his department’s average salary."}
]


else:  # MSSQL
    few_shots= [
    {'Question': "Show all employee names and their corresponding salaries.",
     'SQLQuery': "SELECT staff_name, salary FROM employees;",
     'SQLResult': "suresh - 20000.00, ajith - 15000.00, arul - 18000.00",
     'Answer': "Three employees — Suresh earns ₹20,000, Ajith earns ₹15,000, and Arul earns ₹18,000."},

    {'Question': "List all department names.",
     'SQLQuery': "SELECT depart_name FROM department;", 
     'SQLResult': "data_scientist, data_analyst, data_enginer",
     'Answer': "The company has three departments."},

    {'Question': "Show all employees who work in the data scientist department.",
     'SQLQuery': """SELECT e.staff_name, d.depart_name
FROM employees AS e
JOIN department AS d ON e.department = d.depart_id
WHERE d.depart_name = 'data_scientist';""",
     'SQLResult': "suresh - data_scientist",
     'Answer': "Suresh works in the Data Scientist department."},

    {'Question': "Which employee has the highest salary?",
     'SQLQuery': """SELECT TOP 1 staff_name, salary
FROM employees
ORDER BY salary DESC;""",
     'SQLResult': "suresh - 20000.00",
     'Answer': "Suresh earns the highest salary."},

    {'Question': "Which employee earns the lowest salary?",
     'SQLQuery': """SELECT TOP 1 staff_name, salary
FROM employees
ORDER BY salary ASC;""",
     'SQLResult': "ajith - 15000.00",
     'Answer': "Ajith earns the lowest salary."},

    {'Question': "List all employees with their department and city.",
     'SQLQuery': """SELECT e.staff_name, d.depart_name, l.City
FROM employees AS e
JOIN department AS d ON e.department = d.depart_id
JOIN LOCATION AS l ON e.Location_ID = l.Location_ID;""",
     'SQLResult': "suresh - data_scientist - Salem, ajith - data_analyst - Dallas, arul - data_enginer - Chicago",
     'Answer': "Shows all employees with their departments and cities."},

    {'Question': "Show all employees working outside India.",
     'SQLQuery': """SELECT staff_name, coundry
FROM employees
WHERE coundry <> 'INDIA';""",
     'SQLResult': "ajith - USA, arul - CANADA",
     'Answer': "Ajith and Arul work outside India."},

    {'Question': "Count how many employees are in each department.",
     'SQLQuery': """SELECT d.depart_name, COUNT(e.employee_id) AS employee_count
FROM employees AS e
JOIN department AS d ON e.department = d.depart_id
GROUP BY d.depart_name;""",
     'SQLResult': "data_scientist - 1, data_analyst - 1, data_enginer - 1",
     'Answer': "Each department has one employee."},

    {'Question': "Find the average salary of all employees.",
     'SQLQuery': "SELECT AVG(salary) AS average_salary FROM employees;",
     'SQLResult': "17666.67",
     'Answer': "The average salary is ₹17,666.67."},

    {'Question': "Show employees whose salary is above ₹15,000.",
     'SQLQuery': """SELECT e.staff_name, e.salary, d.depart_name
FROM employees AS e
JOIN department AS d ON e.department = d.depart_id
WHERE e.salary > 15000;""",
     'SQLResult': "suresh - 20000 - data_scientist, arul - 18000 - data_enginer",
     'Answer': "Suresh and Arul earn above ₹15,000."},

    {'Question': "Find departments with average salary above company average.",
     'SQLQuery': """SELECT d.depart_name
FROM department AS d
JOIN employees AS e ON e.department = d.depart_id
GROUP BY d.depart_name
HAVING AVG(e.salary) > (SELECT AVG(salary) FROM employees);""",
     'SQLResult': "data_enginer",
     'Answer': "The Data Engineer department has a higher-than-average salary."},

    {'Question': "Which city contributes the most to total payroll?",
     'SQLQuery': """SELECT TOP 1 l.City, SUM(e.salary) AS TotalSalary
FROM employees AS e
JOIN LOCATION AS l ON e.Location_ID = l.Location_ID
GROUP BY l.City
ORDER BY TotalSalary DESC;""",
     'SQLResult': "Chicago - 25000",
     'Answer': "Chicago contributes most to payroll."},

    {'Question': "List top 3 employees earning 20% above their department’s average salary.",
     'SQLQuery': """SELECT TOP 3 e.staff_name, e.salary, d.depart_name
FROM employees AS e
JOIN department AS d ON e.department = d.depart_id
WHERE e.salary > (SELECT AVG(salary)*1.2 FROM employees AS x WHERE x.department = e.department)
ORDER BY e.salary DESC;""",
     'SQLResult': "arul - 25000 - data_enginer",
     'Answer': "Arul earns more than 20% above the average in his department."}
]



# ===============================================================
# Embeddings / Vector Store — Safe Loader
# ===============================================================
# 📌 Purpose:
#   Load SentenceTransformer embeddings robustly, avoiding the intermittent
#   "meta tensor" error. Prefer GPU when available, otherwise CPU.

# Reduce tokenizer parallel warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Best-effort cache cleanup on reload
try:
    torch.cuda.empty_cache()
except:
    pass
gc.collect()

def safe_load_embeddings():
    """
    Safely initialize HF embeddings with automatic GPU→CPU fallback.

    Strategy:
        1) Default device to CPU to avoid meta tensor creation.
        2) If CUDA available, try GPU first.
        3) On NotImplementedError (meta tensor) or any failure, clean up and load on CPU.
    """
    def _cleanup():
        try:
            torch.cuda.empty_cache()
        except:
            pass
        gc.collect()

    # Avoid meta tensor by defaulting to CPU (harmless if not supported)
    try:
        torch.set_default_device("cpu")
    except Exception:
        pass

    if torch.cuda.is_available():
        try:
            print("🚀 Trying GPU for embeddings...")
            return HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={"device": "cuda"},
            )
        except NotImplementedError:
            print("⚠️ Meta tensor detected! Retrying with CPU...")
            _cleanup()
        except Exception as e:
            print(f"⚠️ GPU init failed ({e}). Retrying on CPU.")
            _cleanup()

    # Fallback: CPU always works
    print("🧠 Loading embeddings on CPU...")
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )

# Load embeddings once per session (memoized in st.session_state)
if "embeddings" not in st.session_state or st.session_state.embeddings is None:
    embeddings = safe_load_embeddings()
    st.session_state.embeddings = embeddings
else:
    embeddings = st.session_state.embeddings

# Display device info (informational only)
if torch.cuda.is_available():
    st.sidebar.info("💻 Using GPU for embeddings")
else:
    st.sidebar.info("🧠 Using CPU for embeddings")

# Build a tiny vector store from few-shot texts for similarity retrieval
to_vectorize = [" ".join(example.values()) for example in few_shots]
vectorstore = Chroma.from_texts(to_vectorize, embeddings, metadatas=few_shots)

example_selector = SemanticSimilarityExampleSelector(
    vectorstore=vectorstore,
    k=2,
)


# ===============================================================
# Prompt Engineering — Adaptive per DB Type (SQLite / MySQL / MSSQL)
# ===============================================================
# 📌 Purpose:
#   Dynamically adjust the LLM's SQL reasoning rules, keywords, and examples
#   based on the active database engine type (db_type).
#   This ensures accurate and safe query generation.

# ✅ Auto-adjust keywords and syntax rules based on DB type
if db_type == "MSSQL":
    syntax_notes = """
    - Use correct **T-SQL** syntax.
    - Use `TOP {top_k}` when limiting results (e.g., “top 5 employees”).
    - Do NOT use `LIMIT`; MSSQL doesn’t support it.
    - Use square brackets [ ] for identifiers (e.g., [salary]).
    - Use `GETDATE()` for current date/time.
    - Use `LEN()` for string length.
    - When joining, always qualify columns with table aliases.
    """
elif db_type == "MySQL":
    syntax_notes = """
    - Use correct **MySQL** syntax.
    - Use `LIMIT {top_k}` when limiting rows (never use TOP).
    - Do NOT use square brackets — use backticks (`column`) only if necessary.
    - Use `NOW()` for current date/time.
    - Use `CHAR_LENGTH()` for string length.
    - Joins should use `INNER JOIN`, `LEFT JOIN`, etc. syntax.
    """
else:
    syntax_notes = """
    - Use correct **SQLite** syntax.
    - Use `LIMIT {top_k}` when limiting rows.
    - Avoid backticks; use plain identifiers.
    - Use `DATE('now')` for current date.
    - Be minimal and efficient — SQLite prefers simple queries.
    """

# ===============================================================
# Core Prompt Definition
# ===============================================================
llm_prompt = f"""
You are an expert SQL developer specialized in **{db_type}** database systems.

Your task is to translate a natural language question into a syntactically correct and optimal {db_type} SQL query,
mentally execute it, and provide the correct answer.

Always follow these detailed rules carefully:

1️⃣ **General Guidelines**
- Use only table names and column names that exist in the provided database schema (`table_info`).
- Never invent tables or columns that do not exist.
- Prefer explicit JOIN syntax instead of comma joins.
- Use proper casing for SQL keywords (SELECT, FROM, WHERE, JOIN, etc.).
- Always select only the columns necessary to answer the question.
- Avoid unnecessary subqueries or nested selects unless required.
- If aggregation is needed (AVG, SUM, COUNT), use proper GROUP BY clauses.

2️⃣ **{db_type} Syntax Rules**
{syntax_notes}

3️⃣ **Output Format**
The LLM must strictly follow this output format:
Question: <restated user question>
SQLQuery: <final, executable {db_type} SQL query>
SQLResult: <hypothetical result or logical output>
Answer: <concise natural language explanation>


⚠️ Absolutely no preamble, commentary, or Markdown fences (no ```sql```).
Only return the format above exactly as shown.
"""

# ===============================================================
# Few-Shot Prompt Template with Example Selector
# ===============================================================
example_prompt = PromptTemplate(
    input_variables=["Question", "SQLQuery", "SQLResult", "Answer"],
    template=(
        "\nQuestion: {Question}\n"
        "SQLQuery: {SQLQuery}\n"
        "SQLResult: {SQLResult}\n"
        "Answer: {Answer}"
    ),
)

few_shot_prompt = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=example_prompt,
    prefix=llm_prompt,
    suffix=PROMPT_SUFFIX,  # includes "Only use these tables" context
    input_variables=["input", "table_info", "top_k"],
)


# ===============================================================
# Guard: Require Gemini before Building the Chain
# ===============================================================
# 📌 Purpose:
#   Stop rendering before we build the chain if Gemini isn't connected,
#   so we don't attempt to invoke an uninitialized LLM.

if st.session_state.gemini_llm is None:
    st.info("💡 Connect Gemini using the sidebar to start asking questions.")
    st.stop()

# With a live LLM and db, we can now build the chain.
llm = st.session_state.gemini_llm
if db is None:
    st.info("Please connect your database (Demo / MySQL / MSSQL)")
    st.stop()
rag_chain = create_sql_query_chain(llm, db, prompt=few_shot_prompt)


# ===============================================================
# SQL Execution Helper
# ===============================================================
# 📌 Purpose:
#   - Clean the LLM’s output to get the raw SQL string.
#   - In demo mode, block destructive write operations (safety).
#   - Execute via LangChain's `db.run` and return (sql, result).

def execute_query(question: str):
    """
    Generate SQL from a natural-language question and execute it.

    Args:
        question (str): user question in plain English

    Returns:
        (cleaned_query, result):
            cleaned_query: str — final SQL sent to DB
            result: query output (type depends on connection/driver)

    Safety:
        - In demo mode, destructive statements are blocked to protect the
          local SQLite file from mutation (e.g., DROP, DELETE, etc.).
    """
    try:
        # 1) Generate SQL using the chain
        response = rag_chain.invoke({"question": question})

        # 2) Clean fences/labels
        cleaned_query = (
            str(response)
            .replace("```sql", "")
            .replace("```", "")
            .replace("SQLQuery:", "")
            .strip()
        )

        # 3) Demo safety block: prevent modifications
        destructive_keywords = [
            "drop ", "delete ", "truncate ", "alter ", "update ", "insert ", "replace ",
            "create ", "rename ", "grant ", "revoke ", "commit ", "rollback ", "savepoint "
        ]
        lower_query = cleaned_query.lower()

        if connection_mode == "Use Demo Database" and any(
            kw in lower_query for kw in destructive_keywords
        ):
            st.warning("⚠️ This query modifies the demo database — execution blocked for safety.")
            st.info("💡 In demo mode, only **read-only** queries like SELECT are allowed.")
            return cleaned_query, "Query blocked (demo mode safety)."

        # 4) Execute safely
        result = db.run(cleaned_query)
        return cleaned_query, result

    except ProgrammingError as e:
        st.error(f"SQL Error: {e}")
        return None, None
    except Exception as e:
        st.error(f"Unexpected error: {e}")
        return None, None


# ===============================================================
# Minor Layout Helpers (UI polish; logic-free)
# ===============================================================
# 📌 Purpose:
#   Keep the rest of the UI content from shifting if a floating panel appears.
#   Pure CSS/JS injection; harmless if panel never appears.

st.markdown("""
    <style>
    .main-content { transition: margin-right 0.5s ease-in-out; margin-right: 0px; }
    .llm-open .main-content { margin-right: 520px !important; }
    .block-container { transition: margin-right 0.5s ease-in-out; }
    #llm-panel { transform: translateX(100%); opacity: 0; transition: all 0.6s ease-in-out; }
    body.llm-open #llm-panel { transform: translateX(0); opacity: 1; }
    #llm-panel { animation: slideIn 0.6s ease forwards; }
    @keyframes slideIn { from { transform: translateX(50px); opacity: 0; } to { transform: translateX(0); opacity: 1; } }
    </style>
    <script>
    const observer = new MutationObserver(() => {
        const panel = document.getElementById('llm-panel');
        if (panel) { document.body.classList.add('llm-open'); }
        else { document.body.classList.remove('llm-open'); }
    });
    observer.observe(document.body, { childList: true, subtree: true });
    </script>
""", unsafe_allow_html=True)


# ===============================================================
# Main UI — History, Input, Execute
# ===============================================================
# 📌 Purpose:
#   - Provide the question input and execution flow.
#   - Maintain a simple query history (in-memory, no persistence).

st.markdown('<div class="main-content">', unsafe_allow_html=True)

# History store
if "history" not in st.session_state:
    st.session_state.history = []   # [{question, sql, result}, ...]

# Sidebar history UI
st.sidebar.title("🕒 Query History")
if st.session_state.history:
    for i, entry in enumerate(reversed(st.session_state.history), 1):
        with st.sidebar.expander(f"{i}. {entry['question'][:40]}..."):
            st.markdown(f"**🧠 SQL Query:**\n```sql\n{entry['sql']}\n```", unsafe_allow_html=True)
            st.markdown(f"**📊 Result:** {entry['result']}")
else:
    st.sidebar.info("No history yet.")

if st.sidebar.button("🗑️ Clear History"):
    st.session_state.history = []
    st.sidebar.success("🧹 Query history cleared.")
    st.rerun()

# Input area
question = st.text_area(
    "Enter your question:",
    placeholder="Type your question here (press Enter for a new line, click Execute to run)...",
    height=100
)


# ===============================================================
# Execute Handler
# ===============================================================
# 📌 Purpose:
#   - Ensures LLM is connected.
#   - Generates SQL, prints it nicely, runs it, shows results.
#   - Renders a styled LLM explanation with equivalent queries.

if st.button("Execute"):
    if question:

        # Ensure Gemini is connected (lazy prompt)
        if "gemini_llm" not in st.session_state or st.session_state.gemini_llm is None:
            st.warning("🔑 Gemini API key not found or invalid. Please enter below to continue.")
            new_key = st.text_input("Enter your Google Gemini API Key:", type="password", key="manual_gemini_key")
            if new_key:
                llm_try, _ = _try_init_gemini(new_key.strip())
                if llm_try:
                    st.session_state.gemini_llm = llm_try
                    st.session_state.gemini_key = new_key.strip()
                    st.success("✅ Gemini connected successfully! Click Execute again.")
                else:
                    st.error("❌ Invalid or expired Gemini API key. Please re-enter a valid key.")
            st.stop()

        # Use connected LLM
        llm = st.session_state.gemini_llm

        # Generate + run
        cleaned_query, query_result = execute_query(question)
        if cleaned_query and query_result is not None:
            import sqlparse

            # ===============================================================
            # Generated SQL Query — with Dynamic Info Message
            # ===============================================================
            st.subheader("🧠 Generated SQL Query:")
            
            # Dynamic note based on database type
            if db_type == "SQLite":
                st.caption("💡 Currently using **SQLite demo database** output. "
                           "If you connect your own **MySQL** or **MSSQL** database, "
                           "the SQL syntax will automatically adapt to that system.")
            elif db_type == "MySQL":
                st.caption("💡 Connected to a **MySQL database**. "
                           "All queries use **MySQL syntax** (e.g., `LIMIT`, backticks, `NOW()`).")
            elif db_type == "MSSQL":
                st.caption("💡 Connected to a **Microsoft SQL Server (MSSQL)** database. "
                           "All queries use **T-SQL syntax** (e.g., `TOP n`, `GETDATE()`, `[brackets]`).")
            else:
                st.caption("💡 Using a custom database connection — SQL syntax will adapt automatically.")
            
            # Format and display query
            formatted_sql = sqlparse.format(cleaned_query, reindent=True, keyword_case='upper')
            st.code(formatted_sql, language="sql")

            st.subheader("📊 Query Result:")
            st.write(query_result)

            # ===============================================================
            # 🧠 LLM Interpretation — Dual DB (MySQL + MSSQL)
            # ===============================================================
            try:
                llm_input = f"""
                You are an expert SQL and data analytics specialist.
                The AI has already generated a valid SQL query for SQLite.
        
                Your tasks:
                1️⃣ Briefly (in 2–3 sentences) explain what the query does and what result it returns.
                2️⃣ Provide equivalent SQL queries for the same logic in:
                    - MySQL
                    - MSSQL
                3️⃣ Finally, summarize the key syntax differences between MySQL and MSSQL.
        
                Output format (strictly follow this):
                💬 Explanation:
                <brief summary>
        
                🔄 Equivalent in MySQL:
                <MySQL SQL>
        
                🔄 Equivalent in MSSQL:
                <MSSQL SQL>
        
                💡 Syntax differences:
                <short notes>
        
                Question: {question}
                SQLQuery (SQLite): {cleaned_query}
                SQLResult: {query_result}
                """
                llm_answer = llm.invoke(llm_input)

                # Clean textual artifacts and prettify embedded SQL blocks
                llm_text = str(llm_answer).replace("```", "").replace("\\n", "\n").strip()

                def format_sql_blocks(text: str) -> str:
                    sql_blocks = re.findall(r"(SELECT .*?;)", text, re.DOTALL | re.IGNORECASE)
                    for sql_block in sql_blocks:
                        formatted = sqlparse.format(sql_block, reindent=True, keyword_case="upper")
                        text = text.replace(sql_block, f"\n```sql\n{formatted}\n```\n")
                    return text

                llm_text = format_sql_blocks(llm_text)
                llm_text = re.sub(r"\bsql\b", "", llm_text, flags=re.IGNORECASE)  # cosmetic only

                # Styled explanation card
                st.markdown("""
                <style>
                .llm-card {
                    background: linear-gradient(145deg, rgba(40, 42, 70, 0.93), rgba(22, 23, 40, 0.95));
                    border: 1px solid rgba(130, 150, 255, 0.25);
                    border-radius: 14px;
                    padding: 25px 30px;
                    margin-top: 25px;
                    margin-bottom: 35px;
                    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.45);
                    transition: all 0.3s ease-in-out;
                }
                .llm-card:hover {
                    box-shadow: 0 6px 22px rgba(100, 120, 255, 0.3);
                    border-color: rgba(170, 180, 255, 0.4);
                }
                .llm-title {
                    font-size: 1.3rem;
                    font-weight: 700;
                    color: #cfd3ff;
                    margin-bottom: 15px;
                }
                .llm-content {
                    color: #e5e7f2;
                    font-size: 0.96rem;
                    line-height: 1.65;
                    white-space: pre-wrap;
                }
                .llm-content code {
                    background: rgba(255, 255, 255, 0.1);
                    color: #b5d9ff;
                    padding: 5px 8px;
                    border-radius: 8px;
                    font-size: 0.9rem;
                }
                </style>
                """, unsafe_allow_html=True)

                st.markdown(f"""
                    <div class="llm-card">
                        <div class="llm-title">🧠 View LLM Interpretation & Equivalent Queries</div>
                        <div class="llm-content">{llm_text}</div>
                    </div>
                """, unsafe_allow_html=True)

            except Exception as e:
                if "API_KEY_INVALID" in str(e) or "unauthorized" in str(e).lower():
                    st.warning("🔑 Gemini API key expired or invalid. Please re-enter your key in the sidebar (User key).")
                else:
                    st.warning(f"⚠️ LLM interpretation step failed: {e}")


            # Save to history (preview only)
            st.session_state.history.append({
                "question": question,
                "sql": cleaned_query,
                "result": str(query_result[:3]) + ("..." if len(query_result) > 3 else "")
            })

        else:
            st.error("⚠️ No result returned due to an error.")
    else:
        st.warning("Please enter a question.")

# Close main-content wrapper (kept twice as in your original code)
# Reason: In your working file this appears twice; removing it is unnecessary
# and could affect layout in edge cases. We keep it for complete parity.
st.markdown('</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)
