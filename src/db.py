import sqlite3
import pandas as pd
from typing import List
from pathlib import Path
from src.utils import config
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_engine():
    from sqlalchemy import create_engine
    db_uri = config['data']['db_path']
    if db_uri.startswith('sqlite:///'):
        db_path = db_uri.replace('sqlite:///', '')
        db_abspath = Path(__file__).parent.parent / db_path
        db_uri = f"sqlite:///{db_abspath.as_posix()}"
    return create_engine(db_uri)

def init_db(schema_path: str = "sql/schema.sql"):
    """Initialize database tables from schema.sql using direct sqlite3."""
    db_path = config['data']['db_path'].replace('sqlite:///', '')
    db_file = Path(__file__).parent.parent / db_path
    
    schema_file = Path(__file__).parent.parent / schema_path
    with open(schema_file, 'r') as f:
        schema_sql = f.read()

    logger.info(f"Initializing database at {db_file}")
    conn = sqlite3.connect(db_file)
    try:
        conn.executescript(schema_sql)
        conn.commit()
    finally:
        conn.close()

def execute_query(query: str, as_df: bool = False):
    """Executes a SQL query and returns results if requested as pandas DataFrame."""
    engine = get_engine()
    if as_df:
        return pd.read_sql(query, engine)
    
    with engine.begin() as conn:
        from sqlalchemy import text
        conn.execute(text(query))

def write_df_to_db(df: pd.DataFrame, table_name: str, if_exists: str = "append", index: bool = False):
    """Writes a DataFrame to the SQL database."""
    engine = get_engine()
    df.to_sql(table_name, con=engine, if_exists=if_exists, index=index)
    logger.info(f"Wrote {len(df)} rows to {table_name}.")
