import os
import pandas as pd
import pytz
from .db import get_db

def load_pos_data():
    csv_path = os.getenv("POS_CSV_PATH", "/workspace/data/pos_transactions.csv")
    if not os.path.exists(csv_path):
        print(f"POS CSV not found at {csv_path}")
        return
    
    try:
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip().str.lower()
        
        if 'order_id' not in df.columns and 'transaction_id' in df.columns:
            df['order_id'] = df['transaction_id']
            
        if 'order_date' in df.columns and 'order_time' in df.columns:
            df['dt_str'] = df['order_date'].astype(str) + ' ' + df['order_time'].astype(str)
            ist = pytz.timezone('Asia/Kolkata')
            df['dt'] = pd.to_datetime(df['dt_str'], dayfirst=True)
            df['dt_utc'] = df['dt'].dt.tz_localize(ist, ambiguous='NaT', nonexistent='NaT').dt.tz_convert(pytz.UTC)
            df['timestamp'] = df['dt_utc'].dt.strftime('%Y-%m-%dT%H:%M:%SZ')
        elif 'timestamp' in df.columns:
            pass # Keep as is, assuming UTC ISO string

        grouped = df.groupby(['order_id', 'store_id', 'timestamp']).agg({
            'total_amount': 'sum',
            'brand_name': lambda x: ','.join(set(x.dropna().astype(str)))
        }).reset_index()
        
        with get_db() as conn:
            for _, row in grouped.iterrows():
                conn.execute(
                    """
                    INSERT OR IGNORE INTO pos_transactions (order_id, store_id, timestamp, total_amount, brands)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (row['order_id'], row['store_id'], row['timestamp'], row['total_amount'], row['brand_name'])
                )
            conn.commit()
    except Exception as e:
        print(f"Error loading POS CSV: {e}")
