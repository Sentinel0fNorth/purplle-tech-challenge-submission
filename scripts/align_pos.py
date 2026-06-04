import csv
import os

input_path = "/workspace/Original/POS - sample transactionsb1e826f.csv"
output_path = "/app/data/pos_transactions_docker.csv"

# Fallbacks for running on host
if not os.path.exists(input_path):
    input_path = "d:/Purplle-backend/Original/POS - sample transactionsb1e826f.csv"
if not os.path.exists(os.path.dirname(output_path)):
    output_path = "data/pos_transactions_docker.csv"

print(f"Reading from {input_path}...")
with open(input_path, mode='r', newline='', encoding='utf-8') as infile:
    reader = csv.reader(infile)
    header = next(reader)
    
    # Identify indices
    # order_id,order_date,order_time,store_id,product_id,brand_name,total_amount
    date_idx = header.index('order_date')
    store_idx = header.index('store_id')
    
    rows = []
    for row in reader:
        if not row:
            continue
        # Translate store_id
        if row[store_idx] == 'ST1008':
            row[store_idx] = 'STORE_1'
        # Shift date: '10-04-2026' -> '02-06-2026'
        row[date_idx] = '02-06-2026'
        rows.append(row)

print(f"Writing {len(rows)} aligned records to {output_path}...")
with open(output_path, mode='w', newline='', encoding='utf-8') as outfile:
    writer = csv.writer(outfile)
    writer.writerow(header)
    writer.writerows(rows)

print("Alignment completed successfully!")
