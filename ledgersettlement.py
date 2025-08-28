from fastapi import FastAPI, HTTPException
from azure.storage.blob import BlobServiceClient
from datetime import datetime
from dotenv import load_dotenv
import requests
import os
import io


import pandas as pd
import numpy as np
import re
import uuid
from thefuzz import fuzz 
import datetime
import itertools

# === FastAPI Setup ===
app = FastAPI()
load_dotenv()

CONNSTRING = os.getenv('CONNSTRING')
OUTPUTCON = os.getenv('OUTPUTCON')
INPUTCON = os.getenv('INPUTCON')

class LedgerSettlement:
    def __init__(self, days, atol, highest_score, accounts): 
        self.transdays = days
        self.atol = atol
        self.highest_score = highest_score
        self.account = accounts
        self.results = []
        self.settlerecs = 0

    def load_ledger(self, file):
        ledger = pd.read_csv(file)
        ledger = ledger[(ledger['MainAccount'].astype(str).isin(self.account))]

        # Clean and prepare data
        ledger['Journal number'] = ledger['Journal number'].fillna("")
        ledger['Voucher'] = ledger['Voucher'].fillna("")
        ledger['MainAccount'] = ledger['MainAccount'].fillna("")
        ledger['Amount'] = pd.to_numeric(ledger['Amount'].astype(str).str.replace(',', ''), errors='coerce').fillna(0)
        ledger['Date'] = pd.to_datetime(ledger['Date'], errors='coerce')  # Ensure Date is in datetime format
        ledger['Description'] = ledger['Description'].fillna("")
        ledger['CostCentre'] = ledger['CostCentre'].fillna("")
        ledger['ProfitCentre'] = ledger['ProfitCentre'].fillna("")
        ledger['Type'] = ledger['Amount'].apply(lambda x: 'debit' if x < 0 else 'credit')
        ledger['ExtractedNumbers'] = ledger['Description'].apply(lambda x: [num for num in re.findall(r'\d+', str(x)) if len(num) > 4])
        ledger['Matched'] = False

        self.ledger = ledger

    def settle(self):
        # Step 1: Sort debit rows by date
        debit_df = self.ledger[(self.ledger['Type'] == 'debit') & (~self.ledger['Matched'])].sort_values(by='Date')
        for _, debit_row in debit_df.iterrows():
            debit_idx = debit_row.name
            debit_numbers = set(debit_row['ExtractedNumbers'])
            

            # Step 2: Filter matching credits by date and match status
            credit_df = self.ledger[
                (self.ledger['Type'] == 'credit') &
                (~self.ledger['Matched']) &
                (self.ledger['CostCentre'] == debit_row['CostCentre']) &
                (self.ledger['ProfitCentre'] == debit_row['ProfitCentre']) &             
                (self.ledger['Date'] >= debit_row['Date'] - pd.Timedelta(days=self.transdays)) &
                (self.ledger['Date'] <= debit_row['Date'] + pd.Timedelta(days=self.transdays))
            ].sort_values(by='Date')

            if debit_numbers:
                # Step 3: Filter credits by matching extracted numbers
                filtered_credits = credit_df[
                    credit_df['ExtractedNumbers'].apply(lambda x: any(num in debit_numbers for num in x))
                ]

                # Step 4: Try 1:1 match
                for credit_idx, credit_row in filtered_credits.iterrows():
                    if np.isclose(abs(credit_row['Amount']), abs(debit_row['Amount']), atol=self.atol):
                        uid = str(uuid.uuid4())
                        self.mark_matched([debit_idx, credit_idx], uid)
                        break
                else:
                    # Step 5: Try combination match
                    self.match_combinations(debit_row, filtered_credits)
                    # Step 4: If still unmatched, try fuzzy match on all unmatched credits in date range
            if not self.ledger.at[debit_idx, 'Matched']:
                self.match_credits_fuzz(debit_row, credit_df)


    def match_combinations(self, debit_row, credit_df):
        debit_idx = debit_row.name
        debit_amount = abs(debit_row['Amount'])

        if credit_df.empty:
            return  # No matching credit rows to check

        amounts = credit_df['Amount'].tolist()
        indices = credit_df.index.tolist()

        for r in range(2, min(7, len(amounts) + 1)):
            for combo in itertools.combinations(enumerate(amounts), r):
                combo_indices, combo_amounts = zip(*combo)
                if np.isclose(sum(combo_amounts), -debit_row['Amount'], atol=self.atol):
                    ledger_indices = [indices[i] for i in combo_indices]
                    uid = str(uuid.uuid4())
                    self.mark_matched([debit_idx] + ledger_indices, uid)
                    return


    def match_credits_fuzz(self, debit_row, credits):
        """
        For a given debit, finds the best matching credit or credit combinations
        based on fuzzy matching of descriptions and amount closeness.
        Ensures each debit is matched and added only once.
        """
        unique_guid = str(uuid.uuid4())
        debit_idx = debit_row.name
        matched_credit_indices = []
        matched_credit_amounts = []

        for credit_idx, credit_row in credits.iterrows():
            score = fuzz.token_set_ratio(debit_row['Description'], credit_row['Description'])

            if score > self.highest_score:
                if np.isclose(credit_row['Amount'], -debit_row['Amount'], atol=self.atol):
                    # Exact match: single credit row
                    self.mark_matched([debit_idx, credit_idx], unique_guid)
                    return  # Exit early after exact match

                # Potential for combination match
                matched_credit_indices.append(credit_idx)
                matched_credit_amounts.append(credit_row['Amount'])

        # Check for combination match
        if matched_credit_indices:
            total = sum(matched_credit_amounts)
            if np.isclose(total, -debit_row['Amount'], atol=self.atol):
                all_indices = [debit_idx] + matched_credit_indices
                self.mark_matched(all_indices, unique_guid)
    

    def mark_matched(self, indices, uid):
        for idx in indices:
            self.ledger.at[idx, 'Matched'] = True
            row = self.ledger.loc[idx]
            self.results.append({
                'Journal number': row.get('Journal number', ''),
                'Voucher': row.get('Voucher', ''),
                'Date': row['Date'],
                'Account': row['MainAccount'],
                'CostCentre': row['CostCentre'],
                'ProfitCentre': row['ProfitCentre'],
                'Description': row['Description'],
                'Amount': row['Amount'],
                'Settlement_Number': uid,
                'Extracted_Numbers': row['ExtractedNumbers']
            })

    def write_unmatched(self):
        unmatched = self.ledger[~self.ledger['Matched']]
        for _, row in unmatched.iterrows():
            self.results.append({
                'Journal number': row.get('Journal number', ''),
                'Voucher': row.get('Voucher', ''),
                'Date': row['Date'],
                'Account': row['MainAccount'],
                'CostCentre': row['CostCentre'],
                'ProfitCentre': row['ProfitCentre'],
                'Description': row['Description'],
                'Amount': row['Amount'],
                'Settlement_Number': '',
                'Extracted_Numbers': row['ExtractedNumbers']
            })

    def write_results(self, file):
        matched_df = pd.DataFrame(self.results)
        matched_df.to_csv(file, index=False)
        output = io.StringIO()
        output = matched_df.to_csv (index_label="idx", encoding = "utf-8")
        print('Matching complete. Results saved to matched_transactions.csv')

        blob_service = BlobServiceClient.from_connection_string(CONNSTRING)

        container_client = blob_service.get_container_client(OUTPUTCON)

        my_string = "matched_transactions.csv"

        # Format the datetime object into a string
        # Example format: "YYYY-MM-DD HH:MM:SS"
        current_datetime = datetime.now()
        formatted_datetime_string = current_datetime.strftime("%Y-%m-%d %H:%M:%S")

        # Combine the string and the formatted datetime string
        combined_string = f"{formatted_datetime_string} {my_string}"

        blob_client = blob_service.get_blob_client(container=OUTPUTCON, 
        blob=combined_string)

        blob_client.upload_blob(output,overwrite=True)

        #blobService = BlockBlobService(account_name=accountName, account_key=accountKey)
        #blobService.create_blob_from_text('output', 'matched_transactions.csv', output)

def executeFromBlob(self):
        CONNECTION_STRING = CONNSTRING
        CONTAINER_NAME = INPUTCON
        FOLDER_PREFIX = "data/" # e.g., "data/csv_files/"

        blob_service_client = BlobServiceClient.from_connection_string(CONNECTION_STRING)
        container_client = blob_service_client.get_container_client(CONTAINER_NAME)

        dataframes = {} # Dictionary to store DataFrames, keyed by blob name

        # List blobs within the specified folder prefix
        for blob in container_client.list_blobs(name_starts_with=FOLDER_PREFIX):
            if blob.name.endswith(".csv"):
                print(f"Processing: {blob.name}")

                # Download blob content to a stream
                download_stream = container_client.get_blob_client(blob.name).download_blob()
                
                # Read content into a BytesIO object
                csv_data = io.BytesIO()
                download_stream.readinto(csv_data)
                csv_data.seek(0) # Reset stream position to the beginning
                
                accounts = ['336413']

                settle = LedgerSettlement(3, 0.001, 90, accounts)
                settle.load_ledger(csv_data)
                settle.settle_ledger()
                settle.write_unsettled()
                settle.write_results('matched_transactions.csv')

                container_client.delete_blob(blob.name);
 

#  --- Runner code with execution tracking ---
#accounts = ['336413--812803-National-CRP--']
#settle = LedgerSettlement(27, 0.001, 90, accounts)