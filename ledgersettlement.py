import pandas as pd
import numpy as np
import re
import uuid
from thefuzz import thefuzz 
import datetime
import itertools

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
        pd.DataFrame(self.results).to_csv(file, index=False)



# --- Runner code with execution tracking ---

accounts = ['336413--812803-National-CRP--']

exec_start = datetime.datetime.now()

settle = LedgerSettlement(27, 0.001, 90, accounts)

print("Starting file load...")
exec_fileload_start = datetime.datetime.now()
settle.load_ledger('LTData_small.csv')
exec_fileload_end = datetime.datetime.now()
print("File load complete.") 

print("Starting settlement...")
exec_settlement_start = datetime.datetime.now()
settle.settle()
settle.write_unmatched()
exec_settlement_end = datetime.datetime.now()
print("Settlement complete.")

print("Starting file write...")
exec_filewrite_start = datetime.datetime.now()
settle.write_results('matched_transactions.csv')
exec_end = datetime.datetime.now()
print("File write complete.")

print("")
print(f"File load execution time: {(exec_fileload_end - exec_fileload_start).total_seconds()} seconds")
print(f"Settlement execution time: {(exec_settlement_end - exec_settlement_start).total_seconds()} seconds")
print(f"File write execution time: {(exec_end - exec_filewrite_start).total_seconds()} seconds")
print(f"Total execution time: {(exec_end - exec_start).total_seconds()} seconds")
