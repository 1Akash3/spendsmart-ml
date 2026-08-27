import re
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

try:
    import pdfplumber
except ImportError:
    pdfplumber = None

from src.schema import Transaction

logger = logging.getLogger(__name__)

class GPayPDFParser:
    """Robust parser for Google Pay PDF statements.
    
    Extracts timestamps, amounts, merchant names, and directions (debit/credit).
    Performs reconciliation of extracted totals against statement summary totals.
    """
    
    def __init__(self, timezone_assumption: str = "Asia/Kolkata"):
        self.timezone_assumption = timezone_assumption
        self.report = {
            "pages_processed": 0,
            "transactions_extracted": 0,
            "transactions_rejected": 0,
            "rejection_reasons": {},
            "debit_total_extracted": 0.0,
            "credit_total_extracted": 0.0,
            "statement_debit_total": None,
            "statement_credit_total": None,
            "reconciliation_status": "PENDING",
            "extraction_warnings": []
        }

    def _log_rejection(self, reason: str):
        self.report["transactions_rejected"] += 1
        self.report["rejection_reasons"][reason] = self.report["rejection_reasons"].get(reason, 0) + 1

    def parse(self, pdf_path: str, user_id: str) -> List[Transaction]:
        if pdfplumber is None:
            raise ImportError("pdfplumber is required for PDF ingestion.")
            
        transactions = []
        path = Path(pdf_path)
        
        with pdfplumber.open(path) as pdf:
            self.report["pages_processed"] = len(pdf.pages)
            for page_num, page in enumerate(pdf.pages, 1):
                text = page.extract_text() or ""
                self._parse_page_text(text, user_id, transactions, page_num)
                
        # Attempt to reconcile
        self._reconcile()
        
        # Save parse report
        report_path = path.parent / f"{path.stem}_parse_report.json"
        with open(report_path, "w") as f:
            json.dump(self.report, f, indent=2)
            
        return transactions

    def _parse_page_text(self, text: str, user_id: str, transactions: List[Transaction], page_num: int):
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        # Simple heuristic state machine to parse multiline transactions
        current_date_str = None
        
        for i, line in enumerate(lines):
            # Attempt to extract statement totals if present
            if "Total Debits" in line or "Total Withdrawals" in line:
                m = re.search(r'₹?\s*([\d,]+\.?\d*)', line)
                if m: self.report["statement_debit_total"] = float(m.group(1).replace(',', ''))
            elif "Total Credits" in line or "Total Deposits" in line:
                m = re.search(r'₹?\s*([\d,]+\.?\d*)', line)
                if m: self.report["statement_credit_total"] = float(m.group(1).replace(',', ''))

            # Extract date (e.g. "Mar 07 2026", "07 Mar 2026")
            # Note: GPay PDF formats vary wildly. This is a generic robust extractor.
            date_match = re.match(r'^([A-Z][a-z]{2}\s\d{1,2},?\s\d{4})', line)
            if date_match:
                current_date_str = date_match.group(1)
                
            # Match Transaction lines
            # Example: Paid to Starbucks ₹ 450.00 21:09:00
            # Example: Received from John ₹ 1,000.00
            txn_match = re.search(r'^(Paid to|Received from|Sent to)\s+(.+?)\s+₹\s*([\d,]+\.?\d*)\s*(?:(\d{2}:\d{2}:\d{2}))?', line)
            
            if txn_match:
                direction_str, merchant, amount_str, time_str = txn_match.groups()
                amount = float(amount_str.replace(',', ''))
                
                direction = "debit" if direction_str in ["Paid to", "Sent to"] else "credit"
                
                # Timestamp resolution
                dt = None
                if current_date_str:
                    try:
                        # Default time if missing
                        t_str = time_str if time_str else "00:00:00"
                        dt_str = f"{current_date_str.replace(',','')} {t_str}"
                        dt = datetime.strptime(dt_str, "%b %d %Y %H:%M:%S")
                    except ValueError:
                        self.report["extraction_warnings"].append(f"Could not parse date on page {page_num}: {current_date_str} {time_str}")
                
                if dt is None:
                    self._log_rejection("Missing or invalid timestamp")
                    continue
                    
                txn = Transaction(
                    user_id=user_id,
                    timestamp=dt,
                    amount=amount,
                    direction=direction,
                    merchant_raw=merchant.strip(),
                    source="gpay_pdf"
                )
                
                transactions.append(txn)
                self.report["transactions_extracted"] += 1
                
                if direction == "debit":
                    self.report["debit_total_extracted"] += amount
                else:
                    self.report["credit_total_extracted"] += amount

    def _reconcile(self):
        """Reconciles extracted totals vs statement totals."""
        status = "SUCCESS"
        tolerance = 1.0 # 1 rupee tolerance for float rounding
        
        if self.report["statement_debit_total"] is not None:
            diff = abs(self.report["debit_total_extracted"] - self.report["statement_debit_total"])
            if diff > tolerance:
                status = "FAILED"
                self.report["extraction_warnings"].append(f"Debit mismatch: Extracted {self.report['debit_total_extracted']} != Statement {self.report['statement_debit_total']}")
                
        if self.report["statement_credit_total"] is not None:
            diff = abs(self.report["credit_total_extracted"] - self.report["statement_credit_total"])
            if diff > tolerance:
                status = "FAILED"
                self.report["extraction_warnings"].append(f"Credit mismatch: Extracted {self.report['credit_total_extracted']} != Statement {self.report['statement_credit_total']}")
                
        if self.report["statement_debit_total"] is None and self.report["statement_credit_total"] is None:
            status = "NO_STATEMENT_TOTALS"
            
        self.report["reconciliation_status"] = status
        
        if status == "FAILED":
            logger.warning(f"Reconciliation failed. See parse_report.json for details.")
