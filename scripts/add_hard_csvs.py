"""Add harder CSVs to the test set that reliably cause failures.

Pattern 1 (test_100-104): European salary format — "85.000,50€"
Pattern 2 (test_105-109): Monthly salary, no hint in column name

These are designed to fail on the baseline agent and get fixed by self-improvement rules.
"""

import csv
import json
import os
import random

random.seed(42)

TEST_DIR = "data/test"
GT_PATH = "data/ground_truth.json"

FIRST_NAMES = ["Tariq", "Yuki", "Jessica", "Ahmed", "Olga", "Robert", "Ingrid", "Carlos", "Diana", "Marcus",
               "Svetlana", "Kenji", "Fatima", "Hans", "Priya", "Miguel", "Astrid", "Wei", "Lena", "Omar"]
LAST_NAMES = ["Mueller", "Singh", "Kim", "Johansson", "Garcia", "Brown", "Anderson", "Rivera", "Lee", "Wilson",
              "Kowalski", "Nguyen", "Schmidt", "Taylor", "Martinez", "Fischer", "Petrov", "Chen", "Berg", "Ali"]
DEPARTMENTS = ["Engineering", "Sales", "Marketing", "Finance", "HR", "R&D", "Product", "Analytics", "DevOps", "QA"]
MANAGERS = ["Tom Brown", "Jane Doe", "Michael Scott", "Sarah Connor", "Pam Beesly", "Bob Wilson",
            "Carlos Rivera", "Eva Green", "Diana Lee", "Alex Kim"]
DOMAINS = ["acme.com", "globex.de", "initech.co.uk", "datapipe.eu", "northwind.se"]

EURO_DATES = ["15.03.2023", "01.11.2021", "28.06.2024", "03.09.2020", "17.12.2022",
              "22.01.2019", "09.07.2023", "14.05.2021", "30.08.2024", "11.02.2020"]

SALARY_FORMATS = [
    lambda a: f"${a:,}",           # $85,000
    lambda a: f"{a//1000}K",       # 85K
    lambda a: f"{a//1000}k",       # 85k
    lambda a: f"${a//1000}k",      # $85k
    lambda a: f"{a/12:,.2f}/mo",   # 7,083.33/mo
    lambda a: f"€{a:,}",           # €85,000
    lambda a: f"{a} USD",          # 85000 USD
]


def make_mixed_salary_csv(idx: int) -> tuple[str, dict]:
    """CSV with mixed salary formats in the same column — some rows $85K, others 7083/mo, others $85,000."""
    fname = f"test_{idx:03d}.csv"
    rows = []
    gt_rows = []

    col_sets = [
        {"name": "employee_name", "email": "work_email", "dept": "department",
         "date": "start_date", "salary": "annual_pay", "mgr": "manager"},
        {"name": "staff", "email": "email", "dept": "team",
         "date": "hire_date", "salary": "salary_info", "mgr": "reports_to"},
        {"name": "name", "email": "contact", "dept": "division",
         "date": "joined", "salary": "comp_package", "mgr": "lead"},
        {"name": "full_name", "email": "email_addr", "dept": "org_unit",
         "date": "onboard_date", "salary": "total_comp", "mgr": "supervisor"},
        {"name": "worker", "email": "mail", "dept": "area",
         "date": "commenced", "salary": "remuneration", "mgr": "boss"},
    ]
    cols = col_sets[idx % len(col_sets)]
    extra = {"status": random.choice(["Active", "Probation"]), "location": "HQ"}

    n_rows = random.randint(6, 9)
    for i in range(n_rows):
        first = random.choice(FIRST_NAMES)
        last = random.choice(LAST_NAMES)
        name = f"{first} {last}"
        domain = random.choice(DOMAINS)
        email = f"{first[0].lower()}{last.lower()}@{domain}"
        dept = random.choice(DEPARTMENTS)

        month = random.randint(1, 12)
        day = random.randint(1, 28)
        year = random.randint(2019, 2025)
        date_str = f"{year}-{month:02d}-{day:02d}"

        annual = random.randint(60, 200) * 1000
        # Each row gets a DIFFERENT salary format
        fmt_fn = SALARY_FORMATS[i % len(SALARY_FORMATS)]
        salary_str = fmt_fn(annual)
        mgr = random.choice(MANAGERS)

        row = {
            cols["name"]: name,
            cols["email"]: email,
            cols["dept"]: dept,
            cols["date"]: date_str,
            cols["salary"]: salary_str,
            cols["mgr"]: mgr,
        }
        row.update(extra)
        rows.append(row)

        gt_rows.append({
            "full_name": name,
            "email": email,
            "department": dept,
            "start_date": date_str,
            "salary_usd": float(annual),
            "manager_name": mgr,
        })

    path = os.path.join(TEST_DIR, fname)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    col_mapping = {
        cols["name"]: "full_name",
        cols["email"]: "email",
        cols["dept"]: "department",
        cols["date"]: "start_date",
        cols["salary"]: "salary_usd",
        cols["mgr"]: "manager_name",
    }

    gt_entry = {
        "csv_file": fname,
        "column_mapping": col_mapping,
        "expected_output": gt_rows,
        "challenges": ["mixed_salary_formats", "extra_distractor_columns"],
        "date_format": "YYYY-MM-DD",
        "salary_format": "mixed_per_row",
    }

    print(f"  Created {fname}: {n_rows} rows, salary formats like '{rows[0][cols['salary']]}', '{rows[1][cols['salary']]}', '{rows[2][cols['salary']]}'")
    return fname, gt_entry


def make_monthly_salary_csv(idx: int) -> tuple[str, dict]:
    """CSV with monthly salary values — column name doesn't say 'monthly'."""
    fname = f"test_{idx:03d}.csv"
    rows = []
    gt_rows = []

    # Column names that don't hint at monthly
    col_sets = [
        {"name": "worker_name", "email": "work_email", "dept": "unit",
         "date": "hire_date", "salary": "compensation", "mgr": "reports_to"},
        {"name": "associate", "email": "email", "dept": "group",
         "date": "onboard_date", "salary": "pay_rate", "mgr": "supervisor"},
        {"name": "staff_name", "email": "staff_email", "dept": "team",
         "date": "joined", "salary": "gross_pay", "mgr": "lead"},
        {"name": "emp_name", "email": "email_id", "dept": "function",
         "date": "start_date", "salary": "base_comp", "mgr": "manager"},
        {"name": "nombre", "email": "correo", "dept": "área",
         "date": "fecha_inicio", "salary": "remuneración", "mgr": "jefe"},
    ]
    cols = col_sets[idx % len(col_sets)]

    # Extra columns including one that hints at frequency
    extra = {"pay_frequency": "monthly", "currency": "USD", "employee_id": f"EMP-{random.randint(100,999)}"}

    n_rows = random.randint(5, 8)
    for i in range(n_rows):
        first = random.choice(FIRST_NAMES)
        last = random.choice(LAST_NAMES)
        name = f"{first} {last}"
        domain = random.choice(DOMAINS)
        email = f"{first[0].lower()}{last.lower()}@{domain}"
        dept = random.choice(DEPARTMENTS)

        # Simple ISO date format (avoid datetime failures muddying the salary test)
        month = random.randint(1, 12)
        day = random.randint(1, 28)
        year = random.randint(2019, 2025)
        date_str = f"{year}-{month:02d}-{day:02d}"

        annual = random.randint(60, 200) * 1000
        monthly = round(annual / 12, 2)
        mgr = random.choice(MANAGERS)

        row = {
            cols["name"]: name,
            cols["email"]: email,
            cols["dept"]: dept,
            cols["date"]: date_str,
            cols["salary"]: monthly,
            cols["mgr"]: mgr,
        }
        row.update(extra)
        rows.append(row)

        gt_rows.append({
            "full_name": name,
            "email": email,
            "department": dept,
            "start_date": date_str,
            "salary_usd": float(annual),
            "manager_name": mgr,
        })

    # Write CSV
    path = os.path.join(TEST_DIR, fname)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    col_mapping = {
        cols["name"]: "full_name",
        cols["email"]: "email",
        cols["dept"]: "department",
        cols["date"]: "start_date",
        cols["salary"]: "salary_usd",
        cols["mgr"]: "manager_name",
    }

    gt_entry = {
        "csv_file": fname,
        "column_mapping": col_mapping,
        "expected_output": gt_rows,
        "challenges": ["monthly_salary_not_annual", "extra_distractor_columns"],
        "date_format": "MM/DD/YYYY",
        "salary_format": "monthly_numeric",
    }

    print(f"  Created {fname}: {n_rows} rows, monthly salary like {rows[0][cols['salary']]}")
    return fname, gt_entry


def main():
    # Load existing ground truth
    with open(GT_PATH) as f:
        gt = json.load(f)

    # Remove any previously generated hard CSVs (100+)
    gt = [e for e in gt if int(e["csv_file"].split("_")[1].split(".")[0]) < 100]

    new_entries = []

    print("Creating mixed salary format CSVs (test_100-104):")
    for i in range(100, 105):
        _, entry = make_mixed_salary_csv(i)
        new_entries.append(entry)

    print("\nCreating monthly salary CSVs (test_105-109):")
    for i in range(105, 110):
        _, entry = make_monthly_salary_csv(i)
        new_entries.append(entry)

    # Append to ground truth
    gt.extend(new_entries)
    with open(GT_PATH, "w") as f:
        json.dump(gt, f, indent=2, default=str)

    print(f"\nGround truth updated: {len(gt)} entries total")
    print("New CSVs: test_100.csv through test_109.csv")


if __name__ == "__main__":
    main()
