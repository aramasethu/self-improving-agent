"""
Generate 100 diverse test CSVs + ground truth mappings.

Each CSV simulates a different company sending employee data in their own format.
The ground truth records the correct column mapping for each CSV.

Error/challenge types:
- Column naming: abbreviations, camelCase, spaces, typos, synonyms, prefixes
- Date formats: ISO, US, EU, written-out, short year, timestamps
- Salary formats: plain, commas, currency symbols, K notation, monthly
- Structural: extra columns, missing columns, split names, whitespace in headers
- Data quality: missing values, inconsistent formats, special characters
"""

import csv
import json
import os
import random
from datetime import datetime, timedelta

random.seed(42)

OUTPUT_DIR = "data/test"
GROUND_TRUTH_FILE = "data/ground_truth.json"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Building blocks for generating diverse column names
# ---------------------------------------------------------------------------
NAME_VARIANTS = [
    ("full_name",    "emp_name"),
    ("full_name",    "employee_name"),
    ("full_name",    "name"),
    ("full_name",    "Full Name"),
    ("full_name",    "EmployeeName"),
    ("full_name",    "worker_name"),
    ("full_name",    "staff_name"),
    ("full_name",    "associate_name"),
    ("full_name",    "Name of Employee"),
    ("full_name",    "emp_full_name"),
    ("full_name",    "person_name"),
    ("full_name",    "display_name"),
    ("full_name",    "empName"),
    ("full_name",    "EMP_NAME"),
    ("full_name",    "EMPLOYEE NAME"),
    ("full_name",    " full_name "),       # leading/trailing whitespace
    ("full_name",    "fullname"),
    ("full_name",    "ename"),
    ("full_name",    "resource"),           # ambiguous: could be anything
    ("full_name",    "personnel"),          # ambiguous HR jargon
]

EMAIL_VARIANTS = [
    ("email",    "email_addr"),
    ("email",    "email"),
    ("email",    "Email Address"),
    ("email",    "work_email"),
    ("email",    "contact_email"),
    ("email",    "emailAddress"),
    ("email",    "e-mail"),
    ("email",    "EMAIL"),
    ("email",    "email_id"),
    ("email",    "corporate_email"),
    ("email",    "emp_email"),
    ("email",    "mail"),
    ("email",    "Email"),
    ("email",    "electronic_mail"),
    ("email",    "user_email"),
    ("email",    "office_email"),
    ("email",    "work_email_address"),
    ("email",    "primary_email"),
    ("email",    "contact"),               # ambiguous: could be phone
    ("email",    "addr"),                   # ambiguous: could be physical address
]

DEPT_VARIANTS = [
    ("department",   "dept"),
    ("department",   "department"),
    ("department",   "Department"),
    ("department",   "division"),
    ("department",   "team"),
    ("department",   "dept_name"),
    ("department",   "business_unit"),
    ("department",   "org_unit"),
    ("department",   "group"),
    ("department",   "DEPT"),
    ("department",   "Department Name"),
    ("department",   "section"),
    ("department",   "unit"),
    ("department",   "functional_area"),
    ("department",   "cost_center"),
    ("department",   "dept_code"),
    ("department",   "work_group"),
    ("department",   "departmentName"),
    ("department",   "vertical"),          # ambiguous startup jargon
    ("department",   "practice"),          # ambiguous: consulting term
]

DATE_VARIANTS = [
    ("start_date",   "hire_dt"),
    ("start_date",   "start_date"),
    ("start_date",   "Start Date"),
    ("start_date",   "hire_date"),
    ("start_date",   "date_joined"),
    ("start_date",   "joining_date"),
    ("start_date",   "employment_start"),
    ("start_date",   "onboard_date"),
    ("start_date",   "date_of_joining"),
    ("start_date",   "startDate"),
    ("start_date",   "HIRE_DATE"),
    ("start_date",   "begin_date"),
    ("start_date",   "commenced"),
    ("start_date",   "start"),
    ("start_date",   "Date Started"),
    ("start_date",   "DOJ"),
    ("start_date",   "join_date"),
    ("start_date",   "entry_date"),
    ("start_date",   "employment_date"),
    ("start_date",   "hired_on"),
]

SALARY_VARIANTS = [
    ("salary_usd",   "annual_sal"),
    ("salary_usd",   "salary"),
    ("salary_usd",   "Salary"),
    ("salary_usd",   "compensation"),
    ("salary_usd",   "Compensation (USD)"),
    ("salary_usd",   "annual_salary"),
    ("salary_usd",   "pay"),
    ("salary_usd",   "base_salary"),
    ("salary_usd",   "total_comp"),
    ("salary_usd",   "SALARY"),
    ("salary_usd",   "yearly_pay"),
    ("salary_usd",   "remuneration"),
    ("salary_usd",   "ctc"),               # ambiguous: "cost to company"
    ("salary_usd",   "annual_compensation"),
    ("salary_usd",   "wage"),
    ("salary_usd",   "salary_usd"),
    ("salary_usd",   "gross_salary"),
    ("salary_usd",   "Annual Pay"),
    ("salary_usd",   "band"),              # very ambiguous: could be pay grade
    ("salary_usd",   "TC"),                # ambiguous: "total comp" abbreviation
]

MANAGER_VARIANTS = [
    ("manager_name", "mgr"),
    ("manager_name", "manager"),
    ("manager_name", "Manager"),
    ("manager_name", "manager_name"),
    ("manager_name", "supervisor"),
    ("manager_name", "reports_to"),
    ("manager_name", "Reports To"),
    ("manager_name", "line_manager"),
    ("manager_name", "direct_manager"),
    ("manager_name", "boss"),
    ("manager_name", "MGR"),
    ("manager_name", "reporting_manager"),
    ("manager_name", "managed_by"),
    ("manager_name", "team_lead"),
    ("manager_name", "mgr_name"),
    ("manager_name", "superior"),
    ("manager_name", "manager_full_name"),
    ("manager_name", "Supervisor"),
    ("manager_name", "N+1"),               # ambiguous: French corporate term for manager
    ("manager_name", "escalation"),         # ambiguous: could be escalation contact
]

# Shuffle variant lists so hard names are distributed across all batches
random.shuffle(NAME_VARIANTS)
random.shuffle(EMAIL_VARIANTS)
random.shuffle(DEPT_VARIANTS)
random.shuffle(DATE_VARIANTS)
random.shuffle(SALARY_VARIANTS)
random.shuffle(MANAGER_VARIANTS)

# Extra columns to randomly add as distractors
EXTRA_COLUMNS = [
    ("location", ["San Francisco", "New York", "Remote", "Chicago", "Austin",
                   "London", "Berlin", "Toronto", "Seattle", "Boston"]),
    ("employee_id", None),  # generated
    ("phone", None),  # generated
    ("title", ["Software Engineer", "Product Manager", "Data Analyst",
               "Designer", "DevOps Engineer", "QA Engineer", "Tech Lead",
               "VP Engineering", "Director", "Intern"]),
    ("status", ["Active", "active", "ACTIVE", "On Leave", "Probation"]),
    ("office_floor", ["1", "2", "3", "4", "5", "G", "B1"]),
    ("badge_number", None),  # generated
    ("nationality", ["US", "CA", "UK", "DE", "IN", "JP", "BR", "AU", "FR", "KR"]),
    ("gender", ["M", "F", "Male", "Female", "Non-binary", "Prefer not to say"]),
    ("blood_type", ["A+", "A-", "B+", "B-", "O+", "O-", "AB+", "AB-"]),
]

# ---------------------------------------------------------------------------
# Sample people data
# ---------------------------------------------------------------------------
FIRST_NAMES = [
    "James", "Mary", "Robert", "Patricia", "John", "Jennifer", "Michael",
    "Linda", "David", "Elizabeth", "William", "Barbara", "Richard", "Susan",
    "Joseph", "Jessica", "Thomas", "Sarah", "Daniel", "Karen", "Matthew",
    "Lisa", "Anthony", "Nancy", "Mark", "Betty", "Andrew", "Margaret",
    "Joshua", "Sandra", "Wei", "Yuki", "Priya", "Ahmed", "Fatima",
    "Carlos", "Ana", "Olga", "Raj", "Mei", "Sven", "Ingrid", "Kenji",
    "Aisha", "Boris", "Elena", "Hassan", "Lena", "Tariq", "Nadia",
]

LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller",
    "Davis", "Rodriguez", "Martinez", "Wilson", "Anderson", "Taylor",
    "Thomas", "Hernandez", "Moore", "Martin", "Lee", "Clark", "Lewis",
    "Chen", "Wang", "Kim", "Patel", "Singh", "Tanaka", "Mueller",
    "Johansson", "Petrov", "Santos", "O'Brien", "Al-Rashid", "Nakamura",
    "van der Berg", "Kowalski", "Fernandez", "Nguyen", "Park", "Liu", "Das",
]

DEPARTMENTS = [
    "Engineering", "Marketing", "Sales", "Finance", "HR",
    "Product", "Operations", "Legal", "Customer Success", "R&D",
    "Data Science", "Design", "IT", "Security", "QA",
    "DevOps", "Business Development", "Analytics", "Support", "Compliance",
]

DOMAINS = [
    "acme.com", "globex.com", "initech.io", "contoso.com", "northwind.com",
    "fabrikam.com", "tailspin.co", "widgetworks.com", "techcorp.io", "datapipe.com",
]

MANAGERS = [
    "Jane Doe", "Bob Wilson", "Carlos Rivera", "Diana Lee", "Eva Green",
    "Michael Scott", "Pam Beesly", "Tom Brown", "Sarah Connor", "Alex Kim",
]

# ---------------------------------------------------------------------------
# Date format generators
# ---------------------------------------------------------------------------
def random_date():
    start = datetime(2019, 1, 1)
    end = datetime(2024, 12, 31)
    delta = end - start
    return start + timedelta(days=random.randint(0, delta.days))


DATE_FORMATTERS = [
    ("iso",           lambda d: d.strftime("%Y-%m-%d")),
    ("us_slash",      lambda d: d.strftime("%m/%d/%Y")),
    ("eu_slash",      lambda d: d.strftime("%d/%m/%Y")),
    ("us_dash",       lambda d: d.strftime("%m-%d-%Y")),
    ("eu_dash",       lambda d: d.strftime("%d-%m-%Y")),
    ("written_long",  lambda d: d.strftime("%B %d, %Y")),
    ("written_short", lambda d: d.strftime("%b %d %Y")),
    ("written_eu",    lambda d: d.strftime("%d %B %Y")),
    ("short_year",    lambda d: d.strftime("%m/%d/%y")),
    ("timestamp",     lambda d: d.strftime("%Y-%m-%dT00:00:00")),
    ("dot_eu",        lambda d: d.strftime("%d.%m.%Y")),
    ("year_first",    lambda d: d.strftime("%Y/%m/%d")),
    ("compact",       lambda d: d.strftime("%Y%m%d")),
    ("written_day",   lambda d: d.strftime("%d %b, %Y")),
    ("month_name",    lambda d: d.strftime("%B %d %Y")),
]

# ---------------------------------------------------------------------------
# Salary format generators
# ---------------------------------------------------------------------------
def format_salary_plain(val):
    return str(val)

def format_salary_commas(val):
    return f"{val:,}"

def format_salary_dollar(val):
    return f"${val:,}"

def format_salary_usd_prefix(val):
    return f"USD {val:,}"

def format_salary_k(val):
    return f"{val // 1000}K"

def format_salary_k_lower(val):
    return f"{val // 1000}k"

def format_salary_decimal(val):
    return f"{val}.00"

def format_salary_dollar_nodec(val):
    return f"${val}"

def format_salary_space(val):
    s = str(val)
    return f"{s[:-3]} {s[-3:]}" if len(s) > 3 else s

def format_salary_monthly(val):
    """Return monthly salary — a tricky case the agent might mishandle."""
    return str(round(val / 12, 2))

SALARY_FORMATTERS = [
    ("plain",        format_salary_plain),
    ("commas",       format_salary_commas),
    ("dollar",       format_salary_dollar),
    ("usd_prefix",   format_salary_usd_prefix),
    ("k_notation",   format_salary_k),
    ("k_lower",      format_salary_k_lower),
    ("decimal",      format_salary_decimal),
    ("dollar_nodec", format_salary_dollar_nodec),
    ("space_sep",    format_salary_space),
    ("monthly",      format_salary_monthly),
]


# ---------------------------------------------------------------------------
# Generate a single CSV test case
# ---------------------------------------------------------------------------
def generate_test_case(index: int):
    """Generate one messy CSV + its ground truth mapping."""

    # Pick column name variants (one per target field)
    name_target, name_col = NAME_VARIANTS[index % len(NAME_VARIANTS)]
    email_target, email_col = EMAIL_VARIANTS[index % len(EMAIL_VARIANTS)]
    dept_target, dept_col = DEPT_VARIANTS[index % len(DEPT_VARIANTS)]
    date_target, date_col = DATE_VARIANTS[index % len(DATE_VARIANTS)]
    sal_target, sal_col = SALARY_VARIANTS[index % len(SALARY_VARIANTS)]
    mgr_target, mgr_col = MANAGER_VARIANTS[index % len(MANAGER_VARIANTS)]

    # Pick date and salary formatters
    date_fmt_name, date_formatter = DATE_FORMATTERS[index % len(DATE_FORMATTERS)]
    sal_fmt_name, sal_formatter = SALARY_FORMATTERS[index % len(SALARY_FORMATTERS)]

    # Decide how many extra distractor columns (0-3)
    num_extra = random.randint(0, 3)
    extra_cols = random.sample(EXTRA_COLUMNS, min(num_extra, len(EXTRA_COLUMNS)))

    # Pick a domain for this "company"
    domain = random.choice(DOMAINS)
    num_rows = random.randint(3, 8)

    # Build the ground truth mapping
    ground_truth_mapping = {
        name_col.strip(): "full_name",
        email_col.strip(): "email",
        dept_col.strip(): "department",
        date_col.strip(): "start_date",
        sal_col.strip(): "salary_usd",
        mgr_col.strip(): "manager_name",
    }

    # Generate rows
    headers = [name_col, email_col, dept_col, date_col, sal_col, mgr_col]
    for extra_name, _ in extra_cols:
        headers.append(extra_name)

    rows = []
    used_names = set()
    for _ in range(num_rows):
        first = random.choice(FIRST_NAMES)
        last = random.choice(LAST_NAMES)
        full = f"{first} {last}"
        while full in used_names:
            first = random.choice(FIRST_NAMES)
            last = random.choice(LAST_NAMES)
            full = f"{first} {last}"
        used_names.add(full)

        email_local = f"{first[0].lower()}{last.lower()}"
        base_salary = random.randint(60, 200) * 1000
        date = random_date()

        row = {
            name_col: full,
            email_col: f"{email_local}@{domain}",
            dept_col: random.choice(DEPARTMENTS),
            date_col: date_formatter(date),
            sal_col: sal_formatter(base_salary),
            mgr_col: random.choice(MANAGERS),
        }

        # Add extra columns
        for extra_name, extra_vals in extra_cols:
            if extra_name == "employee_id":
                row[extra_name] = f"EMP{random.randint(100, 9999):04d}"
            elif extra_name == "phone":
                row[extra_name] = f"+1-{random.randint(200,999)}-{random.randint(100,999)}-{random.randint(1000,9999)}"
            elif extra_name == "badge_number":
                row[extra_name] = f"BDG-{random.randint(1000, 9999)}"
            elif extra_vals:
                row[extra_name] = random.choice(extra_vals)

        rows.append(row)

    # Introduce data quality issues for some CSVs
    error_types = []

    # ~20% chance: add a row with missing name
    if random.random() < 0.20 and rows:
        idx = random.randint(0, len(rows) - 1)
        rows[idx][name_col] = ""
        error_types.append("missing_name")

    # ~15% chance: add a row with missing email
    if random.random() < 0.15 and rows:
        idx = random.randint(0, len(rows) - 1)
        rows[idx][email_col] = ""
        error_types.append("missing_email")

    # ~15% chance: add a row with missing manager
    if random.random() < 0.15 and rows:
        idx = random.randint(0, len(rows) - 1)
        rows[idx][mgr_col] = ""
        error_types.append("missing_manager")

    # ~10% chance: inconsistent date format in one row
    if random.random() < 0.10 and rows:
        idx = random.randint(0, len(rows) - 1)
        alt_fmt_name, alt_formatter = random.choice(DATE_FORMATTERS)
        rows[idx][date_col] = alt_formatter(random_date())
        error_types.append(f"inconsistent_date_{alt_fmt_name}")

    # ~10% chance: salary with unexpected characters in one row
    if random.random() < 0.10 and rows:
        idx = random.randint(0, len(rows) - 1)
        base = random.randint(60, 200) * 1000
        rows[idx][sal_col] = f"~{base} approx"
        error_types.append("messy_salary")

    # ~10% chance: special characters in name
    if random.random() < 0.10 and rows:
        idx = random.randint(0, len(rows) - 1)
        rows[idx][name_col] = "José García-López"
        error_types.append("special_chars_name")

    # Write CSV
    filename = f"test_{index:03d}.csv"
    filepath = os.path.join(OUTPUT_DIR, filename)

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)

    # Ground truth entry
    ground_truth = {
        "csv_file": filename,
        "column_mapping": ground_truth_mapping,
        "date_format": date_fmt_name,
        "salary_format": sal_fmt_name,
        "num_rows": num_rows,
        "num_extra_columns": num_extra,
        "extra_columns": [e[0] for e in extra_cols],
        "error_types": error_types,
        "challenges": [],
    }

    # Document what makes this case challenging
    if name_col.strip() != name_col:
        ground_truth["challenges"].append("whitespace_in_header")
    if name_col.isupper() or email_col.isupper():
        ground_truth["challenges"].append("all_caps_headers")
    if " " in name_col.strip():
        ground_truth["challenges"].append("spaces_in_headers")
    if sal_fmt_name == "monthly":
        ground_truth["challenges"].append("monthly_salary_not_annual")
    if sal_fmt_name in ("dollar", "usd_prefix", "dollar_nodec"):
        ground_truth["challenges"].append("currency_symbols_in_salary")
    if sal_fmt_name in ("k_notation", "k_lower"):
        ground_truth["challenges"].append("k_notation_salary")
    if sal_fmt_name == "commas":
        ground_truth["challenges"].append("commas_in_salary")
    if date_fmt_name in ("eu_slash", "eu_dash", "dot_eu"):
        ground_truth["challenges"].append("ambiguous_date_format_eu")
    if date_fmt_name in ("written_long", "written_short", "written_eu",
                          "written_day", "month_name"):
        ground_truth["challenges"].append("written_date_format")
    if date_fmt_name == "compact":
        ground_truth["challenges"].append("compact_date_YYYYMMDD")
    if date_fmt_name == "timestamp":
        ground_truth["challenges"].append("timestamp_date")
    if num_extra > 0:
        ground_truth["challenges"].append("extra_distractor_columns")
    if error_types:
        ground_truth["challenges"].extend(error_types)

    return ground_truth


# ---------------------------------------------------------------------------
# Main: Generate all 100 test cases
# ---------------------------------------------------------------------------
def main():
    all_ground_truth = []

    for i in range(100):
        gt = generate_test_case(i)
        all_ground_truth.append(gt)

    # Write ground truth
    with open(GROUND_TRUTH_FILE, "w") as f:
        json.dump(all_ground_truth, f, indent=2)

    # Summary
    print(f"Generated {len(all_ground_truth)} test CSVs in {OUTPUT_DIR}/")
    print(f"Ground truth saved to {GROUND_TRUTH_FILE}")

    # Stats
    all_challenges = {}
    for gt in all_ground_truth:
        for c in gt["challenges"]:
            all_challenges[c] = all_challenges.get(c, 0) + 1

    print(f"\nChallenge distribution:")
    for challenge, count in sorted(all_challenges.items(), key=lambda x: -x[1]):
        print(f"  {challenge}: {count}")


if __name__ == "__main__":
    main()
