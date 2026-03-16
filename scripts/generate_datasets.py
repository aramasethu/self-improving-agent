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
    ("full_name",    " full_name "),       # leading/trailing whitespace
    ("full_name",    "fullname"),
    ("full_name",    "resource"),           # ambiguous: could be anything
    ("full_name",    "personnel"),          # ambiguous HR jargon
    ("full_name",    "col_1"),             # opaque/generic
    ("full_name",    "field_a"),           # opaque/generic
    ("full_name",    "nombre"),            # Spanish
    ("full_name",    "nom_complet"),       # French
    ("full_name",    "bezeichnung"),       # German (means "designation")
    ("full_name",    "data_1"),            # totally opaque
    ("full_name",    "string_field_1"),    # type-based name
    ("full_name",    "txt_primary"),       # cryptic internal
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
    ("email",    "electronic_mail"),
    ("email",    "primary_email"),
    ("email",    "contact"),               # ambiguous: could be phone
    ("email",    "addr"),                   # ambiguous: could be physical address
    ("email",    "col_2"),                 # opaque/generic
    ("email",    "field_b"),              # opaque/generic
    ("email",    "correo"),               # Spanish
    ("email",    "courriel"),             # French
    ("email",    "kontakt"),              # German (means "contact")
    ("email",    "data_2"),               # totally opaque
    ("email",    "string_field_2"),       # type-based name
    ("email",    "notification_target"),  # could be anything
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
    ("department",   "work_group"),
    ("department",   "vertical"),          # ambiguous startup jargon
    ("department",   "practice"),          # ambiguous: consulting term
    ("department",   "col_3"),            # opaque/generic
    ("department",   "area"),             # ambiguous: could be location
    ("department",   "abteilung"),        # German
    ("department",   "equipo"),           # Spanish (means "team")
    ("department",   "pod"),              # startup jargon
    ("department",   "stream"),           # agile jargon
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
    ("start_date",   "commenced"),
    ("start_date",   "start"),
    ("start_date",   "DOJ"),
    ("start_date",   "join_date"),
    ("start_date",   "entry_date"),
    ("start_date",   "col_4"),            # opaque/generic
    ("start_date",   "field_d"),          # opaque/generic
    ("start_date",   "fecha_inicio"),     # Spanish
    ("start_date",   "dt_1"),             # cryptic abbreviation
    ("start_date",   "datum"),            # German/Dutch
    ("start_date",   "effective"),        # ambiguous: effective what?
    ("start_date",   "since"),            # ambiguous
    ("start_date",   "timestamp_1"),      # could be any timestamp
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
    ("salary_usd",   "wage"),
    ("salary_usd",   "gross_salary"),
    ("salary_usd",   "Annual Pay"),
    ("salary_usd",   "band"),              # very ambiguous: could be pay grade
    ("salary_usd",   "TC"),                # ambiguous: "total comp" abbreviation
    ("salary_usd",   "col_5"),            # opaque/generic
    ("salary_usd",   "field_e"),          # opaque/generic
    ("salary_usd",   "sueldo"),           # Spanish
    ("salary_usd",   "num_1"),            # cryptic numeric field
    ("salary_usd",   "amount"),           # ambiguous: could be anything monetary
    ("salary_usd",   "figure"),           # very ambiguous
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
    ("manager_name", "N+1"),               # ambiguous: French corporate term for manager
    ("manager_name", "escalation"),         # ambiguous: could be escalation contact
    ("manager_name", "col_6"),            # opaque/generic
    ("manager_name", "field_f"),          # opaque/generic
    ("manager_name", "jefe"),             # Spanish
    ("manager_name", "responsable"),      # French/Spanish
    ("manager_name", "ref_person"),       # cryptic
    ("manager_name", "string_field_3"),   # type-based name
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

# Semantic trap distractors — column names that look like target fields but aren't.
# These are designed to confuse the LLM into mapping the wrong column.
SEMANTIC_TRAP_COLUMNS = [
    # Looks like full_name but it's a project name
    ("project_name", ["Phoenix", "Atlas", "Mercury", "Titan", "Nebula",
                      "Horizon", "Summit", "Cascade", "Apex", "Nova"]),
    # Looks like email but it's a personal (non-work) email
    ("personal_email", None),  # generated
    # Looks like department but it's the previous department
    ("previous_department", ["Engineering", "Marketing", "Sales", "Finance",
                             "HR", "Product", "Operations", "Legal"]),
    # Looks like start_date but it's the review date
    ("last_review_date", None),  # generated as a date
    # Looks like salary but it's a bonus amount
    ("bonus", None),  # generated as a smaller number
    # Looks like manager but it's the previous manager
    ("previous_manager", ["Jane Doe", "Bob Wilson", "Carlos Rivera",
                          "Diana Lee", "Eva Green", "Tom Brown"]),
    # Another name-like column — emergency contact
    ("emergency_contact", ["Maria Smith", "John Davis", "Lisa Chen",
                           "Robert Kim", "Anna Wilson", "James Lee"]),
    # Another date column — end date
    ("end_date", None),  # generated as a date
    # Looks like it could be salary
    ("budget_code", ["BUD-001", "BUD-002", "BUD-003", "BUD-004", "BUD-005"]),
    # Another name column — nickname
    ("nickname", ["Jim", "Bob", "Liz", "Mike", "Sam", "Pat", "Alex", "Chris"]),
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
SPLIT_NAME_VARIANTS = [
    ("first_name", "last_name"),
    ("fname", "lname"),
    ("given_name", "surname"),
    ("First Name", "Last Name"),
    ("prenom", "nom"),            # French
    ("nombre_pila", "apellido"),  # Spanish
    ("vorname", "nachname"),      # German
    ("first", "last"),
]


def generate_test_case(index: int):
    """Generate one messy CSV + its ground truth mapping."""

    # ~15% of CSVs use split first/last name instead of a single name column
    use_split_name = random.random() < 0.15

    # Pick column name variants (one per target field)
    name_target, name_col = NAME_VARIANTS[index % len(NAME_VARIANTS)]
    email_target, email_col = EMAIL_VARIANTS[index % len(EMAIL_VARIANTS)]
    dept_target, dept_col = DEPT_VARIANTS[index % len(DEPT_VARIANTS)]
    date_target, date_col = DATE_VARIANTS[index % len(DATE_VARIANTS)]
    sal_target, sal_col = SALARY_VARIANTS[index % len(SALARY_VARIANTS)]
    mgr_target, mgr_col = MANAGER_VARIANTS[index % len(MANAGER_VARIANTS)]

    if use_split_name:
        split_first_col, split_last_col = random.choice(SPLIT_NAME_VARIANTS)

    # Pick date and salary formatters
    date_fmt_name, date_formatter = DATE_FORMATTERS[index % len(DATE_FORMATTERS)]
    sal_fmt_name, sal_formatter = SALARY_FORMATTERS[index % len(SALARY_FORMATTERS)]

    # Decide how many extra distractor columns (1-5, more noise)
    num_extra = random.randint(1, 5)
    extra_cols = random.sample(EXTRA_COLUMNS, min(num_extra, len(EXTRA_COLUMNS)))

    # Add semantic trap columns (~40% of CSVs get 1-3 traps)
    num_traps = 0
    trap_cols = []
    if random.random() < 0.40:
        num_traps = random.randint(1, 3)
        trap_cols = random.sample(SEMANTIC_TRAP_COLUMNS, min(num_traps, len(SEMANTIC_TRAP_COLUMNS)))

    # Pick a domain for this "company"
    domain = random.choice(DOMAINS)
    num_rows = random.randint(3, 8)

    # Build the ground truth mapping
    if use_split_name:
        ground_truth_mapping = {
            split_first_col: "full_name__first",
            split_last_col: "full_name__last",
            email_col.strip(): "email",
            dept_col.strip(): "department",
            date_col.strip(): "start_date",
            sal_col.strip(): "salary_usd",
            mgr_col.strip(): "manager_name",
        }
    else:
        ground_truth_mapping = {
            name_col.strip(): "full_name",
            email_col.strip(): "email",
            dept_col.strip(): "department",
            date_col.strip(): "start_date",
            sal_col.strip(): "salary_usd",
            mgr_col.strip(): "manager_name",
        }

    # Generate rows
    if use_split_name:
        headers = [split_first_col, split_last_col, email_col, dept_col, date_col, sal_col, mgr_col]
    else:
        headers = [name_col, email_col, dept_col, date_col, sal_col, mgr_col]
    for extra_name, _ in extra_cols:
        headers.append(extra_name)
    for trap_name, _ in trap_cols:
        headers.append(trap_name)

    # Shuffle column order so target fields aren't always first
    random.shuffle(headers)

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

        if use_split_name:
            row = {
                split_first_col: first,
                split_last_col: last,
                email_col: f"{email_local}@{domain}",
                dept_col: random.choice(DEPARTMENTS),
                date_col: date_formatter(date),
                sal_col: sal_formatter(base_salary),
                mgr_col: random.choice(MANAGERS),
            }
        else:
            row = {
                name_col: full,
                email_col: f"{email_local}@{domain}",
                dept_col: random.choice(DEPARTMENTS),
                date_col: date_formatter(date),
                sal_col: sal_formatter(base_salary),
                mgr_col: random.choice(MANAGERS),
            }

        # Add extra distractor columns
        for extra_name, extra_vals in extra_cols:
            if extra_name == "employee_id":
                row[extra_name] = f"EMP{random.randint(100, 9999):04d}"
            elif extra_name == "phone":
                row[extra_name] = f"+1-{random.randint(200,999)}-{random.randint(100,999)}-{random.randint(1000,9999)}"
            elif extra_name == "badge_number":
                row[extra_name] = f"BDG-{random.randint(1000, 9999)}"
            elif extra_vals:
                row[extra_name] = random.choice(extra_vals)

        # Add semantic trap columns (look like target fields but aren't)
        for trap_name, trap_vals in trap_cols:
            if trap_name == "personal_email":
                pfirst = random.choice(FIRST_NAMES).lower()
                row[trap_name] = f"{pfirst}{random.randint(1,99)}@gmail.com"
            elif trap_name == "last_review_date":
                row[trap_name] = date_formatter(random_date())
            elif trap_name == "bonus":
                row[trap_name] = str(random.randint(2, 20) * 1000)
            elif trap_name == "end_date":
                # Most are empty (still employed), some have a date
                if random.random() < 0.3:
                    row[trap_name] = date_formatter(random_date())
                else:
                    row[trap_name] = ""
            elif trap_vals:
                row[trap_name] = random.choice(trap_vals)

        rows.append(row)

    # Introduce data quality issues for some CSVs
    error_types = []

    # ~20% chance: add a row with missing name
    if random.random() < 0.20 and rows:
        idx = random.randint(0, len(rows) - 1)
        if use_split_name:
            rows[idx][split_first_col] = ""
        else:
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
        if use_split_name:
            rows[idx][split_first_col] = "José"
            rows[idx][split_last_col] = "García-López"
        else:
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
        "num_extra_columns": num_extra + num_traps,
        "extra_columns": [e[0] for e in extra_cols] + [t[0] for t in trap_cols],
        "error_types": error_types,
        "challenges": [],
    }

    # Document what makes this case challenging
    if use_split_name:
        ground_truth["challenges"].append("split_first_last_name")
    if not use_split_name and name_col.strip() != name_col:
        ground_truth["challenges"].append("whitespace_in_header")
    if not use_split_name and (name_col.isupper() or email_col.isupper()):
        ground_truth["challenges"].append("all_caps_headers")
    if not use_split_name and " " in name_col.strip():
        ground_truth["challenges"].append("spaces_in_headers")
    # Opaque/generic column names
    opaque_names = {"col_1", "col_2", "col_3", "col_4", "col_5", "col_6",
                    "field_a", "field_b", "field_d", "field_e", "field_f",
                    "data_1", "data_2", "num_1", "dt_1",
                    "string_field_1", "string_field_2", "string_field_3",
                    "txt_primary"}
    all_col_names = set(ground_truth_mapping.keys())
    if all_col_names & opaque_names:
        ground_truth["challenges"].append("opaque_column_names")
    # Multilingual column names
    multilingual = {"nombre", "nom_complet", "bezeichnung", "correo", "courriel",
                    "kontakt", "abteilung", "equipo", "fecha_inicio", "datum",
                    "sueldo", "jefe", "responsable", "prenom", "nom",
                    "nombre_pila", "apellido", "vorname", "nachname"}
    if all_col_names & multilingual:
        ground_truth["challenges"].append("multilingual_headers")
    if trap_cols:
        ground_truth["challenges"].append("semantic_trap_columns")
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
