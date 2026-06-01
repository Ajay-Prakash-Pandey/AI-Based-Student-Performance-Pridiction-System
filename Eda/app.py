from pathlib import Path
import pickle

import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
REG_MODEL_PATH = MODELS_DIR / "student_reg_model.pkl"
CLF_MODEL_PATH = MODELS_DIR / "student_clf_model.pkl"
METADATA_PATH = MODELS_DIR / "student_model_metadata.pkl"
LEGACY_REG_MODEL_PATH = BASE_DIR / "student_reg_model.pkl"
LEGACY_CLF_MODEL_PATH = BASE_DIR / "student_clf_model.pkl"
LEGACY_METADATA_PATH = BASE_DIR / "student_model_metadata.pkl"

SCHOOL_BOARDS = [
    "CBSE",
    "CISCE - ICSE/ISC",
    "NIOS",
    "Andhra Pradesh State Board",
    "Arunachal Pradesh State Board",
    "Assam State Board",
    "Bihar State Board",
    "Chhattisgarh State Board",
    "Goa State Board",
    "Gujarat State Board",
    "Haryana State Board",
    "Himachal Pradesh State Board",
    "Jammu and Kashmir Board",
    "Jharkhand State Board",
    "Karnataka State Board",
    "Kerala State Board",
    "Madhya Pradesh State Board",
    "Maharashtra State Board",
    "Manipur State Board",
    "Meghalaya State Board",
    "Mizoram State Board",
    "Nagaland State Board",
    "Odisha State Board",
    "Punjab State Board",
    "Rajasthan State Board",
    "Sikkim State Board",
    "Tamil Nadu State Board",
    "Telangana State Board",
    "Tripura State Board",
    "Uttar Pradesh State Board",
    "Uttarakhand State Board",
    "West Bengal State Board",
    "Andaman and Nicobar UT Board/CBSE",
    "Chandigarh UT Board/CBSE",
    "Dadra and Nagar Haveli and Daman and Diu UT Board/CBSE",
    "Delhi Directorate/CBSE",
    "Ladakh UT Board/CBSE",
    "Lakshadweep UT Board/CBSE",
    "Puducherry UT Board/CBSE",
]

COLLEGE_SYSTEMS = [
    "Central University",
    "State University",
    "Private University",
    "Deemed University",
    "Autonomous College",
    "Affiliated College",
    "IIT",
    "NIT",
    "IIIT",
    "IIM",
    "AIIMS/Medical University",
    "Agricultural University",
    "Open University",
    "Distance Education",
]

BACHELOR_DEGREES = [
    "BA", "BSc", "BCom", "BBA", "BCA", "BTech/BE", "BArch", "BDes", "BFA",
    "BEd", "BPEd", "BPharm", "BPT", "BMLT", "BOptom", "BSc Nursing",
    "MBBS", "BDS", "BHMS", "BAMS", "BUMS", "BVSc", "LLB", "BA LLB",
    "BCom LLB", "BHM", "BTTM", "BSW", "BLib", "BPlan", "Other Bachelor",
]

SCHOOL_CLASSES = [
    "Nursery/LKG/UKG",
    "Class 1",
    "Class 2",
    "Class 3",
    "Class 4",
    "Class 5",
    "Class 6",
    "Class 7",
    "Class 8",
    "Class 9",
    "Class 10",
    "Class 11",
    "Class 12",
]

DIPLOMA_DEGREES = [
    "Diploma in Engineering",
    "Diploma in Pharmacy",
    "Diploma in Education",
    "Diploma in Nursing",
    "Diploma in Hotel Management",
    "Diploma in Design",
    "Polytechnic Diploma",
    "ITI/Vocational Certificate",
    "Other Diploma/Certificate",
]

MASTER_DEGREES = [
    "MA", "MSc", "MCom", "MBA/PGDM", "MCA", "MTech/ME", "MArch", "MDes",
    "MFA", "MEd", "MPEd", "MPharm", "MPT", "MSc Nursing", "MD", "MS",
    "MDS", "LLM", "MHM", "MTTM", "MSW", "MLib", "MPlan", "MPH",
    "MPhil", "Other Master",
]

DOCTORATE_DEGREES = [
    "PhD",
    "DPhil",
    "DM",
    "MCh",
    "Doctorate in Education",
    "Doctorate in Management",
    "Other Doctorate",
]


def degree_options_for(level: str) -> list[str]:
    if level == "Diploma/Certificate":
        return DIPLOMA_DEGREES
    if level == "Bachelor":
        return BACHELOR_DEGREES
    if level == "Master":
        return MASTER_DEGREES
    return DOCTORATE_DEGREES


def score_to_20(value: float, scoring_system: str) -> int:
    if scoring_system == "CGPA out of 10":
        scaled = value * 2
    elif scoring_system == "CGPA out of 4":
        scaled = value * 5
    else:
        scaled = value / 5
    return int(round(max(0, min(20, scaled))))


def subject_proxy(area: str) -> str:
    quantitative_keywords = ["Math", "Science", "Engineering", "Technology", "Medical", "Commerce"]
    return "Mathematics" if any(keyword in area for keyword in quantitative_keywords) else "Portuguese"


def yes_no(label: str, default_yes: bool = False) -> str:
    options = ["yes", "no"] if default_yes else ["no", "yes"]
    return st.segmented_control(label, options, default=options[0])


def render_metric_card(label: str, value: str, detail: str, accent: str = "#0f766e") -> None:
    st.markdown(
        f"""
        <div class="metric-card" style="border-top-color:{accent};">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-detail">{detail}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


st.set_page_config(page_title="Student Performance AI", page_icon="student", layout="wide")

st.markdown(
    """
    <style>
    :root {
        --ink: #172033;
        --muted: #647084;
        --line: #d6deea;
        --panel: #ffffff;
        --soft: #f6f8fb;
        --navy: #172554;
        --blue: #2563eb;
        --teal: #0f766e;
        --green: #15803d;
        --rose: #be123c;
        --amber: #c26a14;
        --saffron: #f59e0b;
    }

    .stApp {
        background:
            linear-gradient(rgba(23, 37, 84, 0.035) 1px, transparent 1px),
            linear-gradient(90deg, rgba(23, 37, 84, 0.035) 1px, transparent 1px),
            radial-gradient(circle at 10% 0%, rgba(37, 99, 235, 0.14), transparent 24rem),
            radial-gradient(circle at 92% 8%, rgba(245, 158, 11, 0.14), transparent 21rem),
            linear-gradient(180deg, #f8fbff 0%, #eef3f9 100%);
        background-size: 24px 24px, 24px 24px, auto, auto, auto;
    }

    .block-container {
        max-width: 1220px;
        padding-top: 1rem;
        padding-bottom: 2.5rem;
    }

    .hero {
        position: relative;
        overflow: hidden;
        background:
            linear-gradient(135deg, rgba(23, 37, 84, 0.98) 0%, rgba(15, 76, 117, 0.96) 54%, rgba(15, 118, 110, 0.92) 100%);
        color: white;
        border-radius: 8px;
        padding: clamp(1.15rem, 3.2vw, 2.35rem);
        margin-bottom: 1rem;
        border: 1px solid rgba(255, 255, 255, 0.18);
        box-shadow: 0 22px 60px rgba(23, 32, 51, 0.18);
    }

    .hero::after {
        content: "";
        position: absolute;
        inset: auto -8% -35% auto;
        width: min(38vw, 440px);
        height: min(38vw, 440px);
        background:
            linear-gradient(135deg, rgba(245, 158, 11, 0.9), rgba(255, 255, 255, 0));
        border-radius: 50%;
        opacity: .45;
        pointer-events: none;
    }

    .eyebrow {
        display: inline-flex;
        align-items: center;
        gap: .45rem;
        color: #d7f3ee;
        background: rgba(255, 255, 255, 0.12);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 999px;
        padding: .32rem .7rem;
        font-size: .78rem;
        font-weight: 800;
        text-transform: uppercase;
        margin-bottom: .75rem;
    }

    .hero h1 {
        position: relative;
        z-index: 1;
        font-size: clamp(1.75rem, 4vw, 3.35rem);
        line-height: 1.06;
        margin: 0 0 .5rem 0;
        letter-spacing: 0;
        max-width: 860px;
    }

    .hero p {
        position: relative;
        z-index: 1;
        color: rgba(255, 255, 255, 0.86);
        font-size: clamp(.96rem, 1.6vw, 1.1rem);
        max-width: 760px;
        margin: 0;
    }

    .hero-pills {
        position: relative;
        z-index: 1;
        display: flex;
        flex-wrap: wrap;
        gap: .5rem;
        margin-top: 1.15rem;
    }

    .hero-pill {
        background: rgba(255, 255, 255, 0.13);
        border: 1px solid rgba(255, 255, 255, 0.2);
        color: rgba(255, 255, 255, 0.92);
        border-radius: 999px;
        padding: .42rem .7rem;
        font-size: .84rem;
        font-weight: 700;
    }

    .info-panel {
        background: rgba(255, 255, 255, 0.86);
        border: 1px solid var(--line);
        border-left: 4px solid var(--blue);
        border-radius: 8px;
        color: var(--muted);
        font-size: .94rem;
        line-height: 1.55;
        margin: .2rem 0 1rem 0;
        padding: .85rem 1rem;
        box-shadow: 0 10px 26px rgba(23, 32, 51, 0.07);
    }

    .info-panel b {
        color: var(--ink);
    }

    .status-strip {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: .75rem;
        margin: .85rem 0 1.1rem 0;
    }

    .status-item, .metric-card, .scope-box {
        background: rgba(255, 255, 255, 0.92);
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: .9rem 1rem;
        box-shadow: 0 14px 34px rgba(23, 32, 51, 0.08);
    }

    .status-item {
        position: relative;
        overflow: hidden;
    }

    .status-item::before {
        content: "";
        position: absolute;
        inset: 0 auto 0 0;
        width: 4px;
        background: var(--blue);
    }

    .status-item:nth-child(2)::before {
        background: var(--teal);
    }

    .status-item:nth-child(3)::before {
        background: var(--saffron);
    }

    .status-label, .metric-label {
        color: var(--muted);
        font-size: .76rem;
        font-weight: 800;
        text-transform: uppercase;
    }

    .status-value {
        color: var(--ink);
        font-size: 1.06rem;
        font-weight: 800;
        margin-top: .16rem;
    }

    .metric-card {
        min-height: 128px;
        border-top: 4px solid var(--teal);
        transition: transform .16s ease, box-shadow .16s ease, border-color .16s ease;
    }

    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 18px 40px rgba(23, 32, 51, 0.12);
    }

    .metric-value {
        color: var(--ink);
        font-size: clamp(1.45rem, 3vw, 2.2rem);
        font-weight: 800;
        margin-top: .25rem;
        line-height: 1.05;
    }

    .metric-detail {
        color: var(--muted);
        font-size: .9rem;
        margin-top: .45rem;
    }

    .section-title {
        color: var(--ink);
        font-size: 1.02rem;
        font-weight: 800;
        margin: .4rem 0 .45rem 0;
        padding-left: .72rem;
        border-left: 4px solid var(--teal);
    }

    .scope-box {
        color: var(--muted);
        font-size: .92rem;
        line-height: 1.55;
        margin-top: 1rem;
    }

    div[data-testid="stTabs"] {
        margin-top: .25rem;
    }

    div[data-testid="stTabs"] button {
        border-radius: 8px 8px 0 0;
        min-height: 44px;
        font-weight: 750;
    }

    div[data-testid="stTabs"] button[aria-selected="true"] {
        color: var(--teal);
        border-bottom-color: var(--teal);
    }

    div[data-testid="stRadio"] label,
    div[data-testid="stSelectbox"] label,
    div[data-testid="stNumberInput"] label,
    div[data-testid="stSlider"] label {
        color: var(--ink);
        font-weight: 720;
    }

    div[data-testid="stRadio"] label p {
        color: #111827;
        font-weight: 850;
    }

    div[data-testid="stRadio"] div[role="radiogroup"] label {
        background: rgba(255, 255, 255, 0.82);
        border: 1px solid #c7d2e3;
        border-radius: 8px;
        padding: .42rem .7rem;
    }

    div[data-testid="stRadio"] div[role="radiogroup"] label p {
        color: #0f172a;
        font-weight: 850;
    }

    div[data-baseweb="select"] > div,
    div[data-testid="stNumberInput"] input {
        border-color: #c9d4e3;
        background: rgba(255, 255, 255, 0.96);
    }

    div[data-testid="stButton"] > button {
        width: 100%;
        min-height: 3rem;
        border-radius: 8px;
        font-weight: 800;
        background: linear-gradient(135deg, #0f766e 0%, #2563eb 100%);
        border: 1px solid rgba(15, 118, 110, 0.2);
        box-shadow: 0 12px 26px rgba(15, 118, 110, 0.24);
    }

    div[data-testid="stButton"] > button:hover {
        background: linear-gradient(135deg, #115e59 0%, #1d4ed8 100%);
        border-color: rgba(37, 99, 235, 0.35);
        box-shadow: 0 16px 32px rgba(37, 99, 235, 0.24);
    }

    div[data-testid="stProgress"] > div > div > div {
        background: linear-gradient(90deg, #0f766e, #2563eb);
    }

    div[data-testid="stAlert"] {
        border-radius: 8px;
        border: 1px solid var(--line);
    }

    @media (max-width: 760px) {
        .block-container {
            padding-left: .8rem;
            padding-right: .8rem;
        }

        .status-strip {
            grid-template-columns: 1fr;
        }

        .metric-card {
            min-height: auto;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_artifacts():
    reg_path = REG_MODEL_PATH if REG_MODEL_PATH.exists() else LEGACY_REG_MODEL_PATH
    clf_path = CLF_MODEL_PATH if CLF_MODEL_PATH.exists() else LEGACY_CLF_MODEL_PATH
    metadata_path = METADATA_PATH if METADATA_PATH.exists() else LEGACY_METADATA_PATH

    with reg_path.open("rb") as f:
        reg_mod = pickle.load(f)
    with clf_path.open("rb") as f:
        clf_mod = pickle.load(f)

    metadata = {}
    if metadata_path.exists():
        with metadata_path.open("rb") as f:
            metadata = pickle.load(f)

    return reg_mod, clf_mod, metadata


try:
    reg_model, clf_model, metadata = load_artifacts()
except FileNotFoundError:
    st.error("Pre-trained models were not found. Run `python Eda\\train.py` first.")
    st.stop()


feature_columns = metadata.get("feature_columns")
metrics = metadata.get("holdout_metrics", {})
indian_rows = metadata.get("indian_training_rows", 0)

st.markdown(
    """
    <section class="hero">
        <div class="eyebrow">AI Academic Risk Dashboard</div>
        <h1>Indian Student Performance Prediction</h1>
        <p>Responsive academic risk analytics for school boards, state and union territory boards, and college degree pathways.</p>
        <div class="hero-pills">
            <span class="hero-pill">School Boards</span>
            <span class="hero-pill">College Degrees</span>
            <span class="hero-pill">Score + Risk</span>
        </div>
    </section>
    """,
    unsafe_allow_html=True,
)

if metrics:
    st.markdown(
        f"""
        <div class="status-strip">
            <div class="status-item">
                <div class="status-label">Regression MAE</div>
                <div class="status-value">{metrics.get('regression_mae', 0):.2f} / 20</div>
            </div>
            <div class="status-item">
                <div class="status-label">Classification Accuracy</div>
                <div class="status-value">{metrics.get('classification_accuracy', 0) * 100:.1f}%</div>
            </div>
            <div class="status-item">
                <div class="status-label">Indian Training Rows</div>
                <div class="status-value">{indian_rows}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

mode_col, context_col = st.columns([0.95, 2.05], vertical_alignment="bottom")
with mode_col:
    education_level = st.radio(
        "Education Level",
        ["School Board", "College/University"],
        horizontal=True,
        label_visibility="visible",
    )

with context_col:
    if education_level == "School Board":
        board_or_system = st.selectbox("Board", SCHOOL_BOARDS)
    else:
        board_or_system = st.selectbox("College/University System", COLLEGE_SYSTEMS)

if education_level == "School Board":
    st.markdown(
        """
        <div class="info-panel">
            <b>Education Level: School Board</b><br>
            Use this mode for pre-primary, primary, middle, secondary, and senior secondary students
            across CBSE, CISCE/ICSE, NIOS, state boards, and union territory board contexts.
            Enter the class, stream, current marks, attendance, study time, and support details.
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        """
        <div class="info-panel">
            <b>Education Level: College/University</b><br>
            Use this mode for diploma, certificate, bachelor, master, and doctorate students. Select the university system,
            degree, program area, semester scores, attendance, and academic support details.
        </div>
        """,
        unsafe_allow_html=True,
    )

if education_level == "School Board":
    class_or_program = "Class 10"
    degree_level = "Not applicable"
    degree_name = "Not applicable"
    default_age = 16
    max_absences = 250
else:
    degree_level = "Bachelor"
    degree_name = "BTech/BE"
    class_or_program = degree_name
    default_age = 20
    max_absences = 300

tab_academic, tab_profile, tab_support = st.tabs(["Academic", "Profile", "Support & Lifestyle"])

with tab_academic:
    st.markdown('<div class="section-title">Academic Context</div>', unsafe_allow_html=True)
    st.caption("Choose the course/class information and enter the two most recent academic scores.")
    a1, a2, a3 = st.columns(3)
    with a1:
        if education_level == "School Board":
            class_or_program = st.selectbox("Class / School Level", SCHOOL_CLASSES, index=10)
            degree_level = "Not applicable"
            degree_name = "Not applicable"
        else:
            degree_level = st.selectbox("College Level", ["Diploma/Certificate", "Bachelor", "Master", "Doctorate"], index=1)
            degree_options = degree_options_for(degree_level)
            degree_name = st.selectbox("Degree/Program", degree_options)
            class_or_program = degree_name
    with a2:
        if education_level == "School Board":
            subject_area = st.selectbox(
                "Subject Stream",
                ["Mathematics", "Science", "Commerce", "Arts/Humanities", "Vocational", "General"],
            )
        else:
            subject_area = st.selectbox(
                "Program Area",
                [
                    "Engineering/Technology",
                    "Science",
                    "Commerce/Management",
                    "Arts/Humanities",
                    "Medical/Health",
                    "Law",
                    "Education",
                    "Design/Fine Arts",
                    "Hospitality/Tourism",
                    "Social Work",
                    "Other",
                ],
            )
    with a3:
        scoring_options = ["Percentage out of 100", "CGPA out of 10"]
        if education_level == "College/University":
            scoring_options.append("CGPA out of 4")
        scoring_system = st.selectbox("Marks Format", scoring_options)

    score_max = 10.0 if scoring_system == "CGPA out of 10" else 4.0 if scoring_system == "CGPA out of 4" else 100.0
    score_step = 0.1 if "CGPA" in scoring_system else 1.0
    score_default_1 = 6.0 if scoring_system == "CGPA out of 10" else 2.5 if scoring_system == "CGPA out of 4" else 60.0
    score_default_2 = 6.5 if scoring_system == "CGPA out of 10" else 2.7 if scoring_system == "CGPA out of 4" else 65.0

    st.markdown('<div class="section-title">Performance Signals</div>', unsafe_allow_html=True)
    s1, s2, s3, s4 = st.columns(4)
    with s1:
        first_score = st.number_input("Previous Score", 0.0, score_max, score_default_1, score_step)
    with s2:
        second_score = st.number_input("Latest Score", 0.0, score_max, score_default_2, score_step)
    with s3:
        studytime = st.slider("Weekly Study Time", 1, 4, 2, help="1: <2h, 2: 2-5h, 3: 5-10h, 4: >10h")
    with s4:
        failures = st.selectbox("Failed Subjects/Semesters", [0, 1, 2, 3])

with tab_profile:
    st.markdown('<div class="section-title">Student & Institution Profile</div>', unsafe_allow_html=True)
    st.caption("Add basic student, institution, and family background information used by the prediction model.")
    p1, p2, p3 = st.columns(3)
    with p1:
        sex = st.selectbox("Gender", ["Female", "Male"])
        age = st.slider("Age", 10, 35, default_age)
        address = st.selectbox("Home Location", ["Urban (City)", "Rural (Town/Village)"])
    with p2:
        institution_type = st.selectbox("Institution Type", ["Urban/large campus", "Small town/rural campus"])
        famsize = st.selectbox("Family Size", ["GT3", "LE3"])
        parent_status = st.selectbox("Parent Cohabitation Status", ["T", "A"])
    with p3:
        medu = st.slider("Mother/Guardian Education", 0, 4, 3)
        fedu = st.slider("Father/Guardian Education", 0, 4, 3)
        guardian = st.selectbox("Primary Guardian", ["mother", "father", "other"])

    st.markdown('<div class="section-title">Family Background</div>', unsafe_allow_html=True)
    f1, f2, f3 = st.columns(3)
    with f1:
        mjob = st.selectbox("Mother/Guardian Job", ["other", "services", "teacher", "at_home", "health"])
    with f2:
        fjob = st.selectbox("Father/Guardian Job", ["other", "services", "teacher", "at_home", "health"])
    with f3:
        reason = st.selectbox("Institution Choice Reason", ["course", "home", "reputation", "other"])

with tab_support:
    st.markdown('<div class="section-title">Attendance, Support & Habits</div>', unsafe_allow_html=True)
    st.caption("Add attendance, learning support, health, and lifestyle indicators that may affect performance.")
    h1, h2, h3 = st.columns(3)
    with h1:
        traveltime = st.slider("Travel Time", 1, 4, 1)
        absences = st.number_input("Absences", min_value=0, max_value=max_absences, value=4)
        health = st.slider("Current Health Status", 1, 5, 5)
    with h2:
        higher = yes_no("Plans Higher Education?", True)
        internet = yes_no("Home Internet Access?", True)
        schoolsup = yes_no("Extra Educational Support")
        famsup = yes_no("Family Educational Support", True)
    with h3:
        paid = yes_no("Extra Paid Classes")
        activities = yes_no("Extra-curricular Activities", True)
        nursery = yes_no("Nursery/Foundation", True)
        romantic = yes_no("Relationship")

    l1, l2, l3, l4, l5 = st.columns(5)
    with l1:
        famrel = st.slider("Family Relationship", 1, 5, 4)
    with l2:
        freetime = st.slider("Free Time", 1, 5, 3)
    with l3:
        goout = st.slider("Social Frequency", 1, 5, 3)
    with l4:
        dalc = st.slider("Workday Alcohol", 1, 5, 1)
    with l5:
        walc = st.slider("Weekend Alcohol", 1, 5, 1)

g1 = score_to_20(first_score, scoring_system)
g2 = score_to_20(second_score, scoring_system)
subject = subject_proxy(subject_area)
school = "GP" if institution_type == "Urban/large campus" else "MS"
model_age = max(15, min(22, age))
model_absences = int(round(min(93, absences * 93 / max_absences)))

input_data = {
    "school": school,
    "sex": "F" if sex == "Female" else "M",
    "age": model_age,
    "address": "U" if "Urban" in address else "R",
    "famsize": famsize,
    "Pstatus": parent_status,
    "Medu": medu,
    "Fedu": fedu,
    "Mjob": mjob,
    "Fjob": fjob,
    "reason": reason,
    "guardian": guardian,
    "traveltime": traveltime,
    "studytime": studytime,
    "failures": failures,
    "schoolsup": schoolsup,
    "famsup": famsup,
    "paid": paid,
    "activities": activities,
    "nursery": nursery,
    "higher": higher,
    "internet": internet,
    "romantic": romantic,
    "famrel": famrel,
    "freetime": freetime,
    "goout": goout,
    "Dalc": dalc,
    "Walc": walc,
    "health": health,
    "absences": model_absences,
    "G1": g1,
    "G2": g2,
    "subject": subject,
    "education_level": education_level,
    "board_or_system": board_or_system,
    "state_ut": board_or_system,
    "class_or_program": class_or_program,
    "degree_level": degree_level if education_level == "College/University" else "Not applicable",
    "degree_name": degree_name if education_level == "College/University" else "Not applicable",
    "program_area": subject_area,
    "scoring_system": scoring_system,
    "institution_type": institution_type,
}

input_df = pd.DataFrame([input_data])
if feature_columns:
    input_df = input_df.reindex(columns=feature_columns)

predict_col, scope_col = st.columns([0.75, 2.25], vertical_alignment="center")
with predict_col:
    predict_clicked = st.button("Analyze Performance", type="primary")
with scope_col:
    st.caption(f"Selected context: {education_level} | {board_or_system} | {class_or_program}")

if predict_clicked:
    predicted_score = float(reg_model.predict(input_df)[0])
    predicted_score = max(0.0, min(20.0, predicted_score))

    predicted_pass = int(clf_model.predict(input_df)[0])
    pass_probability = None
    if hasattr(clf_model, "predict_proba"):
        pass_probability = float(clf_model.predict_proba(input_df)[0][1])

    risk_label = "Passing Track" if predicted_pass == 1 else "At Risk"
    risk_color = "#0f766e" if predicted_pass == 1 else "#be123c"
    probability_text = "N/A" if pass_probability is None else f"{pass_probability * 100:.1f}%"
    score_percent = min(100, max(0, int(round(predicted_score / 20 * 100))))
    probability_percent = 0 if pass_probability is None else min(100, max(0, int(round(pass_probability * 100))))

    st.markdown('<div class="section-title">Prediction Results</div>', unsafe_allow_html=True)
    r1, r2, r3 = st.columns(3)
    with r1:
        render_metric_card("Predicted Final Score", f"{predicted_score:.2f} / 20", "Converted model scale", "#0f766e")
        st.progress(score_percent)
    with r2:
        render_metric_card("Academic Risk", risk_label, "Classification output", risk_color)
        st.progress(100 if predicted_pass == 1 else 35)
    with r3:
        render_metric_card("Probability of Passing", probability_text, "Model confidence", "#d97706")
        st.progress(probability_percent)

    if predicted_pass == 0 or predicted_score < 11:
        st.warning(
            "This student is showing elevated academic risk. Review absences, recent grades, "
            "study time, and support options."
        )
    else:
        st.success("This student is currently on a stable passing track.")

    st.markdown(
        f"""
        <div class="scope-box">
            <b>Model scope:</b> input marks were converted to {g1}/20 and {g2}/20.
            Model-range values used: age {model_age}, absences {model_absences}.
            Board and degree names are supported in the interface and training schema, but high
            accuracy for those groups requires real anonymized Indian records in the project dataset.
        </div>
        """,
        unsafe_allow_html=True,
    )
