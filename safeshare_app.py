import streamlit as st
import re

st.set_page_config(page_title="SafeShare - Privacy Checker", page_icon="🔒", layout="wide")

st.title("🔒 SafeShare: Online Privacy Checker")
st.write("Paste your text below and I'll highlight possible **privacy risks** before you post online.")

# --- Input Box ---
user_text = st.text_area("✍️ Enter your text (tweet, post, bio, etc.)", height=200)

# --- Risk Patterns ---
risk_patterns = {
    "📍 Address/Location": r"\d{1,3}\s+\w+\s+(street|st|road|rd|avenue|ave|block)",
    "📞 Phone Number": r"\+?\d{10,13}",
    "📧 Email": r"[a-zA-Z0-9+_.-]+@[a-zA-Z0-9.-]+",
    "🆔 Aadhaar/ID Number": r"\d{4}\s\d{4}\s\d{4}",
    "💳 Credit Card": r"\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}"
}

# --- Check Privacy Risks ---
def check_risks(text):
    findings = []
    for label, pattern in risk_patterns.items():
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            findings.append((label, matches))
    return findings

if st.button("🔍 Check Privacy Risks"):
    if not user_text.strip():
        st.warning("⚠️ Please enter some text first!")
    else:
        risks = check_risks(user_text)
        if risks:
            st.error("⚠️ Privacy risks found!")
            for label, matches in risks:
                st.write(f"**{label}:** {', '.join(matches)}")
        else:
            st.success("✅ No obvious privacy risks found. Safe to share!")

# --- Footer ---
st.markdown("---")
st.caption("Made for Innovators Day @ Manakula Vinayagar College 🎓")
