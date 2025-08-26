# safeshare_app.py
import streamlit as st
import re
from PIL import Image
import numpy as np
import cv2
import pytesseract
import io
import tempfile

st.set_page_config(page_title="SafeShare Vision", page_icon="🔒", layout="wide")
st.title("🔒 SafeShare Vision — Text + Image Privacy Guardian")
st.write("Check text for personal data and blur faces in images before sharing.")

# -----------------------------
# Privacy patterns (expandable)
# -----------------------------
risk_patterns = {
    "📞 Phone Number": r"\+?\d{10,13}",
    "📧 Email": r"[a-zA-Z0-9+_.-]+@[a-zA-Z0-9.-]+",
    "🆔 Aadhaar/ID Number": r"\b\d{4}\s\d{4}\s\d{4}\b",
    "💳 Credit Card": r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b",
    # loose address-ish pattern
    "📍 Address-like phrase": r"\d{1,3}\s+\w+\s+(street|st|road|rd|avenue|ave|block|lane|lane\.)"
}
# add location keywords to flag casual mentions
risky_words = ["school", "college", "address", "home", "street", "city", "house", "hostel", "parents"]

# -----------------------------
# Utility: check risks in text
# -----------------------------
def check_risks(text):
    findings = []
    for label, pattern in risk_patterns.items():
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            findings.append((label, matches))
    # word-based checks
    for w in risky_words:
        if re.search(r"\b" + re.escape(w) + r"\b", text, re.IGNORECASE):
            findings.append(("📍 Location word", [w]))
    return findings

# -----------------------------
# Utility: blur faces in CV image
# -----------------------------
def blur_faces_pil(pil_image, blur_strength=35, min_size=30):
    # Convert PIL -> OpenCV
    img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(min_size, min_size))

    if len(faces) == 0:
        return pil_image, 0  # unchanged, zero faces

    for (x, y, w, h) in faces:
        # enlarge box slightly for better blur coverage
        pad_w, pad_h = int(0.15*w), int(0.15*h)
        x1, y1 = max(0, x-pad_w), max(0, y-pad_h)
        x2, y2 = min(img.shape[1], x+w+pad_w), min(img.shape[0], y+h+pad_h)
        face_region = img[y1:y2, x1:x2]
        # apply heavy gaussian blur
        ksize = (blur_strength if blur_strength % 2 == 1 else blur_strength+1, )*2
        try:
            blurred = cv2.GaussianBlur(face_region, ksize, 0)
        except Exception:
            # fallback to a smaller kernel if too large
            blurred = cv2.GaussianBlur(face_region, (51,51), 0)
        img[y1:y2, x1:x2] = blurred

    # Convert back to PIL
    result = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return result, len(faces)

# -----------------------------
# UI: Text checker
# -----------------------------
st.header("✍️ Text Privacy Checker")
user_text = st.text_area("Enter text (post, bio, tweet, message):", height=180)

if st.button("🔍 Check Text"):
    if not user_text.strip():
        st.warning("⚠️ Please enter some text first.")
    else:
        findings = check_risks(user_text)
        if findings:
            st.error("⚠️ Privacy risks found in text:")
            for label, matches in findings:
                st.write(f"- **{label}**: {', '.join(matches)}")
        else:
            st.success("✅ No obvious privacy risks found in text.")

st.markdown("---")

# -----------------------------
# UI: Image checker & blurring
# -----------------------------
st.header("🖼️ Image Privacy Checker & Face Blurring")
st.write("Upload a photo. The app will extract any visible text (OCR) and optionally blur faces.")

uploaded_file = st.file_uploader("Upload image (jpg/png)", type=["jpg", "jpeg", "png"])
blur_toggle = st.checkbox("Automatically blur detected faces", value=True)
blur_strength = st.slider("Blur strength (odd kernel size)", min_value=15, max_value=101, step=2, value=35)

if uploaded_file:
    try:
        # read image as PIL
        image = Image.open(uploaded_file).convert("RGB")
    except Exception as e:
        st.error("Could not open image. Make sure it's a valid image file.")
        st.stop()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Image")
        st.image(image, use_column_width=True)

    # OCR: extract text from image
    with st.spinner("🔎 Running OCR on image..."):
        # Convert to OpenCV gray for better OCR sometimes
        img_np = np.array(image)
        gray_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        # optional thresholding could be added for clarity
        ocr_text = pytesseract.image_to_string(gray_cv)

    # show extracted text and analyze it
    st.subheader("🔍 Extracted text from image (OCR)")
    if ocr_text.strip():
        st.code(ocr_text)
        img_findings = check_risks(ocr_text)
        if img_findings:
            st.error("⚠️ Privacy risks found in image text:")
            for label, matches in img_findings:
                st.write(f"- **{label}**: {', '.join(matches)}")
        else:
            st.success("✅ No obvious privacy risks found in image text.")
    else:
        st.info("No readable text found in the image.")

    # Face blur (if toggled)
    blurred_image = None
    faces_count = 0
    if blur_toggle:
        with st.spinner("🧼 Detecting faces and applying blur..."):
            blurred_image, faces_count = blur_faces_pil(image, blur_strength)
        with col2:
            st.subheader("Processed Image")
            if faces_count > 0:
                st.image(blurred_image, use_column_width=True)
                st.success(f"🔒 Blurred {faces_count} face(s).")
            else:
                st.image(blurred_image, use_column_width=True)
                st.info("No faces detected to blur.")

        # download button for blurred image
        buf = io.BytesIO()
        blurred_image.save(buf, format="PNG")
        buf.seek(0)
        st.download_button("⬇️ Download safe image (PNG)", data=buf, file_name="safeshare_blurred.png", mime="image/png")
    else:
        with col2:
            st.subheader("Processed Image")
            st.image(image, use_column_width=True)
            st.info("Face blurring is turned off. Toggle 'Automatically blur detected faces' to enable.")
