import streamlit as st
import pandas as pd
import io
import cv2
import numpy as np
import pytesseract
import re
from pdf2image import convert_from_bytes
path = "D:/my_project/dataset/554873.pdf"
# Backend function for processing the PDF file
def backend(pdf_file):
    # Extract file name and create ID
    file_name = pdf_file.name
    file_id = file_name.rsplit('.', 1)[0]  # Remove file extension

    # Convert PDF to images
    pages = convert_from_bytes(pdf_file.read(), 350)

    # Work with the image directly (the first and only page)
    image = pages[0]

    # Preprocess the image for OCR
    def preprocess_image_for_ocr(image):
        gray = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary_image = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        coords = np.column_stack(np.where(binary_image > 0))
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        (h, w) = binary_image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(binary_image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        enhanced_image = cv2.convertScaleAbs(rotated, alpha=1.5, beta=0)
        return enhanced_image

    preprocessed_img = preprocess_image_for_ocr(image)

    # Extract text from the preprocessed image
    def extract_text_from_image(image):
        text = pytesseract.image_to_string(image)
        return text

    extracted_text = extract_text_from_image(preprocessed_img)

    def clean_text(text):
        cleaned_text = re.sub(r'\n\s*\n', '\n', text)
        return cleaned_text

    cleaned_text = clean_text(extracted_text)

    def process_text(text):
        def extract_section(text, start_marker, end_marker):
            start = text.find(start_marker) + len(start_marker)
            end = text.find(end_marker)
            section = text[start:end].strip()
            section = re.sub(r'\s+', ' ', section)
            return section

        def replace_section(original_text, start_marker, end_marker, new_text):
            start = original_text.find(start_marker) + len(start_marker)
            end = original_text.find(end_marker)
            new_text = original_text[:start] + ' ' + new_text + '\n' + original_text[end:]
            return new_text

        illness_section = extract_section(text, 'HISTORY OF PRESENTING ILLNESS', 'no h/o')
        updated_text = replace_section(text, 'HISTORY OF PRESENTING ILLNESS', 'no h/o', illness_section)
        past_history_section = extract_section(updated_text, 'PAST HISTORY', 'FAMILY HISTORY')
        final_text = replace_section(updated_text, 'PAST HISTORY', 'FAMILY HISTORY', past_history_section)
        general_examination_text = extract_section(final_text, 'GENERAL EXAMINATION', 'vitals')
        final_text = replace_section(final_text, 'GENERAL EXAMINATION', 'vitals', general_examination_text)

        return final_text

    final_text = process_text(cleaned_text)

    patterns = {
        'ID': file_id,  # Use the extracted file ID
        'DIAGNOSIS': r'DIAGNOSIS\s*(.+?)\s+CHIEF COMPLAINTS',
        'Fever': r'FEVER- (.+?)\s+HEADACHE',
        'Headache': r'HEADACHE\s*(.+?)\s+BODYPAIN',
        'Body Pain': r'BODYPAIN\s*(.+?)\n',
        'Symptoms': r'Patient came with\s*(.+?)\s*no',
        'No Symptom': r'no h/o \s*(.+?)\s*PAST HISTORY',
        'Past History': r'PAST HISTORY\s*(.+?)\s*FAMILY HISTORY',
        'Family History': r'FAMILY HISTORY\s*(.+?)\s*PERSONAL HISTORY',
        'Diet': r'Diet:\s*(.+?)\n',
        'Appetite': r'Appetite:\s*(.+?)\n',
        'Sleep': r'Sleep:\s*(.+?)\n',
        'Bowel and Bladder': r'Bowel and Bladder:\s*(.+?)\n',
        'General Examination': r'Patient is\s*(.+?)\s*vitals',
        'PULSE': r'PULSE\s*(.+?)\n',
        'BP': r'BP\s*(.+?)\n',
        'RR': r'RR\s*(.+?)\n',
        'Temperature': r'TEMPERATURE -\s*(.+?)\n',
        'TREATMENT': r'TREATMENT\s*(.+)',
    }

    def extract_info(text, patterns):
        info = {}
        for key, pattern in patterns.items():
            if key == 'ID':
                info[key] = pattern
            else:
                match = re.search(pattern, text, re.DOTALL)
                if match:
                    info[key] = match.group(1).strip()
                else:
                    info[key] = ""
        return info

    summary = extract_info(final_text, patterns)

    def format_DIAGNOSIS(DIAGNOSIS_text):
        st = ""
        DIAGNOSIS_list = re.split(r'\n+', DIAGNOSIS_text)
        for i in range(len(DIAGNOSIS_list)):
            st += str(i + 1) + ". " + DIAGNOSIS_list[i] + "\n"
        return st

    def format_symptom(symptoms_text):
        st = ""
        symptoms_list = re.split(r'\s*,\s*|\s+and\s+|\n+', symptoms_text)
        for i in range(len(symptoms_list)):
            st += str(i + 1) + ". " + symptoms_list[i] + "\n"
        return st

    def format_no_symptom(no_symptoms_text):
        st = ""
        no_symptoms_list = re.split(r'\s*/\s*', no_symptoms_text)
        for i in range(len(no_symptoms_list)):
            st += str(i + 1) + ". " + no_symptoms_list[i] + "\n"
        return st

    def general_examination(General_Examination_text):
        st = ""
        General_Examination_list = re.split(r'\.\s+', General_Examination_text)
        for i, item in enumerate(General_Examination_list):
            item = item.strip()
            if item:
                st += str(i + 1) + ". " + item + "\n"
        return st

    def format_TREATMENT(TREATMENT_text):
        st = ""
        TREATMENT_list = re.split(r'\n+', TREATMENT_text)
        for i in range(len(TREATMENT_list)):
            st += str(i + 1) + ". " + TREATMENT_list[i] + "\n"
        return st

    summary['DIAGNOSIS'] = format_DIAGNOSIS(summary['DIAGNOSIS']) if summary['DIAGNOSIS'] else summary['DIAGNOSIS']
    summary['Symptoms'] = format_symptom(summary['Symptoms']) if summary['Symptoms'] else summary['Symptoms']
    summary['No Symptom'] = format_no_symptom(summary['No Symptom']) if summary['No Symptom'] else summary['No Symptom']
    summary['General Examination'] = general_examination(summary['General Examination']) if summary['General Examination'] else summary['General Examination']
    summary['TREATMENT'] = format_TREATMENT(summary['TREATMENT']) if summary['TREATMENT'] else summary['TREATMENT']

    return summary



def main():
    st.set_page_config(page_title="PDF Uploader", layout="wide")

    # Custom CSS for background color and styling
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #E8E1FA;
        }
        .main-header {
            font-size: 3em;
            color: #9D1BC2;
            font-family: 'Courier New', monospace;
        }
        .sub-header {
            font-size: 1.5em;
            color: #2196F3;
        }
        .instructions {
            background-color: #f0f8ff;
            padding: 10px;
            border-radius: 5px;
        }
        .divider {
            margin-top: 20px;
            margin-bottom: 20px;
            border-top: 2px solid #FFC107;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Header
    st.markdown('<h1 class="main-header">📄 PDF Uploader and Summarizer</h1>', unsafe_allow_html=True)
    st.write("Upload multiple PDF files, and get a summarized view in a table format.")

    # Instructions
    st.markdown('<div class="instructions">', unsafe_allow_html=True)
    st.markdown("""
        **Instructions:**
        1. Click on the **Browse files** button below to upload your PDFs.
        2. After uploading, summaries of the PDFs will be displayed in a table.
        3. You can download the table as a CSV file.
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)  # Add a colored horizontal divider
    
    # Layout for file uploader and display
    col1, col2 = st.columns([1, 3])  # Define two columns, one narrow, one wide

    with col1:
        st.markdown('<div class="sub-header">Upload PDFs Here:</div>', unsafe_allow_html=True)
        uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True, label_visibility='collapsed')

    with col2:
        if uploaded_files:
            all_summaries = []
            seen_files = set()  # To keep track of filenames

            for uploaded_file in uploaded_files:
                if uploaded_file.name in seen_files:
                    st.warning(f"File '{uploaded_file.name}' is a duplicate and will be skipped.")
                else:
                    seen_files.add(uploaded_file.name)
                    # Process each PDF file
                    pdf_summary = backend(uploaded_file)
                    all_summaries.append(pdf_summary)

            if all_summaries:
                # Convert list of dictionaries to DataFrame
                df = pd.DataFrame(all_summaries)
                
                # Display the DataFrame
                st.write("### Summarized Table")
                st.dataframe(df, use_container_width=True)
                
                # Download button for CSV with custom color
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download as CSV",
                    data=csv,
                    file_name="pdf_summaries.csv",
                    mime="text/csv",
                    key="download-button",
                    help="Click to download the summary table as a CSV file",
                    use_container_width=True,
                )
            else:
                st.error("No text found in the PDFs.")
        else:
            st.info("Upload PDFs to see the summary.")

if __name__ == "__main__":
    main()
