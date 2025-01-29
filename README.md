# Health-Report-Summerizer-Extract-Key-Information
This project aims to summarize patient medical reports by extracting key information from PDF documents using Optical Character Recognition (OCR) and text processing techniques. The system enhances the extracted data using custom preprocessing pipelines to generate a structured and readable summary.

Features
OCR-based Text Extraction: Utilizes Pytesseract to extract text from scanned PDF documents.
Preprocessing Pipeline: Custom preprocessing steps to clean and structure the extracted data.
Structured Summary Generation: Converts the extracted and processed data into a concise summary.
User Interface: A simple web interface built using Streamlit to upload PDFs and display results.
Tools and Technologies
Python: Programming language used for implementation.
Pytesseract: Python wrapper for Google's Tesseract-OCR to extract text from scanned images in PDFs.
PDF2Image: Converts PDF pages into images for OCR processing.
OpenCV: Image processing techniques for improving OCR accuracy.
Pandas: Data manipulation and structuring the extracted data.
Streamlit: Used to create the user interface for easy deployment and interaction.
