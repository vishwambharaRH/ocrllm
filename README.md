# OCR+LLM Processor for Documents

## Overview
This application provides a full pipeline for digitizing and refining scanned PDFs (non-text-layered) using Optical Character Recognition (OCR) and optional post-processing via Large Language Models (LLMs).

> [!NOTE]
> Bugs have been noted with Tesseract OCR in Indic languages. Google Cloud Vision is recommended for better accuracy in such cases.
> Any bugs or feature requests can be reported in the Issues section. 
> For the moment, please be advised to not use Tesseract.


## Features

- Extract text from scanned PDFs using:
  - **Tesseract OCR**
  - **Google Cloud Vision API**
- Post-process extracted text using any LLM via:
  - OpenAI API
  - OpenRouter API
  - Other compatible endpoints
- Full UI for selecting PDF, choosing OCR/LLM engines, setting prompts, and saving results
- Option to perform OCR-only (bypassing LLM)
- Cross-platform: macOS `.app.zip` and Windows `.exe` releases available

---

## Working

- Target PDF is acquired and output text file is path chosen.
- Prior dependency files, prompts, and keys are duly input.
- PDF is split into required number of pages, images are saved in a temporary directory.
- OCR is carried out sequentially on all the images.
- The text output is saved in a temporary .txt, which may or may not be the final file based pn the options selected.
- The text is preprocessed and split into multiple batches to send to the LLM, along with the prompt.
- The LLM-processed text is entered into the output text file.

---

## Dependencies

### Tesseract OCR
- Install Tesseract from the [official documentation](https://tesseract-ocr.github.io/tessdoc/Installation.html).
- Ensure Tesseract is available in your system PATH.

### Google Cloud Vision
- Sign up for Google Cloud Suite.
- Create a project, enable the **Vision API**, and download your `.json` service account key.

---

## Installation

### macOS
1. Download `app.zip` from the [Releases](#) section.
2. Unzip and move the app to `/Applications`.

### Windows
1. Download the `.exe` file from the [Releases](#) section.
2. Double-click to install or run as needed.

---

## How to Use

1. **Launch the App.**
2. Click **Browse** to select your target PDF file.
3. Click **Save As** to choose a directory and filename for the output `.txt` file.
4. Choose the OCR engine:
    - **Tesseract**: No extra setup required (beyond installation).
    - **Google Vision**: Upload your `.json` service account key when prompted.
5. (Optional) Tick the **OCR only** checkbox if you do **not** want LLM processing.
6. If using an LLM:
    - Enter a valid **API key** (Note: OpenAI and OpenRouter use different key formats).
    - Provide your custom prompt, or refer to examples in `prompts.txt`.
7. Click **Start Processing** to begin.
8. If needed, click **Stop** to cancel ongoing processing.

---

## Issues

- Some Tesseract-related bugs are known and currently being worked on.
- Future builds will include more robust error handling and UI feedback.

