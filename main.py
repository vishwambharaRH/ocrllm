# This code is cross platform.
# To run the build, do as follows:
# For macOS: "pyinstaller --name="OCR-LLM Processor" --windowed --icon="YourIcon.icns" --add-data="/opt/homebrew/opt/tesseract/share/tessdata:tessdata" main.py"
# For Windows: "pyinstaller --name="OCR-LLM Processor" --windowed --icon="YourIcon.ico" --add-data="C:\Program Files\Tesseract-OCR\tessdata;tessdata" main.py"

# Windows users should input the path for Tesseract as C:/Program Files/Tesseract-OCR/tesseract.exe
# Mac users must first install Tesseract using Homebrew: `brew install tesseract`, then set the path to /opt/homebrew/bin/tesseract

# OCR + LLM Document Processor (fixed)
# Cross-platform (macOS + Windows)
# Adds a proper language-selection dropdown for Tesseract and Google Vision

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading
import os
import json
from PIL import Image
import pytesseract
import fitz  # PyMuPDF
import requests
import tempfile
import shutil
import sys
import sv_ttk
import time

# Note: openai and google-cloud-vision imports are done dynamically where used so
# the app can still run without those optional dependencies present at import time.

class OcrLlmApp:
    def __init__(self, root):
        """Initialize the application's GUI."""
        self.root = root
        self.root.title("OCR + LLM Document Processor")
        self.root.geometry("800x950")
        self.root.minsize(800, 650)
        sv_ttk.set_theme("light")

        # Instance variables
        self.pdf_path = tk.StringVar()
        self.output_path = tk.StringVar()
        self.json_key_path = tk.StringVar()
        self.tesseract_path_var = tk.StringVar()
        self.api_key = tk.StringVar()
        self.start_page = tk.StringVar()
        self.end_page = tk.StringVar()
        self.api_key_label_var = tk.StringVar()
        self.ocr_only_var = tk.BooleanVar(value=False)
        self.prompt_text = None
        self.progress_var = tk.DoubleVar()
        self.status_var = tk.StringVar(value="Ready")
        self.elapsed_time_var = tk.StringVar(value="Elapsed Time: 00:00")
        self.google_credentials = None
        self.stop_event = threading.Event()
        self.timer_running = False
        self.start_time = 0

        # Mapping of user-visible options to Tesseract and Google Vision language codes
        # For Tesseract use its traineddata short codes; for Vision, use ISO-639 codes if available
        self.LANGUAGE_OPTIONS = {
            "English (eng)": {"tesseract": "eng", "vision": "en"},
            "Sanskrit – IAST / Devanagari (san)": {"tesseract": "san", "vision": "sa"},
            "Hindi (hin)": {"tesseract": "hin", "vision": "hi"},
            "Marathi (mar)": {"tesseract": "mar", "vision": "mr"},
            "Nepali (nep)": {"tesseract": "nep", "vision": "ne"},
            "Konkani (kok)": {"tesseract": "kok", "vision": "kok"},
            "Gujarati (guj)": {"tesseract": "guj", "vision": "gu"},
            "Punjabi – Gurmukhi (pan)": {"tesseract": "pan", "vision": "pa"},
            "Bengali (ben)": {"tesseract": "ben", "vision": "bn"},
            "Assamese (asm)": {"tesseract": "asm", "vision": "as"},
            "Odia (ori)": {"tesseract": "ori", "vision": "or"},
            "Telugu (tel)": {"tesseract": "tel", "vision": "te"},
            "Kannada (kan)": {"tesseract": "kan", "vision": "kn"},
            "Tamil (tam)": {"tesseract": "tam", "vision": "ta"},
            "Malayalam (mal)": {"tesseract": "mal", "vision": "ml"},
            "Sinhala (sin)": {"tesseract": "sin", "vision": "si"},
        }

        # Layout
        self.main_frame = ttk.Frame(self.root, padding="25 25 25 25")
        self.main_frame.pack(expand=True, fill=tk.BOTH)
        self.main_frame.columnconfigure(0, weight=1)
        self.main_frame.rowconfigure(2, weight=1)

        self._create_widgets()
        self.update_api_key_label()
        self.auto_detect_tesseract()

    def _create_widgets(self):
        config_frame = ttk.LabelFrame(self.main_frame, text="Configuration", padding=20)
        config_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        config_frame.columnconfigure(1, weight=1)

        # PDF Upload
        ttk.Label(config_frame, text="PDF Document:").grid(row=0, column=0, sticky=tk.W, pady=8)
        pdf_entry = ttk.Entry(config_frame, textvariable=self.pdf_path, state="readonly", width=60)
        pdf_entry.grid(row=0, column=1, columnspan=2, sticky=tk.EW, padx=5, pady=8)
        ttk.Button(config_frame, text="Browse...", command=self.select_pdf).grid(row=0, column=3, padx=5, pady=8)

        # Output
        ttk.Label(config_frame, text="Output File:").grid(row=1, column=0, sticky=tk.W, pady=8)
        output_entry = ttk.Entry(config_frame, textvariable=self.output_path, state="readonly", width=60)
        output_entry.grid(row=1, column=1, columnspan=2, sticky=tk.EW, padx=5, pady=8)
        ttk.Button(config_frame, text="Save As...", command=self.select_output_file).grid(row=1, column=3, padx=5, pady=8)

        # Page Range
        page_range_frame = ttk.Frame(config_frame)
        page_range_frame.grid(row=2, column=1, columnspan=3, sticky=tk.EW, padx=5, pady=8)
        ttk.Label(config_frame, text="Page Range:").grid(row=2, column=0, sticky=tk.W, pady=8)
        ttk.Label(page_range_frame, text="Start:").pack(side=tk.LEFT, padx=(0, 5))
        start_page_entry = ttk.Entry(page_range_frame, textvariable=self.start_page, width=5)
        start_page_entry.pack(side=tk.LEFT)
        ttk.Label(page_range_frame, text="End:").pack(side=tk.LEFT, padx=(10, 5))
        end_page_entry = ttk.Entry(page_range_frame, textvariable=self.end_page, width=5)
        end_page_entry.pack(side=tk.LEFT)
        ttk.Label(page_range_frame, text="(leave blank)").pack(side=tk.LEFT, padx=(10,0))

        # OCR Engine
        ttk.Label(config_frame, text="OCR Engine:").grid(row=3, column=0, sticky=tk.W, pady=8)
        self.ocr_dropdown = ttk.Combobox(config_frame, values=["Tesseract", "Google Vision"], state="readonly")
        self.ocr_dropdown.set("Tesseract")
        self.ocr_dropdown.grid(row=3, column=1, columnspan=3, sticky=tk.W, padx=5, pady=8)
        self.ocr_dropdown.bind("<<ComboboxSelected>>", self.toggle_ocr_options)

        # Tesseract Path (row 4)
        self.tesseract_path_label = ttk.Label(config_frame, text="Tesseract Path:")
        self.tesseract_path_label.grid(row=4, column=0, sticky=tk.W, pady=8)
        self.tesseract_path_entry = ttk.Entry(config_frame, textvariable=self.tesseract_path_var, width=60)
        self.tesseract_path_entry.grid(row=4, column=1, columnspan=2, sticky=tk.EW, padx=5, pady=8)
        self.tesseract_browse_button = ttk.Button(config_frame, text="Browse...", command=self.select_tesseract_path)
        self.tesseract_browse_button.grid(row=4, column=3, padx=5, pady=8)

        # Google Vision JSON Key (also row 4 but shown only when selected)
        self.google_key_label = ttk.Label(config_frame, text="Google Vision Key:")
        self.google_key_entry = ttk.Entry(config_frame, textvariable=self.json_key_path, state="readonly", width=60)
        self.google_key_button = ttk.Button(config_frame, text="Browse...", command=self.select_json_key)

        # OCR Language selection (row 5)
        ttk.Label(config_frame, text="OCR Language:").grid(row=5, column=0, sticky=tk.W, pady=8)
        self.ocr_lang_var = tk.StringVar()
        self.ocr_lang_dropdown = ttk.Combobox(
            config_frame,
            values=list(self.LANGUAGE_OPTIONS.keys()),
            state="readonly",
            width=40
        )
        self.ocr_lang_dropdown.set("English (eng)")
        self.ocr_lang_dropdown.grid(row=5, column=1, columnspan=3, sticky=tk.W, padx=5)

        # Call once so initial engine-specific widgets are correct
        self.toggle_ocr_options()

        # Mode selection frame (OCR-only checkbox)
        mode_frame = ttk.Frame(self.main_frame)
        mode_frame.grid(row=1, column=0, sticky="w", pady=(0, 10))
        self.ocr_only_checkbox = ttk.Checkbutton(mode_frame, text="Perform OCR Only (No LLM)", variable=self.ocr_only_var, command=self.toggle_llm_fields)
        self.ocr_only_checkbox.pack()

        # LLM Frame
        self.llm_frame = ttk.LabelFrame(self.main_frame, text="LLM Processing", padding=20)
        self.llm_frame.grid(row=2, column=0, sticky="nsew", pady=(0, 20))
        self.llm_frame.columnconfigure(1, weight=1)
        self.llm_frame.rowconfigure(2, weight=1)

        self.llm_provider_label = ttk.Label(self.llm_frame, text="LLM Provider:")
        self.llm_provider_label.grid(row=0, column=0, sticky=tk.W, pady=8)
        self.llm_dropdown = ttk.Combobox(self.llm_frame, values=["OpenAI: gpt-4o", "OpenRouter: deepseek/deepseek-chat"], state="readonly")
        self.llm_dropdown.set("OpenAI: gpt-4o")
        self.llm_dropdown.grid(row=0, column=1, sticky=tk.EW, padx=5, pady=8)
        self.llm_dropdown.bind("<<ComboboxSelected>>", self.update_api_key_label)

        self.api_key_label = ttk.Label(self.llm_frame, textvariable=self.api_key_label_var)
        self.api_key_label.grid(row=1, column=0, sticky=tk.W, pady=8)
        self.api_key_entry = ttk.Entry(self.llm_frame, textvariable=self.api_key, show="*", width=60)
        self.api_key_entry.grid(row=1, column=1, sticky=tk.EW, padx=5, pady=8)

        self.prompt_frame = ttk.LabelFrame(self.llm_frame, text="LLM Prompt", padding=15)
        self.prompt_frame.grid(row=2, column=0, columnspan=2, sticky="nsew", pady=(10,0))
        self.prompt_frame.columnconfigure(0, weight=1)
        self.prompt_frame.rowconfigure(0, weight=1)
        self.prompt_text = tk.Text(self.prompt_frame, height=5, width=70, font=("Helvetica", 11), relief=tk.SOLID, borderwidth=1, wrap=tk.WORD)
        self.prompt_text.grid(row=0, column=0, sticky="nsew")

        # Actions
        action_frame = ttk.Frame(self.main_frame)
        action_frame.grid(row=3, column=0, sticky="ew", pady=(10, 0))
        button_container = ttk.Frame(action_frame)
        button_container.pack(pady=(0, 10))

        self.process_button = tk.Button(button_container, text="Start Processing", command=self.process_all,
                                        bg="#28a745", fg="white", font=("Helvetica", 11, "bold"),
                                        relief="raised", borderwidth=2, padx=10, pady=5,
                                        activebackground="#218838", activeforeground="white")
        self.process_button.pack(side=tk.LEFT, padx=5)

        self.stop_button = tk.Button(button_container, text="Stop", command=self.stop_processing,
                                     bg="#dc3545", fg="white", font=("Helvetica", 11, "bold"),
                                     relief="raised", borderwidth=2, padx=10, pady=5,
                                     activebackground="#c82333", activeforeground="white", state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5)

        status_label = ttk.Label(action_frame, textvariable=self.status_var, font=("Helvetica", 10, "italic"))
        status_label.pack()
        elapsed_label = ttk.Label(action_frame, textvariable=self.elapsed_time_var, font=("Helvetica", 10, "italic"))
        elapsed_label.pack(pady=(0, 5))
        progress_bar = ttk.Progressbar(action_frame, variable=self.progress_var, maximum=100)
        progress_bar.pack(fill=tk.X, pady=5)

    def toggle_llm_fields(self):
        if self.ocr_only_var.get():
            self.llm_frame.grid_remove()
            self.main_frame.rowconfigure(2, weight=0)
        else:
            self.llm_frame.grid()
            self.main_frame.rowconfigure(2, weight=1)

    def _update_timer(self):
        if not self.timer_running: return
        elapsed_seconds = int(time.time() - self.start_time)
        minutes, seconds = divmod(elapsed_seconds, 60)
        self.elapsed_time_var.set(f"Elapsed Time: {minutes:02d}:{seconds:02d}")
        self.root.after(1000, self._update_timer)

    def update_api_key_label(self, event=None):
        selection = self.llm_dropdown.get()
        if "OpenAI" in selection: self.api_key_label_var.set("OpenAI API Key:")
        elif "OpenRouter" in selection: self.api_key_label_var.set("OpenRouter API Key:")
        else: self.api_key_label_var.set("LLM API Key:")

    def toggle_ocr_options(self, event=None):
        if self.ocr_dropdown.get() == "Tesseract":
            self.tesseract_path_label.grid()
            self.tesseract_path_entry.grid()
            self.tesseract_browse_button.grid()
            self.google_key_label.grid_remove()
            self.google_key_entry.grid_remove()
            self.google_key_button.grid_remove()
        elif self.ocr_dropdown.get() == "Google Vision":
            self.tesseract_path_label.grid_remove()
            self.tesseract_path_entry.grid_remove()
            self.tesseract_browse_button.grid_remove()
            self.google_key_label.grid(row=4, column=0, sticky=tk.W, pady=8)
            self.google_key_entry.grid(row=4, column=1, columnspan=2, sticky=tk.EW, padx=5, pady=8)
            self.google_key_button.grid(row=4, column=3, padx=5, pady=8)

    def select_pdf(self):
        path = filedialog.askopenfilename(filetypes=[("PDF files", "*.pdf")])
        if path: self.pdf_path.set(path)

    def select_output_file(self):
        path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text files", ".txt"), ("All files", "*.*")])
        if path: self.output_path.set(path)

    def select_tesseract_path(self):
        if sys.platform == "win32":
            filetypes = [("Tesseract Executable", "tesseract.exe")]
        else:
            filetypes = [("Tesseract Executable", "tesseract")]
        path = filedialog.askopenfilename(filetypes=filetypes, title="Select Tesseract Executable")
        if path: self.tesseract_path_var.set(path)

    def auto_detect_tesseract(self):
        if getattr(sys, 'frozen', False): return
        if sys.platform == "win32":
            path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
            if os.path.exists(path): self.tesseract_path_var.set(path)
        elif sys.platform == "darwin":
            paths = ['/opt/homebrew/bin/tesseract', '/usr/local/bin/tesseract']
            for path in paths:
                if os.path.exists(path):
                    self.tesseract_path_var.set(path)
                    break

    def select_json_key(self):
        path = filedialog.askopenfilename(filetypes=[("JSON files", "*.json")])
        if path:
            try:
                from google.oauth2 import service_account
                self.google_credentials = service_account.Credentials.from_service_account_file(path)
                self.json_key_path.set(path)
                messagebox.showinfo("Success", "Google credentials loaded.")
            except ImportError:
                messagebox.showerror("Error", "The 'google-cloud-vision' library is not installed.")
            except Exception as e:
                messagebox.showerror("Error", f"Invalid Google Service Account JSON: {e}")
                self.google_credentials = None
                self.json_key_path.set("")

    def update_progress(self, stage, value):
        self.status_var.set(stage)
        self.progress_var.set(value)

    def split_text_into_batches(self, text, words_per_batch=1100):
        words = text.split()
        return [" ".join(words[i:i + words_per_batch]) for i in range(0, len(words), words_per_batch)]

    def convert_pdf_to_images(self, pdf_path, start_page, end_page):
        self.root.after(0, self.update_progress, "Converting PDF to images...", 10)
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        if start_page is None: start_page = 1
        if end_page is None: end_page = total_pages
        if not (1 <= start_page <= end_page <= total_pages):
            raise ValueError(f"Invalid page range. The document has {total_pages} pages.")
        images, temp_dir = [], tempfile.mkdtemp(prefix="ocr_images_")
        for i in range(start_page - 1, end_page):
            if self.stop_event.is_set(): return [], None
            pix = doc[i].get_pixmap(dpi=300)
            img_path = os.path.join(temp_dir, f"page_{i}.png")
            pix.save(img_path)
            images.append(img_path)
        return images, temp_dir

    def perform_ocr(self, images, engine):
        self.root.after(0, self.update_progress, f"Performing OCR with {engine}...", 20)
        text = ""
        # Resolve selected language codes
        selected_lang_key = self.ocr_lang_var.get() or "English (eng)"
        lang_entry = self.LANGUAGE_OPTIONS.get(selected_lang_key, {"tesseract": "eng", "vision": "en"})
        tess_lang = lang_entry.get("tesseract", "eng")
        vision_lang = lang_entry.get("vision", "en")

        if engine == "Tesseract":
            tesseract_cmd = self.tesseract_path_var.get()
            if not tesseract_cmd or not os.path.exists(tesseract_cmd):
                raise FileNotFoundError("Tesseract executable not found at the specified path. Please check the path in the Configuration section.")
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

            # Use the selected single language to avoid mixing scripts
            for img_path in images:
                if self.stop_event.is_set(): return ""
                try:
                    text_piece = pytesseract.image_to_string(Image.open(img_path), lang=tess_lang)
                except pytesseract.TesseractError as e:
                    raise RuntimeError(f"Tesseract OCR failed: {e}")
                text += text_piece + "\n"

        elif engine == "Google Vision":
            try:
                from google.cloud import vision
            except ImportError:
                raise ImportError("google-cloud-vision is not installed. Install it with `pip install google-cloud-vision` if you want to use Google Vision OCR.")

            client = vision.ImageAnnotatorClient(credentials=self.google_credentials) if self.google_credentials else vision.ImageAnnotatorClient()
            for img_path in images:
                if self.stop_event.is_set(): return ""
                with open(img_path, "rb") as image_file:
                    content = image_file.read()
                image = vision.Image(content=content)
                image_context = vision.ImageContext(language_hints=[vision_lang])
                response = client.document_text_detection(image=image, image_context=image_context)
                if response.error.message:
                    raise Exception(f"Google Vision API Error: {response.error.message}")
                text += response.full_text_annotation.text + "\n"
        else:
            raise NotImplementedError(f"Unknown OCR engine: {engine}")

        return text

    def call_llm(self, prompt, context, llm_selection, api_key, batch_info=""):
        if self.stop_event.is_set(): return ""
        self.root.after(0, self.update_progress, f"Calling LLM {batch_info}...", self.progress_var.get())
        provider, model_name = llm_selection.split(': ', 1)
        full_prompt = f"{prompt}\n\nPlease process the following text content {batch_info}:\n\n---\n{context}\n---"
        messages = [{"role": "system", "content": "You are an expert assistant..."}, {"role": "user", "content": full_prompt}]
        if provider == "OpenAI":
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(model=model_name, messages=messages)
            return response.choices[0].message.content
        elif provider == "OpenRouter":
            headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
            url = "https://openrouter.ai/api/v1/chat/completions"
            payload = {"model": model_name, "messages": messages}
            response = requests.post(url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        else:
            raise NotImplementedError(f"Provider '{provider}' not supported.")

    def save_output(self, text, output_path):
        self.root.after(0, self.update_progress, "Saving output...", 95)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(text)
        return output_path

    def cleanup_temp_dir(self, temp_dir_path):
        if temp_dir_path and os.path.isdir(temp_dir_path): shutil.rmtree(temp_dir_path)

    def stop_processing(self):
        self.stop_event.set()
        self.stop_button.config(state=tk.DISABLED)
        self.status_var.set("Stopping...")

    def process_all(self):
        if not self.pdf_path.get(): return messagebox.showwarning("Missing Input", "Please select a PDF document.")
        if not self.output_path.get(): return messagebox.showwarning("Missing Input", "Please choose an output file location.")
        if self.ocr_dropdown.get() == "Google Vision" and not self.json_key_path.get(): return messagebox.showwarning("Missing Input", "Please provide the Google Vision JSON key.")
        if self.ocr_dropdown.get() == "Tesseract" and not self.tesseract_path_var.get(): return messagebox.showwarning("Missing Input", "Please provide the path to the Tesseract executable.")
        if not self.ocr_only_var.get():
            if not self.api_key.get(): return messagebox.showwarning("Missing Input", "Please enter your LLM API key.")
            if not self.prompt_text.get("1.0", tk.END).strip(): return messagebox.showwarning("Missing Input", "Please enter a prompt for the LLM.")
        try:
            start_str, end_str = self.start_page.get(), self.end_page.get()
            start = int(start_str) if start_str else None
            end = int(end_str) if end_str else None
            if start is not None and start < 1: return messagebox.showwarning("Invalid Input", "Start page must be 1 or greater.")
            if start and end and start > end: return messagebox.showwarning("Invalid Input", "Start page cannot be greater than end page.")
        except ValueError:
            return messagebox.showwarning("Invalid Input", "Page numbers must be integers.")

        self.stop_event.clear()
        self.process_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.timer_running = True
        self.start_time = time.time()
        self._update_timer()
        worker_thread = threading.Thread(target=self._processing_worker, args=(start, end))
        worker_thread.start()

    def _processing_worker(self, start_page, end_page):
        temp_dir_path = None
        try:
            images, temp_dir_path = self.convert_pdf_to_images(self.pdf_path.get(), start_page, end_page)
            if self.stop_event.is_set(): raise InterruptedError("Process stopped by user.")
            if not images: return messagebox.showinfo("Information", "No pages found in the specified range.")
            context_text = self.perform_ocr(images, self.ocr_dropdown.get())
            if self.stop_event.is_set(): raise InterruptedError("Process stopped by user.")
            if self.ocr_only_var.get():
                final_output = context_text
            else:
                self.root.after(0, self.update_progress, "Splitting text into batches...", 40)
                batches = self.split_text_into_batches(context_text)
                num_batches = len(batches)
                llm_results = []
                llm_progress_start, llm_progress_end = 40, 95
                for i, batch in enumerate(batches):
                    if self.stop_event.is_set(): raise InterruptedError("Process stopped by user.")
                    progress = llm_progress_start + ((i + 1) / num_batches) * (llm_progress_end - llm_progress_start)
                    self.root.after(0, self.progress_var.set, progress)
                    batch_info = f"(Batch {i+1} of {num_batches})"
                    prompt = self.prompt_text.get("1.0", tk.END).strip()
                    llm_selection = self.llm_dropdown.get()
                    api_key = self.api_key.get()
                    result = self.call_llm(prompt, batch, llm_selection, api_key, batch_info)
                    llm_results.append(result)
                final_output = "\n\n---\n\n".join(llm_results)

            output_file = self.save_output(final_output, self.output_path.get())
            self.root.after(0, self.update_progress, "Done!", 100)
            messagebox.showinfo("Success", f"Processing complete! Output saved to:\n{output_file}")
        except InterruptedError as e:
            self.root.after(0, self.update_progress, str(e), 0)
        except Exception as e:
            messagebox.showerror("Error", f"An unexpected error occurred:\n{e}")
            self.root.after(0, self.update_progress, "Failed", 0)
        finally:
            self.timer_running = False
            if temp_dir_path: self.cleanup_temp_dir(temp_dir_path)
            self.process_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)
            if not self.stop_event.is_set():
                self.root.after(2000, self.update_progress, "Ready", 0)
                self.root.after(2000, self.elapsed_time_var.set, "Elapsed Time: 00:00")


if __name__ == "__main__":
    root = tk.Tk()
    app = OcrLlmApp(root)
    root.mainloop()
