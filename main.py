# This code is cross platform.
# To run the build, do as follows:
# For macOS: "pyinstaller --name="OCR-LLM Processor" --windowed --icon="YourIcon.icns" --add-data="/opt/homebrew/opt/tesseract/share/tessdata:tessdata" main.py")
# For Windows: "pyinstaller --name="OCR-LLM Processor" --windowed --icon="YourIcon.ico" --add-data="C:\Program Files\Tesseract-OCR\tessdata;tessdata" main.py")

import tkinter as tk
from tkinter import filedialog, messagebox, ttk, font
import threading
import os
import json
from PIL import Image
import pytesseract
import fitz  # PyMuPDF
import requests
import tempfile # For creating temporary directories safely
import shutil # For safely removing directories
import sys # To help find the Tesseract path
import sv_ttk # For a modern look and feel
import time # To track elapsed time

# The 'openai' library is now used for OpenAI calls.
# It will be imported dynamically to provide a clear error message if not installed.

# The 'google-cloud-vision' library is an optional dependency.
# It will be imported dynamically if the user selects it.

# ========== TESSERACT PATH CONFIGURATION ==========
def configure_tesseract_path():
    """
    Finds and sets the path for the Tesseract executable. When the app is bundled
    with PyInstaller, it points to the Tesseract engine inside the bundle.
    If run as a script, it checks standard installation paths on Windows and macOS.
    """
    # If the app is running as a bundled executable (created by PyInstaller)
    if getattr(sys, 'frozen', False):
        bundle_dir = sys._MEIPASS
        if sys.platform == "win32":
            tesseract_path = os.path.join(bundle_dir, 'tesseract', 'tesseract.exe')
        elif sys.platform == "darwin":
            tesseract_path = os.path.join(bundle_dir, 'tesseract', 'tesseract')
        
        pytesseract.pytesseract.tesseract_cmd = tesseract_path
        tessdata_dir = os.path.join(bundle_dir, 'tessdata')
        os.environ['TESSDATA_PREFIX'] = tessdata_dir
    
    # If running as a normal script, check default install locations
    # to avoid issues where Tesseract is not in the system's PATH.
    elif sys.platform == "win32":
        tesseract_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        if os.path.exists(tesseract_path):
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
    elif sys.platform == "darwin":
        # Common Homebrew paths for Tesseract on macOS
        tesseract_paths = [
            '/opt/homebrew/bin/tesseract', # For Apple Silicon Macs
            '/usr/local/bin/tesseract'     # For Intel Macs
        ]
        for path in tesseract_paths:
            if os.path.exists(path):
                pytesseract.pytesseract.tesseract_cmd = path
                return

# ========== APPLICATION CLASS ==========

class OcrLlmApp:
    def __init__(self, root):
        """Initialize the application's GUI."""
        self.root = root
        self.root.title("OCR + LLM Document Processor")
        # Set initial and minimum window size
        self.root.geometry("800x900")
        self.root.minsize(800, 600) # Set a more flexible minimum height

        # Use the sv-ttk theme for a modern, happier look
        sv_ttk.set_theme("light")

        # --- Instance Variables ---
        self.pdf_path = tk.StringVar()
        self.output_path = tk.StringVar()
        self.json_key_path = tk.StringVar()
        self.api_key = tk.StringVar()
        self.start_page = tk.StringVar()
        self.end_page = tk.StringVar()
        self.api_key_label_var = tk.StringVar()
        self.ocr_only_var = tk.BooleanVar(value=False) # Variable for the new checkbox
        self.prompt_text = None
        self.progress_var = tk.DoubleVar()
        self.status_var = tk.StringVar(value="Ready")
        self.elapsed_time_var = tk.StringVar(value="Elapsed Time: 00:00")
        self.google_credentials = None
        self.stop_event = threading.Event()
        self.timer_running = False
        self.start_time = 0

        # --- Main Layout ---
        self.main_frame = ttk.Frame(self.root, padding="25 25 25 25")
        # Configure grid layout for resizing
        self.main_frame.pack(expand=True, fill=tk.BOTH)
        self.main_frame.columnconfigure(0, weight=1)
        self.main_frame.rowconfigure(2, weight=1) # Allow the LLM frame row to expand/disappear

        self._create_widgets()
        self.update_api_key_label()
        configure_tesseract_path()

    def _create_widgets(self):
        """Create and arrange all the GUI widgets with a polished layout."""
        # --- Default Font Setup ---
        default_font = font.nametofont("TkDefaultFont")
        default_font.configure(size=11, family="Segoe UI" if sys.platform == "win32" else "Helvetica Neue")
        self.root.option_add("*Font", default_font)

        # --- Styles ---
        style = ttk.Style()
        style.configure("Accent.TButton", font=(default_font.cget("family"), 11, "bold"), padding=6)
        style.configure("Danger.TButton", font=(default_font.cget("family"), 11, "bold"), padding=6, foreground="white", background="#dc3545")
        style.map("Danger.TButton", background=[("active", "#c82333")])

        # --- Configuration Frame ---
        config_frame = ttk.LabelFrame(self.main_frame, text="Configuration", padding=20)
        config_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        config_frame.columnconfigure(1, weight=1)

        # PDF Upload
        ttk.Label(config_frame, text="PDF Document:").grid(row=0, column=0, sticky=tk.W, pady=6)
        pdf_entry = ttk.Entry(config_frame, textvariable=self.pdf_path, state="readonly", width=60)
        pdf_entry.grid(row=0, column=1, columnspan=2, sticky=tk.EW, padx=6, pady=6)
        ttk.Button(config_frame, text="Browse...", command=self.select_pdf).grid(row=0, column=3, padx=6, pady=6)

        # Target Save File
        ttk.Label(config_frame, text="Output File:").grid(row=1, column=0, sticky=tk.W, pady=6)
        output_entry = ttk.Entry(config_frame, textvariable=self.output_path, state="readonly", width=60)
        output_entry.grid(row=1, column=1, columnspan=2, sticky=tk.EW, padx=6, pady=6)
        ttk.Button(config_frame, text="Save As...", command=self.select_output_file).grid(row=1, column=3, padx=6, pady=6)

        # Page Range Selection
        page_range_frame = ttk.Frame(config_frame)
        page_range_frame.grid(row=2, column=1, columnspan=3, sticky=tk.EW, padx=6, pady=6)
        ttk.Label(config_frame, text="Page Range:").grid(row=2, column=0, sticky=tk.W, pady=6)

        ttk.Label(page_range_frame, text="Start:").pack(side=tk.LEFT, padx=(0, 5))
        ttk.Entry(page_range_frame, textvariable=self.start_page, width=5).pack(side=tk.LEFT)

        ttk.Label(page_range_frame, text="End:").pack(side=tk.LEFT, padx=(10, 5))
        ttk.Entry(page_range_frame, textvariable=self.end_page, width=5).pack(side=tk.LEFT)

        ttk.Label(page_range_frame, text="(leave blank for all pages)").pack(side=tk.LEFT, padx=(10, 0))

        # OCR Engine
        ttk.Label(config_frame, text="OCR Engine:").grid(row=3, column=0, sticky=tk.W, pady=6)
        self.ocr_dropdown = ttk.Combobox(config_frame, values=["Tesseract", "Google Vision"], state="readonly")
        self.ocr_dropdown.set("Tesseract")
        self.ocr_dropdown.grid(row=3, column=1, columnspan=3, sticky=tk.W, padx=6, pady=6)
        self.ocr_dropdown.bind("<<ComboboxSelected>>", self.toggle_google_key_visibility)

        # Google Vision JSON Key
        self.google_key_label = ttk.Label(config_frame, text="Google Vision Key:")
        self.google_key_label.grid(row=4, column=0, sticky=tk.W, pady=6)
        self.google_key_entry = ttk.Entry(config_frame, textvariable=self.json_key_path, state="readonly", width=60)
        self.google_key_entry.grid(row=4, column=1, columnspan=2, sticky=tk.EW, padx=6, pady=6)
        self.google_key_button = ttk.Button(config_frame, text="Browse...", command=self.select_json_key)
        self.google_key_button.grid(row=4, column=3, padx=6, pady=6)
        self.google_key_label.grid_remove()
        self.google_key_entry.grid_remove()
        self.google_key_button.grid_remove()

        # --- Mode Selection Frame ---
        mode_frame = ttk.Frame(self.main_frame)
        mode_frame.grid(row=1, column=0, sticky="w", pady=(0, 10), padx=10)
        self.ocr_only_checkbox = ttk.Checkbutton(mode_frame, text="Perform OCR Only (No LLM)", variable=self.ocr_only_var, command=self.toggle_llm_fields)
        self.ocr_only_checkbox.pack()

        # --- LLM Frame ---
        self.llm_frame = ttk.LabelFrame(self.main_frame, text="LLM Processing", padding=20)
        self.llm_frame.grid(row=2, column=0, sticky="nsew", pady=(0, 20))
        self.llm_frame.columnconfigure(1, weight=1)
        self.llm_frame.rowconfigure(2, weight=1)

        self.llm_provider_label = ttk.Label(self.llm_frame, text="LLM Provider:")
        self.llm_provider_label.grid(row=0, column=0, sticky=tk.W, pady=6)
        self.llm_dropdown = ttk.Combobox(self.llm_frame, values=["OpenAI: gpt-4o", "OpenRouter: deepseek/deepseek-chat"], state="readonly")
        self.llm_dropdown.set("OpenAI: gpt-4o")
        self.llm_dropdown.grid(row=0, column=1, sticky=tk.EW, padx=6, pady=6)
        self.llm_dropdown.bind("<<ComboboxSelected>>", self.update_api_key_label)

        self.api_key_label = ttk.Label(self.llm_frame, textvariable=self.api_key_label_var)
        self.api_key_label.grid(row=1, column=0, sticky=tk.W, pady=6)
        self.api_key_entry = ttk.Entry(self.llm_frame, textvariable=self.api_key, show="*", width=60)
        self.api_key_entry.grid(row=1, column=1, sticky=tk.EW, padx=6, pady=6)

        # Prompt Frame with Scrollbar
        self.prompt_frame = ttk.LabelFrame(self.llm_frame, text="LLM Prompt", padding=15)
        self.prompt_frame.grid(row=2, column=0, columnspan=2, sticky="nsew", pady=(10, 0))
        self.prompt_frame.columnconfigure(0, weight=1)
        self.prompt_frame.rowconfigure(0, weight=1)

        self.prompt_text = tk.Text(self.prompt_frame, height=6, width=70, wrap=tk.WORD, relief=tk.FLAT, borderwidth=1)
        self.prompt_text.grid(row=0, column=0, sticky="nsew", padx=(0, 5))

        prompt_scrollbar = ttk.Scrollbar(self.prompt_frame, orient="vertical", command=self.prompt_text.yview)
        prompt_scrollbar.grid(row=0, column=1, sticky="ns")
        self.prompt_text.config(yscrollcommand=prompt_scrollbar.set)

        # --- Action Frame ---
        action_frame = ttk.Frame(self.main_frame)
        action_frame.grid(row=3, column=0, sticky="ew", pady=(10, 0))

        button_container = ttk.Frame(action_frame)
        button_container.pack(pady=(0, 10))

        self.process_button = ttk.Button(button_container, text="Start Processing", command=self.process_all, style="Accent.TButton")
        self.process_button.pack(side=tk.LEFT, padx=6)

        self.stop_button = ttk.Button(button_container, text="Stop", command=self.stop_processing, style="Danger.TButton", state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=6)

        status_label = ttk.Label(action_frame, textvariable=self.status_var, font=(default_font.cget("family"), 10, "italic"))
        status_label.pack()

        elapsed_label = ttk.Label(action_frame, textvariable=self.elapsed_time_var, font=(default_font.cget("family"), 10, "italic"))
        elapsed_label.pack(pady=(0, 5))

        progress_bar = ttk.Progressbar(action_frame, variable=self.progress_var, maximum=100)
        progress_bar.pack(fill=tk.X, pady=5)



    def toggle_llm_fields(self):
        """Hides or shows the entire LLM section."""
        if self.ocr_only_var.get():
            self.llm_frame.grid_remove()
            # Adjust row configuration to allow bottom frame to move up
            self.main_frame.rowconfigure(2, weight=0)
        else:
            self.llm_frame.grid()
            # Restore row configuration
            self.main_frame.rowconfigure(2, weight=1)


    def _update_timer(self):
        """Updates the elapsed time label every second."""
        if not self.timer_running:
            return
        elapsed_seconds = int(time.time() - self.start_time)
        minutes, seconds = divmod(elapsed_seconds, 60)
        self.elapsed_time_var.set(f"Elapsed Time: {minutes:02d}:{seconds:02d}")
        self.root.after(1000, self._update_timer)

    def update_api_key_label(self, event=None):
        selection = self.llm_dropdown.get()
        if "OpenAI" in selection:
            self.api_key_label_var.set("OpenAI API Key:")
        elif "OpenRouter" in selection:
            self.api_key_label_var.set("OpenRouter API Key:")
        else:
            self.api_key_label_var.set("LLM API Key:")

    def toggle_google_key_visibility(self, event=None):
        """Shows or hides the Google Vision key widgets."""
        if self.ocr_dropdown.get() == "Google Vision":
            self.google_key_label.grid()
            self.google_key_entry.grid()
            self.google_key_button.grid()
        else:
            self.google_key_label.grid_remove()
            self.google_key_entry.grid_remove()
            self.google_key_button.grid_remove()

    def select_pdf(self):
        path = filedialog.askopenfilename(filetypes=[("PDF files", "*.pdf")])
        if path:
            self.pdf_path.set(path)
            
    def select_output_file(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
            title="Choose location to save output file"
        )
        if path:
            self.output_path.set(path)

    def select_json_key(self):
        path = filedialog.askopenfilename(filetypes=[("JSON files", "*.json")])
        if path:
            try:
                from google.oauth2 import service_account
                self.google_credentials = service_account.Credentials.from_service_account_file(path)
                self.json_key_path.set(path)
                messagebox.showinfo("Success", "Google credentials loaded successfully.")
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
        """
        Splits a large text into smaller batches based on word count.
        1500 tokens is roughly 1125 words. We use 1100 for a safe margin.
        """
        words = text.split()
        batches = []
        for i in range(0, len(words), words_per_batch):
            batch = words[i:i + words_per_batch]
            batches.append(" ".join(batch))
        return batches

    def convert_pdf_to_images(self, pdf_path, start_page, end_page):
        self.root.after(0, self.update_progress, "Converting PDF to images...", 10)
        doc = fitz.open(pdf_path)
        
        total_pages = len(doc)
        if start_page is None: start_page = 1
        if end_page is None: end_page = total_pages
        
        if not (1 <= start_page <= end_page <= total_pages):
            raise ValueError(f"Invalid page range. The document has {total_pages} pages.")
            
        images = []
        temp_dir = tempfile.mkdtemp(prefix="ocr_images_")
        
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
        if engine == "Tesseract":
            for img_path in images:
                if self.stop_event.is_set(): return ""
                try:
                    text += pytesseract.image_to_string(Image.open(img_path), lang="eng+san") + "\n"
                except pytesseract.TesseractNotFoundError:
                    raise Exception("Tesseract not found. Please install it and ensure it's in your system's PATH.")
        elif engine == "Google Vision":
            if self.stop_event.is_set(): return ""
            from google.cloud import vision
            client = vision.ImageAnnotatorClient(credentials=self.google_credentials)
            for img_path in images:
                if self.stop_event.is_set(): return ""
                with open(img_path, "rb") as image_file:
                    content = image_file.read()
                image = vision.Image(content=content)
                response = client.document_text_detection(image=image)
                if response.error.message:
                    raise Exception(f"Google Vision API Error: {response.error.message}")
                text += response.full_text_annotation.text + "\n"
        return text

    def call_llm(self, prompt, context, llm_selection, api_key, batch_info=""):
        if self.stop_event.is_set(): return ""
        self.root.after(0, self.update_progress, f"Calling LLM {batch_info}...", self.progress_var.get())
        
        try:
            provider, model_name = llm_selection.split(': ', 1)
        except ValueError:
            raise ValueError("Invalid LLM selection format.")

        full_prompt = (
            f"{prompt}\n\n"
            f"Please process the following text content {batch_info}:\n\n"
            f"---\n{context}\n---"
        )
        
        messages = [
            {"role": "system", "content": "You are an expert assistant. You will be given text from a document, possibly in batches. Analyze the provided text and follow the user's instructions carefully."},
            {"role": "user", "content": full_prompt}
        ]

        if provider == "OpenAI":
            try:
                from openai import OpenAI
                client = OpenAI(api_key=api_key)
                response = client.chat.completions.create(model=model_name, messages=messages)
                return response.choices[0].message.content
            except ImportError:
                raise ImportError("The 'openai' library is not installed.")

        elif provider == "OpenRouter":
            headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
            url = "https://openrouter.ai/api/v1/chat/completions"
            payload = {"model": model_name, "messages": messages}
            response = requests.post(url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]
        else:
            raise NotImplementedError(f"Provider '{provider}' not supported.")

    def save_output(self, text, output_path):
        self.root.after(0, self.update_progress, "Saving output...", 95)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(text)
        return output_path

    def cleanup_temp_dir(self, temp_dir_path):
        if temp_dir_path and os.path.isdir(temp_dir_path):
            shutil.rmtree(temp_dir_path)

    def stop_processing(self):
        self.stop_event.set()
        self.stop_button.config(state=tk.DISABLED)
        self.status_var.set("Stopping...")

    def process_all(self):
        # --- Input Validation ---
        if not self.pdf_path.get():
            messagebox.showwarning("Missing Input", "Please select a PDF document.")
            return
        if not self.output_path.get():
            messagebox.showwarning("Missing Input", "Please choose an output file location.")
            return
        if self.ocr_dropdown.get() == "Google Vision" and not self.json_key_path.get():
            messagebox.showwarning("Missing Input", "Please provide the Google Vision JSON key.")
            return
        if not self.ocr_only_var.get():
            if not self.api_key.get():
                messagebox.showwarning("Missing Input", "Please enter your LLM API key.")
                return
            if not self.prompt_text.get("1.0", tk.END).strip():
                messagebox.showwarning("Missing Input", "Please enter a prompt for the LLM.")
                return
        
        try:
            start_str = self.start_page.get()
            end_str = self.end_page.get()
            start = int(start_str) if start_str else None
            end = int(end_str) if end_str else None
            if start is not None and start < 1:
                messagebox.showwarning("Invalid Input", "Start page must be 1 or greater.")
                return
            if start and end and start > end:
                messagebox.showwarning("Invalid Input", "Start page cannot be greater than end page.")
                return
        except ValueError:
            messagebox.showwarning("Invalid Input", "Page numbers must be integers.")
            return

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
            if not images:
                messagebox.showinfo("Information", "No pages found in the specified range.")
                return
            
            context_text = self.perform_ocr(images, self.ocr_dropdown.get())
            if self.stop_event.is_set(): raise InterruptedError("Process stopped by user.")
            
            final_output = ""
            if self.ocr_only_var.get():
                final_output = context_text
            else:
                # Process with LLM, including batching
                self.root.after(0, self.update_progress, "Splitting text into batches...", 40)
                batches = self.split_text_into_batches(context_text)
                num_batches = len(batches)
                llm_results = []
                
                llm_progress_start = 40
                llm_progress_end = 95
                
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
            if temp_dir_path:
                self.cleanup_temp_dir(temp_dir_path)
            self.process_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)
            if not self.stop_event.is_set():
                self.root.after(2000, self.update_progress, "Ready", 0)
                self.root.after(2000, self.elapsed_time_var.set, "Elapsed Time: 00:00")


if __name__ == "__main__":
    root = tk.Tk()
    app = OcrLlmApp(root)
    root.mainloop()

