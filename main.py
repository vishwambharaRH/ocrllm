import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading
import os
import json
from PIL import Image
import pytesseract
import fitz  # PyMuPDF
import requests

# The 'openai' library is now used for OpenAI calls.
# It will be imported dynamically to provide a clear error message if not installed.

# The 'google-cloud-vision' library is an optional dependency.
# It will be imported dynamically if the user selects it.

# ========== APPLICATION CLASS ==========

class OcrLlmApp:
    def __init__(self, root):
        """Initialize the application's GUI."""
        self.root = root
        self.root.title("OCR + LLM Document Processor")
        self.root.geometry("800x800")
        self.root.configure(bg="#f0f0f0")

        # --- Instance Variables ---
        self.pdf_path = tk.StringVar()
        self.json_key_path = tk.StringVar()
        self.api_key = tk.StringVar()
        self.start_page = tk.StringVar()
        self.end_page = tk.StringVar()
        self.api_key_label_var = tk.StringVar() # For dynamic label
        self.prompt_text = None
        self.progress_var = tk.DoubleVar()
        self.status_var = tk.StringVar(value="Ready")
        self.google_credentials = None

        # --- Style Configuration ---
        self.style = ttk.Style(self.root)
        self.style.theme_use('clam')
        self.style.configure("TLabel", background="#f0f0f0", font=("Helvetica", 12))
        self.style.configure("TButton", font=("Helvetica", 11, "bold"), padding=10)
        self.style.configure("TFrame", background="#f0f0f0")
        self.style.configure("Header.TLabel", font=("Helvetica", 16, "bold"))
        self.style.configure("Accent.TButton", background="#0078D7", foreground="white")
        self.style.map("Accent.TButton",
            background=[('active', '#005a9e')]
        )

        # --- Main Layout ---
        self.main_frame = ttk.Frame(self.root, padding="20 20 20 20")
        self.main_frame.pack(expand=True, fill=tk.BOTH)

        self._create_widgets()
        self.update_api_key_label() # Set initial label

    def _create_widgets(self):
        """Create and arrange all the GUI widgets."""
        # --- Configuration Frame ---
        config_frame = ttk.LabelFrame(self.main_frame, text="Configuration", padding=15)
        config_frame.pack(fill=tk.X, pady=(0, 20))
        config_frame.columnconfigure(1, weight=1)

        # PDF Upload
        ttk.Label(config_frame, text="PDF Document:").grid(row=0, column=0, sticky=tk.W, pady=5)
        pdf_entry = ttk.Entry(config_frame, textvariable=self.pdf_path, state="readonly", width=60)
        pdf_entry.grid(row=0, column=1, columnspan=2, sticky=tk.EW, padx=5)
        ttk.Button(config_frame, text="Browse...", command=self.select_pdf).grid(row=0, column=3, padx=5)

        # Page Range Selection
        page_range_frame = ttk.Frame(config_frame)
        page_range_frame.grid(row=1, column=1, columnspan=3, sticky=tk.EW, padx=5)
        ttk.Label(config_frame, text="Page Range:").grid(row=1, column=0, sticky=tk.W, pady=5)
        
        ttk.Label(page_range_frame, text="Start:").pack(side=tk.LEFT, padx=(0, 5))
        start_page_entry = ttk.Entry(page_range_frame, textvariable=self.start_page, width=5)
        start_page_entry.pack(side=tk.LEFT)
        
        ttk.Label(page_range_frame, text="End:").pack(side=tk.LEFT, padx=(10, 5))
        end_page_entry = ttk.Entry(page_range_frame, textvariable=self.end_page, width=5)
        end_page_entry.pack(side=tk.LEFT)
        
        ttk.Label(page_range_frame, text="(leave blank for all pages)").pack(side=tk.LEFT, padx=(10,0))


        # OCR Engine
        ttk.Label(config_frame, text="OCR Engine:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.ocr_dropdown = ttk.Combobox(config_frame, values=["Tesseract", "Google Vision"], state="readonly")
        self.ocr_dropdown.set("Tesseract")
        self.ocr_dropdown.grid(row=2, column=1, columnspan=3, sticky=tk.W, padx=5)
        self.ocr_dropdown.bind("<<ComboboxSelected>>", self.toggle_google_key_visibility)

        # Google Vision JSON Key (Initially hidden)
        self.google_key_label = ttk.Label(config_frame, text="Google Vision Key:")
        self.google_key_entry = ttk.Entry(config_frame, textvariable=self.json_key_path, state="readonly", width=60)
        self.google_key_button = ttk.Button(config_frame, text="Browse...", command=self.select_json_key)
        
        # LLM Provider
        ttk.Label(config_frame, text="LLM Provider:").grid(row=4, column=0, sticky=tk.W, pady=5)
        self.llm_dropdown = ttk.Combobox(config_frame, values=["OpenAI: gpt-4o", "OpenRouter: deepseek/deepseek-chat"], state="readonly")
        self.llm_dropdown.set("OpenAI: gpt-4o")
        self.llm_dropdown.grid(row=4, column=1, columnspan=3, sticky=tk.W, padx=5)
        self.llm_dropdown.bind("<<ComboboxSelected>>", self.update_api_key_label)

        # API Key Entry (with dynamic label)
        ttk.Label(config_frame, textvariable=self.api_key_label_var).grid(row=5, column=0, sticky=tk.W, pady=5)
        api_key_entry = ttk.Entry(config_frame, textvariable=self.api_key, show="*", width=60)
        api_key_entry.grid(row=5, column=1, columnspan=3, sticky=tk.EW, padx=5)

        # --- Prompt Frame ---
        prompt_frame = ttk.LabelFrame(self.main_frame, text="LLM Prompt", padding=15)
        prompt_frame.pack(expand=True, fill=tk.BOTH, pady=(0, 20))
        
        self.prompt_text = tk.Text(prompt_frame, height=8, width=70, font=("Helvetica", 11), relief=tk.SOLID, borderwidth=1, wrap=tk.WORD)
        self.prompt_text.pack(expand=True, fill=tk.BOTH, side=tk.LEFT)
        
        scrollbar = ttk.Scrollbar(prompt_frame, orient=tk.VERTICAL, command=self.prompt_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.prompt_text.config(yscrollcommand=scrollbar.set)

        # --- Action Frame ---
        action_frame = ttk.Frame(self.main_frame)
        action_frame.pack(fill=tk.X)

        # Process Button
        process_button = ttk.Button(action_frame, text="Start Processing", command=self.process_all, style="Accent.TButton")
        process_button.pack(pady=(0, 10))

        # Progress Bar and Status
        status_label = ttk.Label(action_frame, textvariable=self.status_var, font=("Helvetica", 10, "italic"))
        status_label.pack()
        progress_bar = ttk.Progressbar(action_frame, variable=self.progress_var, maximum=100)
        progress_bar.pack(fill=tk.X, pady=5)

    def update_api_key_label(self, event=None):
        """Updates the API key label based on the selected LLM provider."""
        selection = self.llm_dropdown.get()
        if "OpenAI" in selection:
            self.api_key_label_var.set("OpenAI API Key:")
        elif "OpenRouter" in selection:
            self.api_key_label_var.set("OpenRouter API Key:")
        else:
            self.api_key_label_var.set("LLM API Key:")

    def toggle_google_key_visibility(self, event=None):
        """Show or hide the Google Vision JSON key selection widgets."""
        if self.ocr_dropdown.get() == "Google Vision":
            self.google_key_label.grid(row=3, column=0, sticky=tk.W, pady=5)
            self.google_key_entry.grid(row=3, column=1, columnspan=2, sticky=tk.EW, padx=5)
            self.google_key_button.grid(row=3, column=3, padx=5)
        else:
            self.google_key_label.grid_remove()
            self.google_key_entry.grid_remove()
            self.google_key_button.grid_remove()

    def select_pdf(self):
        """Open a dialog to select a PDF file."""
        path = filedialog.askopenfilename(filetypes=[("PDF files", "*.pdf")])
        if path:
            self.pdf_path.set(path)

    def select_json_key(self):
        """Open a dialog to select a Google Cloud JSON key file."""
        path = filedialog.askopenfilename(filetypes=[("JSON files", "*.json")])
        if path:
            try:
                from google.oauth2 import service_account
                self.google_credentials = service_account.Credentials.from_service_account_file(path)
                self.json_key_path.set(path)
                messagebox.showinfo("Success", "Google credentials loaded successfully.")
            except ImportError:
                messagebox.showerror("Error", "The 'google-cloud-vision' library is not installed. Please install it to use this feature.")
            except Exception as e:
                messagebox.showerror("Error", f"Invalid Google Service Account JSON: {e}")
                self.google_credentials = None
                self.json_key_path.set("")

    def update_progress(self, stage, value):
        """Update the progress bar and status label from the main thread."""
        self.status_var.set(stage)
        self.progress_var.set(value)

    def convert_pdf_to_images(self, pdf_path, start_page, end_page):
        """Convert a specified range of PDF pages into PNG images."""
        self.root.after(0, self.update_progress, "Converting PDF to images...", 10)
        doc = fitz.open(pdf_path)
        
        # Validate page range against the document's page count
        total_pages = len(doc)
        if start_page is None: start_page = 1
        if end_page is None: end_page = total_pages
        
        if not (1 <= start_page <= end_page <= total_pages):
            raise ValueError(f"Invalid page range. The document has {total_pages} pages.")
            
        images = []
        temp_dir = "temp_ocr_images"
        os.makedirs(temp_dir, exist_ok=True)
        
        # Loop through the specified page range (adjusting for 0-based index)
        for i in range(start_page - 1, end_page):
            pix = doc[i].get_pixmap(dpi=300)
            img_path = os.path.join(temp_dir, f"page_{i}.png")
            pix.save(img_path)
            images.append(img_path)
        return images

    def perform_ocr(self, images, engine):
        """Perform OCR on a list of images using the selected engine."""
        self.root.after(0, self.update_progress, f"Performing OCR with {engine}...", 30)
        text = ""
        if engine == "Tesseract":
            for img_path in images:
                try:
                    text += pytesseract.image_to_string(Image.open(img_path), lang="eng+san") + "\n"
                except pytesseract.TesseractNotFoundError:
                    raise Exception("Tesseract is not installed or not in your PATH. Please install it to use this feature.")
        elif engine == "Google Vision":
            from google.cloud import vision
            client = vision.ImageAnnotatorClient(credentials=self.google_credentials)
            for img_path in images:
                with open(img_path, "rb") as image_file:
                    content = image_file.read()
                image = vision.Image(content=content)
                response = client.document_text_detection(image=image)
                if response.error.message:
                    raise Exception(f"Google Vision API Error: {response.error.message}")
                text += response.full_text_annotation.text + "\n"
        return text

    def call_llm(self, prompt, context, llm_selection, api_key):
        """Call the appropriate LLM API based on user selection."""
        self.root.after(0, self.update_progress, "Calling LLM...", 70)
        
        try:
            provider, model_name = llm_selection.split(': ', 1)
        except ValueError:
            raise ValueError("Invalid LLM selection format. Expected 'Provider: model_name'.")

        messages = [
            {"role": "system", "content": "You are an expert assistant. Analyze the provided text and follow the user's instructions carefully."},
            {"role": "user", "content": f"Here is the context extracted from a document:\n\n---\n{context}\n---\n\nBased on this context, please perform the following task:\n{prompt}"}
        ]

        if provider == "OpenAI":
            try:
                from openai import OpenAI
                client = OpenAI(api_key=api_key)
                response = client.chat.completions.create(
                    model=model_name,
                    messages=messages
                )
                return response.choices[0].message.content
            except ImportError:
                raise ImportError("The 'openai' library is not installed. Please run 'pip install openai' to use this feature.")

        elif provider == "OpenRouter":
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            url = "https://openrouter.ai/api/v1/chat/completions"
            payload = {"model": model_name, "messages": messages}
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]
        else:
            raise NotImplementedError(f"The LLM provider '{provider}' is not supported.")

    def save_output(self, text):
        """Save the final output to a text file."""
        self.root.after(0, self.update_progress, "Saving output...", 95)
        output_path = "ocr_llm_output.txt"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(text)
        return output_path

    def cleanup_images(self, image_paths):
        """Delete the temporary image files."""
        temp_dir = "temp_ocr_images"
        for path in image_paths:
            try:
                os.remove(path)
            except OSError as e:
                print(f"Error removing file {path}: {e}")
        try:
            os.rmdir(temp_dir)
        except OSError as e:
            print(f"Error removing directory {temp_dir}: {e}")

    def process_all(self):
        """Validate inputs and start the processing thread."""
        # --- Input Validation ---
        if not self.pdf_path.get():
            messagebox.showwarning("Missing Input", "Please select a PDF document.")
            return
        if self.ocr_dropdown.get() == "Google Vision" and not self.json_key_path.get():
            messagebox.showwarning("Missing Input", "Please provide the Google Vision JSON key.")
            return
        if not self.api_key.get():
            messagebox.showwarning("Missing Input", "Please enter your LLM API key.")
            return
        if not self.prompt_text.get("1.0", tk.END).strip():
            messagebox.showwarning("Missing Input", "Please enter a prompt.")
            return
            
        # --- Page Range Validation ---
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

        # --- Start Worker Thread ---
        worker_thread = threading.Thread(target=self._processing_worker, args=(start, end))
        worker_thread.start()

    def _processing_worker(self, start_page, end_page):
        """The actual processing logic that runs in a separate thread."""
        try:
            # Step 1: Convert PDF to Images with page range
            images = self.convert_pdf_to_images(self.pdf_path.get(), start_page, end_page)
            
            if not images:
                messagebox.showinfo("Information", "No pages found in the specified range to process.")
                self.root.after(0, self.update_progress, "Ready", 0)
                return
            
            # Step 2: Perform OCR
            context_text = self.perform_ocr(images, self.ocr_dropdown.get())
            
            # Step 3: Call LLM
            prompt = self.prompt_text.get("1.0", tk.END).strip()
            llm_selection = self.llm_dropdown.get()
            api_key = self.api_key.get()
            final_output = self.call_llm(prompt, context_text, llm_selection, api_key)
            
            # Step 4: Save Output
            output_file = self.save_output(final_output)
            
            self.root.after(0, self.update_progress, "Done!", 100)
            messagebox.showinfo("Success", f"Processing complete! Output saved to:\n{os.path.abspath(output_file)}")

        except requests.exceptions.HTTPError as e:
            error_message = f"An API error occurred (Status code: {e.response.status_code})."
            try:
                error_info = e.response.json()
                error_detail = error_info.get("error", {}).get("message", "No additional details provided.")
                error_message += f"\n\nDetails: {error_detail}"
            except json.JSONDecodeError:
                error_message += f"\n\nResponse: {e.response.text}"
            messagebox.showerror("API Error", error_message)
            self.root.after(0, self.update_progress, "Failed", 0)
        except Exception as e:
            # Catching specific OpenAI errors if the library is imported
            try:
                from openai import APIError
                if isinstance(e, APIError):
                    messagebox.showerror("OpenAI API Error", f"An error occurred with the OpenAI API:\n{e}")
                else:
                    messagebox.showerror("Error", f"An unexpected error occurred:\n{e}")
            except ImportError:
                 messagebox.showerror("Error", f"An unexpected error occurred:\n{e}")
            self.root.after(0, self.update_progress, "Failed", 0)
        finally:
            # Step 5: Cleanup
            if 'images' in locals():
                self.cleanup_images(images)
            # Reset progress bar after a delay
            self.root.after(2000, self.update_progress, "Ready", 0)


if __name__ == "__main__":
    # To use Tesseract, you must have it installed and in your system's PATH.
    # Download from: https://github.com/tesseract-ocr/tesseract
    
    # To use Google Vision, you need to install the client library:
    # pip install google-cloud-vision google-auth
    
    # To use OpenAI, you need to install the client library:
    # pip install openai
    
    # Other dependencies:
    # pip install Pillow PyMuPDF requests
    
    root = tk.Tk()
    app = OcrLlmApp(root)
    root.mainloop()

