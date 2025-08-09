import tkinter as tk
from tkinter import filedialog, scrolledtext, messagebox
import openai
import base64
import threading
import os

# --- IMPORTANT SECURITY WARNING ---
# You should not hardcode your API key directly in the script like this.
# It's better to use environment variables or a secure key management system.
# For example: api_key = os.environ.get("OPENROUTER_API_KEY")
API_KEY = "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"

# --- OpenAI Client Setup for OpenRouter ---
try:
    client = openai.OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=API_KEY,
    )
except Exception as e:
    messagebox.showerror("API Error", f"Failed to initialize OpenAI client: {e}")
    client = None

class VisionChatApp:
    def __init__(self, root):
        """
        Initializes the main application window and its widgets.
        """
        self.root = root
        self.root.title("Vision Chatbot")
        self.root.geometry("600x700")
        self.root.configure(bg="#2c2f33")

        self.image_path = None

        # --- Main Frame ---
        main_frame = tk.Frame(root, bg="#2c2f33")
        main_frame.pack(pady=20, padx=20, fill="both", expand=True)

        # --- Text Prompt Entry ---
        prompt_label = tk.Label(main_frame, text="Your Question:", fg="white", bg="#2c2f33", font=("Helvetica", 12))
        prompt_label.pack(pady=(0, 5), anchor="w")
        self.prompt_entry = tk.Entry(main_frame, width=50, bg="#40444b", fg="white", insertbackground='white', font=("Helvetica", 12), relief="flat", borderwidth=4)
        self.prompt_entry.pack(fill="x")

        # --- Image Selection ---
        self.image_label = tk.Label(main_frame, text="No image selected.", fg="#99aab5", bg="#2c2f33", font=("Helvetica", 10))
        self.image_label.pack(pady=(10, 5), anchor="w")
        
        browse_button = tk.Button(main_frame, text="Upload Image", command=self.browse_image, bg="#7289da", fg="white", font=("Helvetica", 11, "bold"), relief="flat")
        browse_button.pack(pady=5, fill="x")

        # --- Submit Button ---
        submit_button = tk.Button(main_frame, text="Get Response", command=self.start_query_thread, bg="#57f287", fg="black", font=("Helvetica", 12, "bold"), relief="flat")
        submit_button.pack(pady=20, fill="x", ipady=5)

        # --- Response Display Area ---
        response_label = tk.Label(main_frame, text="AI Response:", fg="white", bg="#2c2f33", font=("Helvetica", 12))
        response_label.pack(pady=(10, 5), anchor="w")
        self.response_text = scrolledtext.ScrolledText(main_frame, wrap=tk.WORD, height=15, bg="#40444b", fg="white", font=("Helvetica", 12), relief="flat", borderwidth=4)
        self.response_text.pack(fill="both", expand=True)
        self.response_text.configure(state='disabled') # Make it read-only

    def browse_image(self):
        """
        Opens a file dialog to select an image and updates the label.
        """
        # File dialog to select an image file
        file_path = filedialog.askopenfilename(
            title="Select an Image",
            filetypes=[("Image Files", "*.png *.jpg *.jpeg *.webp *.gif")]
        )
        if file_path:
            self.image_path = file_path
            # Display the name of the selected file
            self.image_label.config(text=f"Selected: {os.path.basename(file_path)}", fg="white")
        else:
            self.image_path = None
            self.image_label.config(text="No image selected.", fg="#99aab5")

    def encode_image_to_base64(self, path):
        """
        Encodes the image file at the given path to a Base64 string.
        """
        try:
            with open(path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            messagebox.showerror("File Error", f"Could not read or encode the image: {e}")
            return None

    def start_query_thread(self):
        """
        Starts a new thread to run the API query to prevent the GUI from freezing.
        """
        # Run the API call in a separate thread
        query_thread = threading.Thread(target=self.submit_query)
        query_thread.daemon = True # Allows main program to exit even if thread is running
        query_thread.start()

    def submit_query(self):
        """
        Prepares and sends the request to the OpenRouter API and displays the response.
        This function is executed in a separate thread.
        """
        if not client:
            messagebox.showerror("API Error", "OpenAI client is not configured.")
            return
            
        prompt = self.prompt_entry.get()
        if not prompt:
            messagebox.showwarning("Input Error", "Please enter a question.")
            return

        # Show a loading message
        self.update_response_text("Getting response, please wait...")

        messages = [{"role": "user", "content": []}]
        
        # Add the text part to the message
        messages[0]["content"].append({
            "type": "text",
            "text": prompt
        })

        # If an image is selected, encode and add it to the message
        if self.image_path:
            base64_image = self.encode_image_to_base64(self.image_path)
            if base64_image:
                messages[0]["content"].append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                })
        
        try:
            # --- API Call ---
            completion = client.chat.completions.create(
                model="mistralai/mistral-small-3.2-24b-instruct:free",
                messages=messages,
                max_tokens=1024, # Limit the response length
            )
            response_content = completion.choices[0].message.content
            self.update_response_text(response_content)

        except openai.APIStatusError as e:
            error_message = f"API Error: {e.status_code}\n{e.response.text}"
            messagebox.showerror("API Error", error_message)
            self.update_response_text("Failed to get a response. Check the error message.")
        except Exception as e:
            messagebox.showerror("Error", f"An unexpected error occurred: {e}")
            self.update_response_text("An error occurred.")

    def update_response_text(self, text):
        """
        Updates the response text area in a thread-safe way.
        """
        self.response_text.configure(state='normal') # Enable writing
        self.response_text.delete(1.0, tk.END)
        self.response_text.insert(tk.END, text)
        self.response_text.configure(state='disabled') # Disable writing

if __name__ == "__main__":
    root = tk.Tk()
    app = VisionChatApp(root)
    root.mainloop()
