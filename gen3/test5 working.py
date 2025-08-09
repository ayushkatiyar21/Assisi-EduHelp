import tkinter as tk
from tkinter import filedialog, scrolledtext, messagebox
import openai
import base64
import threading
import os
from PIL import Image, ImageTk

# --- IMPORTANT SECURITY WARNING ---
# It's recommended to use environment variables for your API key.
# For example: api_key = os.environ.get("OPENROUTER_API_KEY")
# Using the API Key and model from Code 1 as requested.
API_KEY = "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"
API_MODEL = "google/gemma-3-27b-it:free"

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
        Initializes the application, combining the GUI from Code 2
        with the vision model functionality from Code 1.
        """
        self.root = root
        self.root.title("Vision Chatbot")
        self.root.geometry("700x650") # Adjusted height for the new layout
        self.root.configure(bg="#f7fafc")

        self.image_path = None
        self.thinking = False

        # --- Style Definitions (from Code 2) ---
        self.font_style = ("Segoe UI", 13)
        self.font_style_bold = ("Segoe UI", 13, "bold")
        self.bg_color = "#f7fafc"
        self.text_area_bg = "#fffefa"
        self.text_color = "#333333"
        self.button_bg = "#fbbf24"
        self.button_fg = "#1e293b"
        self.button_active_bg = "#f59e42"
        self.user_bubble_bg = "#e0e7ef"
        self.bot_bubble_bg = "#fff7e6"
        self.user_bubble_fg = "#1e293b"
        self.bot_bubble_fg = "#b45309"
        self.input_bg = "#f1f5f9"
        self.input_border = "#fbbf24"
        self.input_border_inactive = "#cbd5e1"

        # --- Header Bar with Logo (from Code 2) ---
        header = tk.Frame(root, bg="#fef9c3", height=60)
        header.pack(fill=tk.X)
        
        # NOTE: You need a 'logo.png' or 'logo.jpg' file in the same directory for the logo to appear.
        logo_path = os.path.join(os.path.dirname(__file__), "logo.png")
        if not os.path.exists(logo_path):
            logo_path = os.path.join(os.path.dirname(__file__), "logo.jpg")

        self.logo_img = None
        if os.path.exists(logo_path):
            try:
                img = Image.open(logo_path)
                img = img.resize((48, 48), Image.Resampling.LANCZOS)
                self.logo_img = ImageTk.PhotoImage(img)
                logo_label = tk.Label(header, image=self.logo_img, bg="#fef9c3", bd=0)
                logo_label.pack(side=tk.LEFT, padx=(15, 5), pady=6)
            except Exception as e:
                print(f"Could not load logo: {e}")
                self.logo_img = None

        header_label = tk.Label(header, text="Vision Chat", font=("Segoe UI", 18, "bold"), fg="#d97706", bg="#fef9c3")
        header_label.pack(side=tk.LEFT, padx=(6, 16), pady=10)

        # --- Chat Area (from Code 2) ---
        chat_frame = tk.Frame(root, bg=self.bg_color)
        chat_frame.pack(padx=10, pady=0, fill=tk.BOTH, expand=True)
        self.chat_area = scrolledtext.ScrolledText(chat_frame, wrap=tk.WORD, bg=self.text_area_bg, state='disabled',
                                                   font=self.font_style, padx=10, pady=10, bd=0, highlightthickness=0)
        self.chat_area.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        self.chat_area.tag_config('user', foreground="#64748b", font=self.font_style_bold)
        self.chat_area.tag_config('bot', foreground="#d97706", font=self.font_style_bold)

        # --- Input Frame (modified to include image upload) ---
        input_frame = tk.Frame(root, bg=self.bg_color)
        input_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=24, pady=16)

        self.image_status_label = tk.Label(input_frame, text="No image selected.", fg="#64748b", bg=self.bg_color, font=("Segoe UI", 10))
        self.image_status_label.pack(fill=tk.X, pady=(0, 5))

        self.msg_entry = tk.Entry(input_frame, font=self.font_style, bg=self.input_bg, fg=self.text_color,
                                  highlightthickness=2, highlightcolor=self.input_border,
                                  insertbackground=self.text_color, bd=1, relief=tk.FLAT)
        self.msg_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=8, padx=(0, 4))
        self.msg_entry.bind("<Return>", self.start_query_thread)

        self.upload_button = tk.Button(input_frame, text="Upload", font=self.font_style_bold, bg="#7289da", fg="white",
                                      activebackground="#5b6eae", activeforeground="white", command=self.browse_image,
                                      bd=0, padx=15, pady=8, relief=tk.FLAT, height=1)
        self.upload_button.pack(side=tk.LEFT, padx=(5, 5), ipady=2)

        self.send_button = tk.Button(input_frame, text="Send", font=self.font_style_bold, bg=self.button_bg, fg=self.button_fg,
                                     activebackground=self.button_active_bg, activeforeground=self.button_fg,
                                     command=self.start_query_thread, bd=0, padx=20, pady=8, relief=tk.FLAT, height=1)
        self.send_button.pack(side=tk.RIGHT, padx=(5, 0), ipady=2)
        
        # --- Initial Message ---
        self.add_message("Bot", "Welcome! Please ask a question. You can also upload an image to discuss.")

    # --- Methods from Code 1 (Functionality) ---
    
    def browse_image(self):
        """Opens a file dialog to select an image and updates the label."""
        file_path = filedialog.askopenfilename(
            title="Select an Image",
            filetypes=[("Image Files", "*.png *.jpg *.jpeg *.webp *.gif")]
        )
        if file_path:
            self.image_path = file_path
            self.image_status_label.config(text=f"Selected: {os.path.basename(file_path)}", fg=self.bot_bubble_fg)
        else:
            self.image_path = None
            self.image_status_label.config(text="No image selected.", fg="#64748b")

    def encode_image_to_base64(self, path):
        """Encodes the image file at the given path to a Base64 string."""
        try:
            with open(path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            messagebox.showerror("File Error", f"Could not read or encode the image: {e}")
            return None

    def start_query_thread(self, event=None):
        """Starts a new thread to run the API query to prevent the GUI from freezing."""
        user_input = self.msg_entry.get()
        if user_input.strip() == "" and not self.image_path:
            messagebox.showwarning("Input Error", "Please enter a question or upload an image.")
            return

        self.add_message("User", user_input if user_input.strip() else "[Image Query]")
        self.msg_entry.delete(0, tk.END)
        self.msg_entry.config(state='disabled')
        self.send_button.config(state='disabled')
        self.upload_button.config(state='disabled')

        self.animate_thinking()
        
        query_thread = threading.Thread(target=self.submit_query, args=(user_input,))
        query_thread.daemon = True
        query_thread.start()

    def submit_query(self, prompt):
        """Prepares and sends the request to the OpenRouter API and displays the response."""
        if not client:
            self.root.after(0, self.update_chat_with_response, "Error: OpenAI client not configured.")
            return

        messages = [{"role": "user", "content": []}]
        
        # Default prompt if only an image is provided
        final_prompt = prompt if prompt.strip() else "What is in this image?"
        
        messages[0]["content"].append({"type": "text", "text": final_prompt})

        # Encode and add the image if one is selected
        image_to_process = self.image_path
        if image_to_process:
            base64_image = self.encode_image_to_base64(image_to_process)
            if base64_image:
                messages[0]["content"].append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                })
            else:
                self.root.after(0, self.update_chat_with_response, "Error: Could not encode image.")
                return

        try:
            completion = client.chat.completions.create(
                model=API_MODEL,
                messages=messages,
                max_tokens=1024,
            )
            response_content = completion.choices[0].message.content
            self.root.after(0, self.update_chat_with_response, response_content)

        except openai.APIStatusError as e:
            error_message = f"API Error: {e.status_code}\n{e.response.text}"
            self.root.after(0, self.update_chat_with_response, error_message)
        except Exception as e:
            error_message = f"An unexpected error occurred: {e}"
            self.root.after(0, self.update_chat_with_response, error_message)

    # --- Methods from Code 2 (GUI Management) ---

    def _bubble(self, text, bg, fg):
        """Creates a styled label to act as a chat bubble."""
        bubble = tk.Label(self.chat_area, text=text, bg=bg, fg=fg, wraplength=550,
                          justify=tk.LEFT, anchor="w", font=self.font_style, padx=12, pady=6, bd=0)
        return bubble

    def add_message(self, sender, message):
        """Adds a message to the chat area with bubble styling."""
        self.chat_area.configure(state='normal')
        self.chat_area.insert(tk.END, "\n")
        if sender.lower() == 'user':
            self.chat_area.insert(tk.END, "You:\n", ('user',))
            self.chat_area.window_create(tk.END, window=self._bubble(message, self.user_bubble_bg, self.user_bubble_fg))
        else: # 'Bot' or 'System'
            self.chat_area.insert(tk.END, "Bot:\n", ('bot',))
            self.chat_area.window_create(tk.END, window=self._bubble(message, self.bot_bubble_bg, self.bot_bubble_fg))
        
        self.chat_area.insert(tk.END, "\n")
        self.chat_area.configure(state='disabled')
        self.chat_area.yview(tk.END)

    def animate_thinking(self):
        """Displays a 'Thinking...' animation in the chat."""
        self.thinking = True
        self.chat_area.configure(state='normal')
        self.chat_area.insert(tk.END, "\n")
        self._thinking_label = tk.Label(self.chat_area, text="Bot is thinking...", font=self.font_style, fg=self.bot_bubble_fg, bg=self.text_area_bg)
        self.chat_area.window_create(tk.END, window=self._thinking_label)
        self.chat_area.insert(tk.END, "\n")
        self.chat_area.configure(state='disabled')
        self.chat_area.yview(tk.END)

    def remove_thinking(self):
        """Removes the 'Thinking...' animation."""
        if hasattr(self, "_thinking_label") and self._thinking_label.winfo_exists():
            self._thinking_label.destroy()
        self.thinking = False

    def update_chat_with_response(self, response):
        """Callback to update the GUI with the bot's response."""
        try:
            self.remove_thinking()
            self.add_message("Bot", response)
            # Reset image path for the next query
            self.image_path = None
            self.image_status_label.config(text="No image selected.", fg="#64748b")
        finally:
            # Re-enable input fields
            self.msg_entry.config(state='normal')
            self.send_button.config(state='normal')
            self.upload_button.config(state='normal')
            self.msg_entry.focus_set()

if __name__ == "__main__":
    if not client:
        print("Exiting: OpenAI client could not be initialized.")
    else:
        try:
            root = tk.Tk()
            app = VisionChatApp(root)
            root.mainloop()
        except Exception as e:
            print(f"An error occurred while starting the application: {e}")