# API key has spaces to prevent it from revoking

import tkinter as tk
from tkinter import scrolledtext, messagebox, filedialog, simpledialog
import groq
import threading
import itertools
import os
import base64
import sys
import traceback
import logging
import platform

# Add PIL for image handling (for JPG/PNG)
from PIL import Image, ImageTk, ImageOps


# Configure a simple logger to capture crashes and diagnostics
logging.basicConfig(
    filename=os.path.join(os.path.dirname(__file__), "app.log"),
    filemode="a",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)


def _guess_mime_type(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    return {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
    }.get(ext, "image/png")


class GroqChatbot:
    """
    - Text-only chat
    - Vision: image + text prompt to a vision-capable model
    """
    def __init__(self, api_key, text_model="llama3-8b-8192", vision_model="llama-3.2-11b-vision-preview"):
        try:
            self.client = groq.Groq(api_key=api_key)
            self.history = [{"role": "system", "content": "You are a helpful assistant."}]
            self.text_model = text_model
            self.vision_model = vision_model
        except Exception as e:
            logging.exception("Failed to initialize Groq client")
            try:
                messagebox.showerror(
                    "Initialization Error",
                    f"Failed to initialize Groq client. Is the API key valid?\nError: {e}",
                )
            except Exception:
                pass
            self.client = None

    def send_message(self, message: str) -> str:
        if not self.client:
            return "Error: Groq client is not initialized."
        self.history.append({"role": "user", "content": message})
        try:
            chat_completion = self.client.chat.completions.create(
                messages=self.history,
                model=self.text_model,
            )
            bot_response_text = chat_completion.choices[0].message.content
            self.history.append({"role": "assistant", "content": bot_response_text})
            return bot_response_text
        except groq.AuthenticationError as e:
            logging.exception("Authentication error")
            self.history.pop()
            return f"Authentication Error: Invalid API key. Please check your key.\nDetails: {e}"
        except groq.APIConnectionError as e:
            logging.exception("API connection error")
            self.history.pop()
            return f"Connection Error: Could not connect to the API. Please check your network.\nDetails: {e.__cause__}"
        except Exception as e:
            logging.exception("Unexpected error in send_message")
            self.history.pop()
            return f"An unexpected error occurred: {e}"

    def send_image_message(self, image_b64: str, mime_type: str, prompt: str) -> str:
        if not self.client:
            return "Error: Groq client is not initialized."

        user_placeholder = f"[User sent an image ({mime_type})] Prompt: {prompt}"
        self.history.append({"role": "user", "content": user_placeholder})

        multimodal_user = {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt or "Describe this image."},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime_type};base64,{image_b64}"},
                },
            ],
        }

        try:
            chat_completion = self.client.chat.completions.create(
                messages=self.history + [multimodal_user],
                model=self.vision_model,
            )
            bot_response_text = chat_completion.choices[0].message.content
            self.history.append({"role": "assistant", "content": bot_response_text})
            return bot_response_text
        except groq.AuthenticationError as e:
            logging.exception("Authentication error (vision)")
            self.history.pop()
            return f"Authentication Error: Invalid API key. Please check your key.\nDetails: {e}"
        except groq.APIConnectionError as e:
            logging.exception("API connection error (vision)")
            self.history.pop()
            return f"Connection Error: Could not connect to the API. Please check your network.\nDetails: {e.__cause__}"
        except Exception as e:
            logging.exception("Unexpected error in send_image_message")
            self.history.pop()
            hint = ""
            if "model" in str(e).lower():
                hint = "\nHint: The selected vision model may not be available for your account. Try another, e.g. 'llava-v1.5-7b-4096' or 'llama-3.2-90b-vision-preview'."
            return f"An unexpected error occurred: {e}{hint}"


class ChatbotApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Assisi EduHelp")
        self.root.geometry("700x700")
        self.root.configure(bg="#f7fafc")  # Light background

        # Keep references to images displayed in the chat (Tk requires this)
        self._image_refs = []

        # --- Style Definitions (light/soft theme) ---
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

        # --- Header Bar with Logo ---
        header = tk.Frame(root, bg="#fef9c3", height=60)
        header.pack(fill=tk.X)

        # Load the logo image
        try:
            logo_path = os.path.join(os.path.dirname(__file__), "logo.png")
            if not os.path.exists(logo_path):
                logo_path = os.path.join(os.path.dirname(__file__), "logo.jpg")
        except Exception:
            logo_path = ""

        self.logo_img = None
        if logo_path and os.path.exists(logo_path):
            try:
                img = Image.open(logo_path)
                img = img.resize((48, 48), Image.LANCZOS)
                self.logo_img = ImageTk.PhotoImage(img)
                logo_label = tk.Label(header, image=self.logo_img, bg="#fef9c3", bd=0)
                logo_label.pack(side=tk.LEFT, padx=(15, 5), pady=6)
            except Exception as e:
                logging.info(f"Logo load failed: {e}")
                self.logo_img = None

        header_label = tk.Label(header, text="Assisi EduHelp", font=("Segoe UI", 18, "bold"), fg="#d97706", bg="#fef9c3")
        header_label.pack(side=tk.LEFT, padx=(6, 16), pady=10)

        # --- Main Frame ---
        main_frame = tk.Frame(root, bg=self.bg_color)
        main_frame.pack(padx=10, pady=(0, 0), fill=tk.BOTH, expand=True)

        # --- Chat Area ---
        self.chat_area = scrolledtext.ScrolledText(
            main_frame,
            wrap=tk.WORD,
            bg=self.text_area_bg,
            fg=self.text_color,
            font=self.font_style,
            state='disabled',
            padx=10, pady=10,
            bd=0, highlightthickness=0
        )
        self.chat_area.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        self.chat_area.tag_config('user', foreground="#64748b", font=self.font_style_bold)
        self.chat_area.tag_config('bot', foreground="#d97706", font=self.font_style_bold)
        self.chat_area.tag_config('text', foreground=self.text_color, font=self.font_style)

        # --- Input Frame outside main_frame, at window bottom ---
        input_frame = tk.Frame(root, bg=self.bg_color)
        input_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=24, pady=16)

        # --- Message Entry ---
        self.msg_entry = tk.Entry(input_frame, font=self.font_style, bg=self.input_bg, fg=self.text_color,
                                  highlightthickness=2, highlightcolor=self.input_border,
                                  insertbackground=self.text_color, bd=1, relief=tk.FLAT)
        self.msg_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=8, padx=(0, 4))
        self.msg_entry.bind("<FocusIn>", lambda e: self.msg_entry.config(highlightbackground=self.input_border))
        self.msg_entry.bind("<FocusOut>", lambda e: self.msg_entry.config(highlightbackground=self.input_border_inactive))
        self.msg_entry.bind("<Return>", self.send_message_event)
        self.msg_entry.bind("<Control-Return>", self.send_message_event)
        self.msg_entry.bind("<Control-Enter>", self.send_message_event)

        # --- Buttons on the right: Image + Send ---
        self.image_button = tk.Button(
            input_frame,
            text="Image",
            font=self.font_style_bold,
            bg="#fde68a", fg=self.button_fg,
            activebackground="#fcd34d",
            activeforeground=self.button_fg,
            command=self.attach_image_event,
            bd=0, padx=16, pady=8, relief=tk.FLAT, height=1,
            highlightthickness=2, highlightbackground="#fde68a"
        )
        self.image_button.pack(side=tk.RIGHT, padx=(10, 0), ipady=2)
        self.image_button.bind("<Enter>", lambda e: self.image_button.config(bg="#fcd34d"))
        self.image_button.bind("<Leave>", lambda e: self.image_button.config(bg="#fde68a"))

        self.send_button = tk.Button(
            input_frame,
            text="Send",
            font=self.font_style_bold,
            bg=self.button_bg, fg=self.button_fg,
            activebackground=self.button_active_bg,
            activeforeground=self.button_fg,
            command=self.send_message_event,
            bd=0, padx=20, pady=8, relief=tk.FLAT, height=1,
            highlightthickness=2, highlightbackground=self.button_bg
        )
        self.send_button.pack(side=tk.RIGHT, padx=(10, 0), ipady=2)
        self.send_button.bind("<Enter>", lambda e: self.send_button.config(bg=self.button_active_bg))
        self.send_button.bind("<Leave>", lambda e: self.send_button.config(bg=self.button_bg))

        # --- Initialization Logic ---
        self.api_key = "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"
        if not self.api_key or self.api_key == "":
            try:
                messagebox.showerror("API Key Missing", "Please enter your Groq API key in the `self.api_key` variable in the code.")
            except Exception:
                pass
            self.root.destroy()
            return

        self.chatbot = GroqChatbot(
            api_key=self.api_key,
            text_model="llama3-8b-8192",
            vision_model="llama-3.2-11b-vision-preview"
        )
        if self.chatbot.client:
            self.add_message("System", "Welcome to Assisi EduHelp! You can start chatting now.\nTip: Click 'Image' to analyze a picture.")
        else:
            self.root.destroy()
            return

        self.thinking = False

    # --- Utility: create thumbnail with correct orientation ---
    def _make_thumbnail(self, path: str, max_w=520, max_h=360) -> Image.Image:
        img = Image.open(path)
        img = ImageOps.exif_transpose(img)  # Fix orientation from EXIF
        img.thumbnail((max_w, max_h), Image.LANCZOS)
        return img

    # --- Message Bubble Factory ---
    def _bubble(self, text, bg, fg):
        bubble = tk.Label(self.chat_area, text=text, bg=bg, fg=fg, wraplength=550,
                          justify=tk.LEFT, anchor="w", font=self.font_style, padx=12, pady=6, bd=0)
        return bubble

    # --- Add Image Bubble ---
    def add_image(self, sender: str, pil_image: Image.Image):
        photo = ImageTk.PhotoImage(pil_image)
        self._image_refs.append(photo)  # prevent GC

        self.chat_area.configure(state='normal')
        self.chat_area.insert(tk.END, "\n")
        if sender.lower() == 'user':
            self.chat_area.insert(tk.END, "You:\n", ('user',))
            lbl = tk.Label(self.chat_area, image=photo, bg=self.user_bubble_bg, bd=0)
        else:
            self.chat_area.insert(tk.END, "Bot:\n", ('bot',))
            lbl = tk.Label(self.chat_area, image=photo, bg=self.bot_bubble_bg, bd=0)

        self.chat_area.window_create(tk.END, window=lbl)
        self.chat_area.insert(tk.END, "\n")
        self.chat_area.configure(state='disabled')
        self.chat_area.yview(tk.END)

    # --- Add Message with Bubbles and Spacing ---
    def add_message(self, sender, message):
        self.chat_area.configure(state='normal')
        self.chat_area.insert(tk.END, "\n")
        if sender.lower() == 'user':
            self.chat_area.insert(tk.END, "You:\n", ('user',))
            self.chat_area.window_create(tk.END, window=self._bubble(message, self.user_bubble_bg, self.user_bubble_fg))
        elif sender.lower() == 'bot':
            self.chat_area.insert(tk.END, "Bot:\n", ('bot',))
            self.chat_area.window_create(tk.END, window=self._bubble(message, self.bot_bubble_bg, self.bot_bubble_fg))
        else:
            self.chat_area.insert(tk.END, f"{message}\n\n", ('text',))
        self.chat_area.insert(tk.END, "\n")
        self.chat_area.configure(state='disabled')
        self.chat_area.yview(tk.END)

    # --- Animated "Thinking..." Indicator ---
    def animate_thinking(self):
        self.thinking = True
        self._thinking_cycle = itertools.cycle(["Thinking.", "Thinking..", "Thinking..."])
        self._thinking_label = tk.Label(self.chat_area, text="Thinking.", font=self.font_style, fg=self.bot_bubble_fg, bg=self.text_area_bg)
        self.chat_area.configure(state='normal')
        self.chat_area.insert(tk.END, "\n")
        self.chat_area.window_create(tk.END, window=self._thinking_label)
        self.chat_area.insert(tk.END, "\n")
        self.chat_area.configure(state='disabled')
        self.chat_area.yview(tk.END)
        self._animate_thinking_step()

    def _animate_thinking_step(self):
        if getattr(self, "thinking", False):
            self._thinking_label.config(text=next(self._thinking_cycle))
            self.root.after(500, self._animate_thinking_step)

    def remove_thinking(self):
        self.thinking = False
        if hasattr(self, "_thinking_label"):
            try:
                self._thinking_label.destroy()
            except Exception:
                pass
        self.chat_area.yview(tk.END)

    # --- Sending Text Message Event ---
    def send_message_event(self, event=None):
        try:
            user_input = self.msg_entry.get()
            if user_input.strip() == "" or self.chatbot is None:
                return
            self.add_message("User", user_input)
            self.msg_entry.delete(0, tk.END)
            self.msg_entry.config(state='disabled')
            self.send_button.config(state='disabled')
            self.image_button.config(state='disabled')
            self.animate_thinking()
            thread = threading.Thread(target=self.get_bot_response, args=(user_input,), daemon=True)
            thread.start()
        except Exception:
            logging.exception("send_message_event failed")

    def get_bot_response(self, user_input):
        try:
            response = self.chatbot.send_message(user_input)
        except Exception:
            logging.exception("get_bot_response crashed")
            response = "An unexpected error occurred while getting the response."
        self.root.after(0, self.update_chat_with_response, response)

    # --- Safe file chooser with macOS-friendly fallback ---
    def _choose_image_file(self) -> str | None:
        home = os.path.expanduser("~")
        path = None
        try:
            # Use a stable initial directory; some macOS Tk builds crash with weird CWDs
            path = filedialog.askopenfilename(
                parent=self.root,
                title="Select an image",
                initialdir=home,
                filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.webp;*.bmp"), ("All Files", "*.*")]
            )
        except Exception as e:
            logging.exception("askopenfilename failed")

        if path:
            return path

        # Fallback: manual path entry to avoid native dialog issues
        try:
            manual = simpledialog.askstring(
                "Enter Image Path",
                "Could not open the file dialog.\nPlease paste the full path to an image (PNG/JPG/WEBP/BMP):",
                parent=self.root,
            )
        except Exception:
            logging.exception("askstring fallback failed")
            manual = None

        if manual:
            manual = manual.strip()
            if os.path.isfile(manual):
                return manual
            try:
                messagebox.showerror("Invalid Path", "That path does not exist or is not a file.")
            except Exception:
                pass
        return None

    # --- Image Attach + Vision Flow ---
    def attach_image_event(self):
        if self.chatbot is None:
            return
        try:
            self.root.update_idletasks()

            path = self._choose_image_file()
            if not path:
                return

            try:
                prompt = simpledialog.askstring(
                    "Image Prompt",
                    "What should I look for in this image? (Leave empty to 'Describe this image.')",
                    parent=self.root
                )
            except Exception:
                logging.exception("askstring for image prompt failed")
                prompt = ""

            if prompt is None:
                return
            prompt = prompt.strip() or "Describe this image."

            # Show file name + prompt and image preview
            self.add_message("User", f"Image: {os.path.basename(path)}\n{prompt}")
            try:
                thumb = self._make_thumbnail(path)
                self.add_image("User", thumb)
            except Exception as e:
                logging.exception("Failed to preview image")
                self.add_message("System", f"Failed to preview image: {e}")

            # Disable inputs during request
            self.msg_entry.config(state='disabled')
            self.send_button.config(state='disabled')
            self.image_button.config(state='disabled')
            self.animate_thinking()

            # Process in background
            thread = threading.Thread(target=self.get_bot_image_response, args=(path, prompt), daemon=True)
            thread.start()
        except Exception:
            logging.exception("attach_image_event failed")
            try:
                messagebox.showerror("Error", "Failed to open the image dialog. Please try again.")
            except Exception:
                pass

    def get_bot_image_response(self, path: str, prompt: str):
        try:
            with open(path, "rb") as f:
                data = f.read()
            mime = _guess_mime_type(path)
            b64 = base64.b64encode(data).decode("utf-8")
            response = self.chatbot.send_image_message(b64, mime, prompt)
        except Exception:
            logging.exception("get_bot_image_response crashed")
            response = "Failed to process image. Please try another file."
        self.root.after(0, self.update_chat_with_response, response)

    def update_chat_with_response(self, response):
        try:
            self.remove_thinking()
            self.add_message("Bot", response)
        except Exception:
            logging.exception("update_chat_with_response failed")
        finally:
            try:
                self.msg_entry.config(state='normal')
                self.send_button.config(state='normal')
                self.image_button.config(state='normal')
                self.msg_entry.focus_set()
            except Exception:
                pass


if __name__ == "__main__":
    try:
        os.environ.setdefault("TK_SILENCE_DEPRECATION", "1")

        root = tk.Tk()
        app = ChatbotApp(root)

        try:
            import tkinter
            logging.info(f"Tk version: {tkinter.TkVersion}, Tcl version: {tkinter.TclVersion}, Platform: {platform.platform()}")
        except Exception:
            pass

        root.mainloop()
    except Exception as e:
        logging.exception("Application crashed at top-level")
        print(f"An error occurred while starting the application: {e}", file=sys.stderr)
        traceback.print_exc()