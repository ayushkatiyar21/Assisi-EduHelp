# API key has spaces to prevent it from revoking

import tkinter as tk
from tkinter import scrolledtext, messagebox
import groq
import threading
import itertools
import os

# Add PIL for image handling (for JPG/PNG)
from PIL import Image, ImageTk

class GroqChatbot:
    def __init__(self, api_key):
        try:
            self.client = groq.Groq(api_key=api_key)
            self.history = [{"role": "system", "content": "You are a helpful assistant."}]
        except Exception as e:
            messagebox.showerror("Initialization Error", f"Failed to initialize Groq client. Is the API key valid?\nError: {e}")
            self.client = None

    def send_message(self, message):
        if not self.client:
            return "Error: Groq client is not initialized."
        self.history.append({"role": "user", "content": message})
        try:
            chat_completion = self.client.chat.completions.create(
                messages=self.history,
                model="llama3-8b-8192",
            )
            bot_response_text = chat_completion.choices[0].message.content
            self.history.append({"role": "assistant", "content": bot_response_text})
            return bot_response_text
        except groq.AuthenticationError as e:
            error_message = f"Authentication Error: Invalid API key. Please check your key.\nDetails: {e}"
            print(error_message)
            self.history.pop()
            return error_message
        except groq.APIConnectionError as e:
            error_message = f"Connection Error: Could not connect to the API. Please check your network.\nDetails: {e.__cause__}"
            print(error_message)
            self.history.pop()
            return error_message
        except Exception as e:
            error_message = f"An unexpected error occurred: {e}"
            print(error_message)
            self.history.pop()
            return error_message

class ChatbotApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Assisi EduHelp")
        self.root.geometry("700x600")
        self.root.configure(bg="#f7fafc")  # Light background

        # --- Style Definitions (light/soft theme) ---
        self.font_style = ("Segoe UI", 13)
        self.font_style_bold = ("Segoe UI", 13, "bold")
        self.bg_color = "#f7fafc"              # Very light gray
        self.text_area_bg = "#fffefa"          # Off-white
        self.text_color = "#333333"            # Dark gray text
        self.button_bg = "#fbbf24"             # Soft yellow/orange
        self.button_fg = "#1e293b"             # Deep blue/gray
        self.button_active_bg = "#f59e42"      # Slightly deeper orange
        self.user_bubble_bg = "#e0e7ef"        # Soft pale blue
        self.bot_bubble_bg = "#fff7e6"         # Soft warm beige
        self.user_bubble_fg = "#1e293b"        # Deep blue/gray
        self.bot_bubble_fg = "#b45309"         # Soft orange-brown
        self.input_bg = "#f1f5f9"
        self.input_border = "#fbbf24"
        self.input_border_inactive = "#cbd5e1"

        # --- Header Bar with Logo ---
        header = tk.Frame(root, bg="#fef9c3", height=60)
        header.pack(fill=tk.X)

        # Load the logo image
        logo_path = os.path.join(os.path.dirname(__file__), "logo.png")
        if not os.path.exists(logo_path):
            logo_path = os.path.join(os.path.dirname(__file__), "logo.jpg")
        self.logo_img = None
        if os.path.exists(logo_path):
            try:
                img = Image.open(logo_path)
                img = img.resize((48, 48), Image.ANTIALIAS)
                self.logo_img = ImageTk.PhotoImage(img)
                logo_label = tk.Label(header, image=self.logo_img, bg="#fef9c3", bd=0)
                logo_label.pack(side=tk.LEFT, padx=(15, 5), pady=6)
            except Exception:
                self.logo_img = None  # fallback: no image loaded

        # Title text beside logo
        header_label = tk.Label(header, text="Assisi EduHelp", font=("Segoe UI", 18, "bold"), fg="#d97706", bg="#fef9c3")
        header_label.pack(side=tk.LEFT, padx=(6, 16), pady=10)

        # --- Main Frame ---
        main_frame = tk.Frame(root, bg=self.bg_color)
        main_frame.pack(padx=10, pady=(0,0), fill=tk.BOTH, expand=True)

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

        # --- Send Button ---
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
            messagebox.showerror("API Key Missing", "Please enter your Groq API key in the `self.api_key` variable in the code.")
            self.root.destroy()
            return

        self.chatbot = GroqChatbot(api_key=self.api_key)
        if self.chatbot.client:
            self.add_message("System", "Welcome to Assisi EduHelp! You can start chatting now.")
        else:
            self.root.destroy()

        self.thinking = False

    # --- Message Bubble Factory ---
    def _bubble(self, text, bg, fg):
        bubble = tk.Label(self.chat_area, text=text, bg=bg, fg=fg, wraplength=550,
                          justify=tk.LEFT, anchor="w", font=self.font_style, padx=12, pady=6, bd=0)
        return bubble

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
            self._thinking_label.destroy()
        self.chat_area.yview(tk.END)

    # --- Sending Message Event ---
    def send_message_event(self, event=None):
        user_input = self.msg_entry.get()
        if user_input.strip() == "" or self.chatbot is None:
            return
        self.add_message("User", user_input)
        self.msg_entry.delete(0, tk.END)
        self.msg_entry.config(state='disabled')
        self.send_button.config(state='disabled')
        self.animate_thinking()
        thread = threading.Thread(target=self.get_bot_response, args=(user_input,))
        thread.start()

    def get_bot_response(self, user_input):
        response = self.chatbot.send_message(user_input)
        self.root.after(0, self.update_chat_with_response, response)

    def update_chat_with_response(self, response):
        try:
            self.remove_thinking()
            self.add_message("Bot", response)
        finally:
            self.msg_entry.config(state='normal')
            self.send_button.config(state='normal')
            self.msg_entry.focus_set()

if __name__ == "__main__":
    try:
        root = tk.Tk()
        app = ChatbotApp(root)
        root.mainloop()
    except Exception as e:
        print(f"An error occurred while starting the application: {e}")