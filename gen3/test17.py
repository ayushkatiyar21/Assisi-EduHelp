import os
import base64
import threading
from tkinter import filedialog, messagebox

# 3rd party
import customtkinter as ctk
from PIL import Image, ImageOps

# For pretty math display
import sympy as sp

# OpenRouter via OpenAI SDK
import openai
from openai import OpenAI

# --- Configuration ---
API_KEY = "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"
API_MODEL = "google/gemma-3-27b-it:free"
BASE_URL = "https://openrouter.ai/api/v1"

UI_SCALE = 1.35  # UI scaling factor
RESAMPLE_BEST = getattr(Image, "LANCZOS", Image.BICUBIC)

class VisionChatApp:
    def __init__(self, root: ctk.CTk):
        self.root = root
        self.root.title("Assisi EduHelp")
        self.root.geometry("920x780")
        self.root.minsize(720, 600)

        ctk.set_appearance_mode("Light")
        ctk.set_default_color_theme("blue")
        ctk.set_window_scaling(UI_SCALE)
        ctk.set_widget_scaling(UI_SCALE)

        self.colors = {
            "bg": "#F5F7FB",
            "surface": "#FFFFFF",
            "surface_alt": "#F2F5FA",
            "border": "#E5E7EB",
            "primary": "#2563EB",
            "primary_hover": "#1D4ED8",
            "muted": "#6B7280",
            "user_bubble": "#E8F1FF",
            "user_border": "#CFE3FF",
            "user_text": "#0F172A",
            "bot_bubble": "#FFF6E5",
            "bot_border": "#FDE6BF",
            "bot_text": "#1F2937",
            "system_bubble": "#EEF2FF",
            "system_border": "#DDE3FF",
            "system_text": "#111827",
        }

        # State
        self.image_path = None
        self.image_preview = None
        self.thinking = False
        self._streaming = False
        self._stream_buffer = ""
        self._cursor_visible = True
        self._cursor_glyph = "│"
        self._stream_label = None
        self._img_cache: list[ctk.CTkImage] = []
        self.client: OpenAI | None = None
        self.wraplength = int(720 * UI_SCALE)
        self.placeholder_image = self._make_placeholder_image((96, 96))

        self.root.configure(fg_color=self.colors["bg"])
        self._build_header()
        self._build_body()
        self._build_input_card()
        self.add_message(
            "Bot",
            "Hello! I’m Assisi EduHelp—your study companion. Type a question or attach a photo of your problem, and I’ll guide you step by step."
        )
        self._init_client()

    def _build_header(self):
        self.header = ctk.CTkFrame(
            self.root,
            fg_color=self.colors["surface"],
            corner_radius=0,
            border_width=0,
        )
        self.header.pack(fill="x")

        container = ctk.CTkFrame(self.header, fg_color="transparent")
        container.pack(fill="x", padx=16, pady=10)

        left = ctk.CTkFrame(container, fg_color="transparent")
        left.pack(side="left")

        logo_path = os.path.join(os.path.dirname(__file__), "logo.png")
        if not os.path.exists(logo_path):
            alt_logo = os.path.join(os.path.dirname(__file__), "logo.jpg")
            logo_path = alt_logo if os.path.exists(alt_logo) else None

        if logo_path:
            try:
                logo_img = Image.open(logo_path)
                logo_target_size = (int(350 * UI_SCALE), int(102 * UI_SCALE))
                logo_img = ImageOps.contain(logo_img, logo_target_size, method=RESAMPLE_BEST)
                logo_logical_size = (max(1, int(logo_img.width / UI_SCALE)), max(1, int(logo_img.height / UI_SCALE)))
                self.logo_ctk_img = ctk.CTkImage(light_image=logo_img.copy(), dark_image=logo_img.copy(), size=logo_logical_size)
                ctk.CTkLabel(left, image=self.logo_ctk_img, text="").pack(side="left", padx=(2, 10))
            except Exception:
                self.logo_ctk_img = None

        divider = ctk.CTkFrame(self.root, height=1, fg_color=self.colors["border"], corner_radius=0)
        divider.pack(fill="x")

    def _build_body(self):
        self.chat_card = ctk.CTkFrame(
            self.root,
            fg_color=self.colors["surface"],
            corner_radius=16,
            border_width=1,
            border_color=self.colors["border"],
        )
        self.chat_card.pack(fill="both", expand=True, padx=16, pady=(12, 8))

        self.chat_scroll = ctk.CTkScrollableFrame(
            self.chat_card,
            fg_color="transparent",
        )
        self.chat_scroll.pack(fill="both", expand=True, padx=12, pady=12)

        self._message_widgets = []

    def _build_input_card(self):
        self.input_card = ctk.CTkFrame(
            self.root,
            fg_color=self.colors["surface"],
            corner_radius=16,
            border_width=1,
            border_color=self.colors["border"],
        )
        self.input_card.pack(fill="x", padx=16, pady=(0, 16))

        self.preview_row = ctk.CTkFrame(self.input_card, fg_color="transparent")
        preview_box = ctk.CTkFrame(
            self.preview_row,
            fg_color=self.colors["surface_alt"],
            corner_radius=12,
            border_width=1,
            border_color=self.colors["border"],
            width=96,
            height=96,
        )
        preview_box.pack_propagate(False)
        preview_box.pack(side="left", padx=(12, 10), pady=(12, 6))

        self.preview_label = ctk.CTkLabel(preview_box, text="", text_color=self.colors["muted"])
        self.preview_label.pack(expand=True)

        self.image_status = ctk.CTkLabel(
            self.preview_row,
            text="No image selected.",
            anchor="w",
            text_color=self.colors["muted"],
        )
        self.image_status.pack(side="left", fill="x", expand=True, padx=(2, 0), pady=(12, 6))

        bottom_row = ctk.CTkFrame(self.input_card, fg_color="transparent")
        bottom_row.pack(fill="x", padx=12, pady=(6, 12))

        self.msg_entry = ctk.CTkEntry(
            bottom_row,
            placeholder_text="Type your message...",
            height=46,
            corner_radius=12,
            border_width=1,
            border_color=self.colors["border"],
        )
        self.msg_entry.pack(side="left", fill="x", expand=True, padx=(0, 8))
        self.msg_entry.bind("<Return>", self.start_query_thread)

        self.upload_button = ctk.CTkButton(
            bottom_row,
            text="Upload",
            fg_color=self.colors["primary"],
            hover_color=self.colors["primary_hover"],
            corner_radius=10,
            command=self.browse_image,
            width=110,
        )
        self.upload_button.pack(side="left", padx=(0, 8))

        self.send_button = ctk.CTkButton(
            bottom_row,
            text="Send",
            fg_color=self.colors["primary"],
            hover_color=self.colors["primary_hover"],
            corner_radius=10,
            command=self.start_query_thread,
            width=110,
        )
        self.send_button.pack(side="left")

        self.progress_row = ctk.CTkFrame(self.input_card, fg_color="transparent")
        self.progress_bar = ctk.CTkProgressBar(
            self.progress_row,
            mode="indeterminate",
            width=260,
            progress_color=self.colors["primary"],
        )

    def autoscroll(self):
        if self._message_widgets:
            try:
                self.chat_scroll._parent_canvas.yview_moveto(1.0)
            except Exception:
                pass

    def _bubble(self, text: str, kind: str):
        bg, border, txt, align = self._bubble_colors(kind)
        row = ctk.CTkFrame(self.chat_scroll, fg_color="transparent")
        row.pack(fill="x", padx=4, pady=6)
        bubble = ctk.CTkFrame(
            row,
            fg_color=bg,
            corner_radius=14,
            border_width=1,
            border_color=border,
        )
        bubble.pack(side="right" if align == "right" else "left", padx=6)
        # If message starts with "sympy:" interpret as sympy expression and pretty print
        pretty_text = None
        if text.strip().startswith("sympy:"):
            expr_str = text.strip()[6:].strip()
            try:
                expr = sp.sympify(expr_str)
                pretty_text = sp.pretty(expr, use_unicode=True)
            except Exception:
                pretty_text = "Error parsing expression."
        if pretty_text:
            label = ctk.CTkLabel(
                bubble,
                text=pretty_text,
                justify="left",
                anchor="w",
                text_color=txt,
                font=ctk.CTkFont(size=15, family="Consolas"),
            )
        else:
            label = ctk.CTkLabel(
                bubble,
                text=text,
                justify="left",
                wraplength=self.wraplength,
                anchor="w",
                text_color=txt,
                font=ctk.CTkFont(size=14),
            )
        label.pack(padx=12, pady=10)
        self._message_widgets.append(bubble)
        self.autoscroll()
        return label

    def _bubble_with_image(self, text: str, kind: str, image_path: str):
        bg, border, txt, align = self._bubble_colors(kind)
        row = ctk.CTkFrame(self.chat_scroll, fg_color="transparent")
        row.pack(fill="x", padx=4, pady=6)
        bubble = ctk.CTkFrame(
            row,
            fg_color=bg,
            corner_radius=14,
            border_width=1,
            border_color=border,
        )
        bubble.pack(side="right" if align == "right" else "left", padx=6)
        try:
            pil = Image.open(image_path)
            max_w_logical = int(420 * UI_SCALE)
            max_h_logical = int(320 * UI_SCALE)
            ctk_img = self._prepare_ctk_image(pil, (max_w_logical, max_h_logical))
            ctk.CTkLabel(bubble, image=ctk_img, text="").pack(padx=10, pady=(10, 6))
        except Exception as e:
            text = f"{text}\n\n[Image preview unavailable: {e}]"
        if text:
            txt_label = ctk.CTkLabel(
                bubble,
                text=text,
                justify="left",
                wraplength=self.wraplength,
                anchor="w",
                text_color=txt,
                font=ctk.CTkFont(size=14),
            )
            txt_label.pack(padx=12, pady=(0, 10))
        self._message_widgets.append(bubble)
        self.autoscroll()

    def _prepare_ctk_image(self, pil_img: Image.Image, max_logical_size: tuple[int, int]) -> ctk.CTkImage:
        max_w_logical, max_h_logical = max_logical_size
        target_px_size = (max_w_logical, max_h_logical)
        img = pil_img.copy()
        img = ImageOps.contain(img, target_px_size, method=RESAMPLE_BEST)
        logical_w = max(1, int(round(img.width / UI_SCALE)))
        logical_h = max(1, int(round(img.height / UI_SCALE)))
        return self._to_ctk_image(img, (logical_w, logical_h))

    def _bubble_colors(self, kind: str):
        if kind == "user":
            return (
                self.colors["user_bubble"],
                self.colors["user_border"],
                self.colors["user_text"],
                "right",
            )
        if kind == "system":
            return (
                self.colors["system_bubble"],
                self.colors["system_border"],
                self.colors["system_text"],
                "left",
            )
        return (
            self.colors["bot_bubble"],
            self.colors["bot_border"],
            self.colors["bot_text"],
            "left",
        )

    def add_message(self, sender: str, message: str):
        s = sender.lower()
        if s == "user":
            self._bubble(message, "user")
        elif s == "system":
            self._bubble(message, "system")
        else:
            self._bubble(message, "bot")

    def _hide_preview_row(self, reset=False):
        if hasattr(self, "preview_row") and self.preview_row.winfo_manager():
            self.preview_row.pack_forget()
        if reset:
            try:
                self.image_status.configure(text="No image selected.", text_color=self.colors["muted"])
            except Exception:
                pass
            self.root.after_idle(lambda: self.preview_label.configure(image=self.placeholder_image, text="No image"))
        self.image_preview = None

    def show_thinking(self):
        if not self.thinking:
            self.thinking = True
            self.progress_row.pack(pady=(0, 10))
            self.progress_bar.pack()
            self.progress_bar.start()

    def hide_thinking(self):
        if self.thinking:
            self.progress_bar.stop()
            self.progress_bar.pack_forget()
            self.progress_row.pack_forget()
            self.thinking = False

    def begin_streaming(self):
        self._streaming = True
        self._stream_buffer = ""
        self._cursor_visible = True
        self._stream_label = self._bubble("", "bot")
        self.hide_thinking()
        self._animate_stream_update()

    def _animate_stream_update(self):
        if not self._streaming or self._stream_label is None:
            return
        try:
            cursor = f" {self._cursor_glyph}" if self._cursor_visible else ""
            self._stream_label.configure(text=self._stream_buffer + cursor)
        except Exception:
            pass
        self._cursor_visible = not self._cursor_visible
        self.root.after(120, self._animate_stream_update)

    def append_stream_text(self, text_chunk: str):
        if text_chunk:
            self._stream_buffer += text_chunk

    def end_streaming(self):
        self._streaming = False
        if self._stream_label is not None:
            try:
                self._stream_label.configure(text=self._stream_buffer)
            except Exception:
                pass
        self._finish_request_cleanup()

    def _init_client(self):
        try:
            self.client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
        except Exception as e:
            self.client = None
            self.add_message("System", f"Failed to connect to the model: {e}")

    def _to_ctk_image(self, pil_img: Image.Image, size: tuple[int, int]) -> ctk.CTkImage:
        light = pil_img.copy()
        dark = pil_img.copy()
        cimg = ctk.CTkImage(light_image=light, dark_image=dark, size=size)
        self._img_cache.append(cimg)
        if len(self._img_cache) > 8:
            self._img_cache = self._img_cache[-8:]
        return cimg

    def _make_placeholder_image(self, size: tuple[int, int]) -> ctk.CTkImage:
        w, h = size
        light_bg = Image.new("RGBA", (w, h), (242, 245, 250, 255))
        dark_bg = Image.new("RGBA", (w, h), (30, 41, 59, 255))
        return ctk.CTkImage(light_image=light_bg, dark_image=dark_bg, size=size)

    def browse_image(self):
        file_path = filedialog.askopenfilename(
            title="Select an Image",
            filetypes=[("Image Files", "*.png *.jpg *.jpeg *.webp *.gif")]
        )
        if not file_path:
            self.image_path = None
            self._hide_preview_row(reset=True)
            return

        self.image_path = file_path
        self.image_status.configure(text=f"Selected: {os.path.basename(file_path)}", text_color="#111827")

        if not self.preview_row.winfo_manager():
            self.preview_row.pack(fill="x", padx=12, pady=(12, 6))

        try:
            pil = Image.open(file_path)
            ctk_img = self._prepare_ctk_image(pil, (int(96 * UI_SCALE), int(96 * UI_SCALE)))
            self.image_preview = ctk_img
            self.root.after_idle(lambda: self.preview_label.configure(image=self.image_preview, text=""))
        except Exception as e:
            messagebox.showerror("Image Error", f"Could not load preview: {e}")
            self.root.after_idle(lambda: self.preview_label.configure(image=self.placeholder_image, text="No image"))
            self._hide_preview_row(reset=True)
            self.image_preview = None

    @staticmethod
    def _guess_mime_type(path: str) -> str:
        ext = os.path.splitext(path)[1].lower()
        return {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".webp": "image/webp",
            ".gif": "image/gif",
        }.get(ext, "image/jpeg")

    @staticmethod
    def encode_image_to_base64(path: str) -> str | None:
        try:
            with open(path, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        except Exception as e:
            messagebox.showerror("File Error", f"Could not read or encode the image: {e}")
            return None

    def start_query_thread(self, event=None):
        user_input = (self.msg_entry.get() or "").strip()

        if not user_input and not self.image_path:
            messagebox.showwarning("Input Error", "Please enter a question or upload an image.")
            return

        if self.client is None:
            self.add_message("System", "OpenAI client not configured.")
            return

        if self.image_path:
            self._bubble_with_image(user_input if user_input else "[Image Query]", "user", self.image_path)
            self._hide_preview_row(reset=False)
        else:
            self.add_message("User", user_input)

        self.msg_entry.delete(0, "end")
        self.msg_entry.configure(state="disabled")
        self.send_button.configure(state="disabled")
        self.upload_button.configure(state="disabled")
        self.show_thinking()
        t = threading.Thread(target=self.submit_query_streaming, args=(user_input,))
        t.daemon = True
        t.start()

    def submit_query_streaming(self, prompt: str):
        final_prompt = prompt if (prompt or "").strip() else "What is in this image?"
        messages = [{"role": "user", "content": []}]
        messages[0]["content"].append({"type": "text", "text": final_prompt})

        if self.image_path:
            base64_image = self.encode_image_to_base64(self.image_path)
            if not base64_image:
                self.root.after(0, self.update_chat_with_response, "Error: Could not encode image.")
                return
            mime = self._guess_mime_type(self.image_path)
            messages[0]["content"].append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{base64_image}"}
            })

        self.root.after(0, self.begin_streaming)

        try:
            stream = self.client.chat.completions.create(
                model=API_MODEL,
                messages=messages,
                max_tokens=1024,
                stream=True,
            )
            for chunk in stream:
                try:
                    delta = None
                    if hasattr(chunk.choices[0], "delta"):
                        delta = chunk.choices[0].delta
                    elif isinstance(chunk.choices[0], dict) and "delta" in chunk.choices[0]:
                        delta = chunk.choices[0]["delta"]

                    text_part = ""
                    if isinstance(delta, dict):
                        text_part = delta.get("content") or ""
                    else:
                        text_part = getattr(delta, "content", None) or ""

                    if not text_part and hasattr(chunk.choices[0], "text"):
                        text_part = getattr(chunk.choices[0], "text") or ""
                    if not text_part and isinstance(chunk.choices[0], dict):
                        text_part = chunk.choices[0].get("text") or ""

                    if text_part:
                        self.append_stream_text(text_part)
                except Exception:
                    continue

            self.root.after(0, self.end_streaming)

        except openai.APIStatusError as e:
            error_message = f"API Error: {e.status_code}\n{e.response.text}"
            self.root.after(0, self.update_chat_with_response, error_message)
        except Exception as e:
            error_message = f"An unexpected error occurred: {e}"
            self.root.after(0, self.update_chat_with_response, error_message)

    def _finish_request_cleanup(self):
        self.image_path = None
        self._hide_preview_row(reset=True)
        self.image_preview = None
        self.msg_entry.configure(state="normal")
        self.send_button.configure(state="normal")
        self.upload_button.configure(state="normal")
        self.msg_entry.focus_set()

    def update_chat_with_response(self, response: str):
        try:
            self.hide_thinking()
            if self._streaming:
                self.end_streaming()
            self.add_message("Bot", response)
        finally:
            self._finish_request_cleanup()

if __name__ == "__main__":
    app_root = ctk.CTk()
    app = VisionChatApp(app_root)
    app_root.mainloop()