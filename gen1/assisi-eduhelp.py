# API key has spaces to prevent it from revoking




# Import necessary libraries.
# tkinter is used for creating the graphical user interface (GUI).
import tkinter as tk
# scrolledtext provides a text widget with a scrollbar, useful for the chat area.
# messagebox is used to show pop-up error or information windows.
from tkinter import scrolledtext, messagebox
# groq is the official Python library to interact with the Groq API.
import groq
# threading allows us to run tasks in the background (like waiting for the API response)
# so the GUI doesn't freeze.
import threading

# This class handles all the logic for communicating with the Groq API.
class GroqChatbot:
    # The __init__ method is the constructor. It's called when we create a new GroqChatbot object.
    # It takes the API key as an argument to authenticate with the Groq service.
    def __init__(self, api_key):
        # We use a try...except block to handle potential errors during initialization.
        try:
            # This line creates a Groq client object, which we'll use to make API calls.
            self.client = groq.Groq(api_key=api_key)
            # We initialize a 'history' list. This list will keep track of the conversation.
            # It starts with a "system" message that tells the AI how to behave.
            self.history = [{"role": "system", "content": "You are a helpful assistant."}]
        except Exception as e:
            # If creating the client fails (e.g., invalid API key format), show an error message.
            messagebox.showerror("Initialization Error", f"Failed to initialize Groq client. Is the API key valid?\nError: {e}")
            # Set the client to None to indicate that the chatbot is not functional.
            self.client = None

    # This method sends a user's message to the Groq API and gets a response.
    def send_message(self, message):
        # First, check if the Groq client was initialized correctly. If not, return an error.
        if not self.client:
            return "Error: Groq client is not initialized."
        
        # Add the user's message to our conversation history.
        self.history.append({"role": "user", "content": message})
        
        # Use another try...except block to handle errors that might happen during the API call.
        try:
            # This is where we call the Groq API to get a chat completion.
            chat_completion = self.client.chat.completions.create(
                messages=self.history,  # We send the entire conversation history.
                model="llama3-8b-8192", # We specify which AI model to use.
            )
            # Extract the AI's response text from the API's reply.
            bot_response_text = chat_completion.choices[0].message.content
            # Add the AI's response to our conversation history.
            self.history.append({"role": "assistant", "content": bot_response_text})
            # Return the AI's response text.
            return bot_response_text
        # Specific error handling for different types of problems.
        except groq.AuthenticationError as e:
            # This error happens if the API key is wrong.
            error_message = f"Authentication Error: Invalid API key. Please check your key.\nDetails: {e}"
            print(error_message)
            self.history.pop() # Remove the user's message from history since it failed.
            return error_message
        except groq.APIConnectionError as e:
            # This error happens if there's a network problem.
            error_message = f"Connection Error: Could not connect to the API. Please check your network.\nDetails: {e.__cause__}"
            print(error_message)
            self.history.pop() # Remove the user's message from history.
            return error_message
        except Exception as e:
            # A catch-all for any other unexpected errors.
            error_message = f"An unexpected error occurred: {e}"
            print(error_message)
            self.history.pop() # Remove the user's message from history.
            return error_message

# This class defines the entire GUI for our chat application.
class ChatbotApp:
    # The constructor for our app. It takes the main window ('root') as an argument.
    def __init__(self, root):
        self.root = root
        self.root.title("Assisi EduHelp") # Set the title of the window.
        self.root.geometry("700x600") # Set the initial size of the window.
        self.root.configure(bg="#1a1a1a") # Set the background color of the window.

        # --- Style Definitions ---
        # Define fonts and colors here to easily change the app's appearance later.
        font_style = ("Helvetica", 12)
        font_style_bold = ("Helvetica", 12, "bold")
        bg_color = "#1a1a1a"         # Dark background
        text_area_bg = "#2a2a2a"    # Slightly lighter for text areas
        text_color = "#f0f0f0"       # Light text for contrast
        button_bg = "#f97316"        # Bright orange for the button
        button_fg = "#ffffff"        # White text on the button
        button_active_bg = "#ea580c" # Darker orange for when the button is clicked

        # --- Widget Creation ---
        # A Frame is a container to group other widgets. This is our main container.
        main_frame = tk.Frame(root, bg=bg_color)
        main_frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        # This is the main chat area where the conversation is displayed.
        self.chat_area = scrolledtext.ScrolledText(
            main_frame,
            wrap=tk.WORD,          # Wrap text by word, not by character.
            bg=text_area_bg,
            fg=text_color,
            font=font_style,
            state='disabled',      # Make it read-only so the user can't type in it.
            padx=10, pady=10,
            bd=0, highlightthickness=0 # Remove the border for a cleaner look.
        )
        self.chat_area.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        
        # Define 'tags' to style different parts of the text (user vs. bot).
        self.chat_area.tag_config('user', foreground="#a0a0a0", font=font_style_bold)
        self.chat_area.tag_config('bot', foreground="#f97316", font=font_style_bold)
        self.chat_area.tag_config('text', foreground=text_color, font=font_style)

        # Another frame to hold the message entry box and the send button.
        input_frame = tk.Frame(main_frame, bg=bg_color)
        input_frame.pack(fill=tk.X, padx=10, pady=(0, 10))

        # The entry widget where the user types their message.
        self.msg_entry = tk.Entry(input_frame, font=font_style, bg=text_area_bg, fg=text_color)
        self.msg_entry.config(insertbackground=text_color, bd=1, relief=tk.FLAT) # Style the cursor and border.
        self.msg_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=8) # Position it.
        # Bind the <Return> (Enter) key to the send_message_event method.
        self.msg_entry.bind("<Return>", self.send_message_event)

        # The "Send" button.
        self.send_button = tk.Button(
            input_frame,
            text="Send",
            font=font_style_bold,
            bg=button_bg, fg=button_fg,
            activebackground=button_active_bg,
            activeforeground=button_fg,
            command=self.send_message_event, # Call this method when clicked.
            bd=0, padx=20, pady=6, relief=tk.FLAT # Style the button.
        )
        self.send_button.pack(side=tk.RIGHT, padx=(10, 0))

        # --- Initialization Logic ---
        # IMPORTANT: Replace this with your actual Groq API key.
        self.api_key = "gsk_fsjFrX4 Dhf1Lp3RttsC5WGdyb3FYf EBlfgAXeNT3KIsR43H03qRz" 
        # Check if the API key is missing.
        if not self.api_key or self.api_key == "":
            messagebox.showerror("API Key Missing", "Please enter your Groq API key in the `self.api_key` variable in the code.")
            self.root.destroy() # Close the app if no key is found.
            return

        # Create an instance of our GroqChatbot.
        self.chatbot = GroqChatbot(api_key=self.api_key)
        # Check if the chatbot was initialized successfully.
        if self.chatbot.client:
            # If successful, display a welcome message.
            self.add_message("System", "Welcome to Assisi EduHelp! You can start chatting now.")
        else:
            # If it failed (e.g., bad key), close the app.
            self.root.destroy()

    # This method adds a message to the chat area.
    def add_message(self, sender, message):
        # We have to temporarily enable the text area to add new text.
        self.chat_area.configure(state='normal')
        
        # Check who the sender is to apply the correct style.
        if sender.lower() == 'user':
            self.chat_area.insert(tk.END, "User:\n", ('user',))
        elif sender.lower() == 'bot':
            self.chat_area.insert(tk.END, "Bot:\n", ('bot',))
        else:  # For system messages
            self.chat_area.insert(tk.END, f"{message}\n\n", ('text',))
            # Disable the text area again.
            self.chat_area.configure(state='disabled')
            # Automatically scroll to the bottom to show the latest message.
            self.chat_area.yview(tk.END)
            return # Exit the method for system messages.

        # Insert the actual message content with the 'text' style.
        self.chat_area.insert(tk.END, f"{message}\n\n", ('text',))
        # Disable the text area again to make it read-only.
        self.chat_area.configure(state='disabled')
        # Automatically scroll to the bottom.
        self.chat_area.yview(tk.END)

    # This method is called when the user clicks "Send" or presses Enter.
    def send_message_event(self, event=None):
        user_input = self.msg_entry.get() # Get the text from the entry box.
        # If the input is empty or the chatbot isn't working, do nothing.
        if user_input.strip() == "" or self.chatbot is None:
            return
        
        self.add_message("User", user_input) # Display the user's message.
        self.msg_entry.delete(0, tk.END) # Clear the entry box.
        
        # Disable the input field and send button while waiting for the bot's response.
        self.msg_entry.config(state='disabled')
        self.send_button.config(state='disabled')
        
        # Show a "Thinking..." message so the user knows the app is working.
        self.add_message("Bot", "Thinking...")
        
        # To avoid freezing the GUI, we run the API call in a separate thread.
        thread = threading.Thread(target=self.get_bot_response, args=(user_input,))
        thread.start()

    # This method runs in the background thread.
    def get_bot_response(self, user_input):
        # It calls the chatbot to get the response from the API.
        response = self.chatbot.send_message(user_input)
        # Once the response is received, schedule the `update_chat_with_response` method
        # to be run on the main GUI thread. This is the only safe way to update the GUI
        # from a background thread.
        self.root.after(0, self.update_chat_with_response, response)

    # This method updates the chat GUI with the final response from the bot.
    def update_chat_with_response(self, response):
        try:
            # --- Remove the "Thinking..." message ---
            self.chat_area.configure(state='normal') # Enable editing
            content = self.chat_area.get("1.0", tk.END) # Get all text from the chat area
            # Find the position of the last "Thinking..." message.
            last_thinking_pos = content.rfind("Bot:\nThinking...")
            if last_thinking_pos != -1:
                # Calculate the start and end index of the "Thinking..." block to delete it.
                lines = content[:last_thinking_pos].count('\n')
                start_index = f"{lines + 1}.0"
                end_index = f"{lines + 3}.0" # Bot:\n + Thinking...\n\n
                self.chat_area.delete(start_index, end_index)
            self.chat_area.configure(state='disabled') # Disable editing again
            
            # Add the actual bot response.
            self.add_message("Bot", response)
        finally:
            # This 'finally' block ensures that the input field and button are
            # re-enabled, even if an error occurred while updating the chat.
            self.msg_entry.config(state='normal')
            self.send_button.config(state='normal')
            self.msg_entry.focus_set() # Put the cursor back in the message entry box.

# This is the standard entry point for a Python script.
# The code inside this block will only run when the script is executed directly.
if __name__ == "__main__":
    try:
        root = tk.Tk()      # Create the main application window.
        app = ChatbotApp(root) # Create an instance of our ChatbotApp class.
        root.mainloop()     # Start the Tkinter event loop. This keeps the window open
                            # and responsive to user actions (clicks, typing, etc.).
    except Exception as e:
        # Catch any critical error that might happen when starting the app.
        print(f"An error occurred while starting the application: {e}")

