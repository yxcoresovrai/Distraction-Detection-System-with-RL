# redirector.py

import tkinter as tk
import webbrowser

def send_notification(goal: str):
    root = tk.Tk()
    root.title("⚠️ Refocus Alert")
    root.geometry("420x160")
    root.attributes("-topmost", True)

    message = f"You’ve already made progress. Don’t lose focus now. Stay on: \n\n{goal}"
    label = tk.Label(root, text=message, font=("Helvetica", 12), padx=10, pady=10)
    label.pack()

    def open_notion():
        webbrowser.open("https://www.notion.so")
        root.destroy()

    def open_kaggle():
        webbrowser.open("https://www.kaggle.com")
        root.destroy()

    def close_popup():
        root.destroy()

    tk.Button(root, text="📓 Open Notion", command=open_notion, width=20).pack(pady=5)
    tk.Button(root, text="📊 Open Kaggle", command=open_kaggle, width=20).pack(pady=5)
    tk.Button(root, text="❌ Dismiss", command=close_popup, width=20).pack(pady=5)

    root.mainloop()

def redirect_to(site: str):
    site_map = {
        "notion": "https://www.notion.so",
        "kaggle": "https://www.kaggle.com"
    }
    url = site_map.get(site.lower())
    if url:
        webbrowser.open(url)
