"""
Directory Selection Dialog - Modern first-launch setup.
"""

import customtkinter as ctk
from tkinter import filedialog
from typing import List, Optional


class DirectoryDialog(ctk.CTkToplevel):
    """Dialog for selecting directories to index."""

    def __init__(self, parent, existing_dirs: List[str] = None):
        super().__init__(parent)
        self.title("DocSearch - Select Folders")
        self.geometry("620x480")
        self.resizable(False, False)

        self.selected_dirs: List[str] = list(existing_dirs or [])
        self.result: Optional[List[str]] = None

        self._create_widgets()

        # Make modal
        self.transient(parent)
        self.grab_set()
        self.focus_set()

        # Center on parent
        self.update_idletasks()
        x = parent.winfo_x() + (parent.winfo_width() - 620) // 2
        y = parent.winfo_y() + (parent.winfo_height() - 480) // 2
        self.geometry(f"+{x}+{y}")

    def _create_widgets(self):
        # ── Header ───────────────────────────────────────────
        header_frame = ctk.CTkFrame(self, fg_color="transparent")
        header_frame.pack(fill="x", padx=24, pady=(24, 0))

        icon = ctk.CTkLabel(
            header_frame,
            text="\U0001F4C1",
            font=ctk.CTkFont(size=28),
        )
        icon.pack(side="left", padx=(0, 12))

        text_frame = ctk.CTkFrame(header_frame, fg_color="transparent")
        text_frame.pack(side="left")

        ctk.CTkLabel(
            text_frame,
            text="Select folders to index",
            font=ctk.CTkFont(size=18, weight="bold"),
        ).pack(anchor="w")

        ctk.CTkLabel(
            text_frame,
            text="DocSearch will scan these folders for PDFs, documents, images, and spreadsheets.",
            font=ctk.CTkFont(size=11),
            text_color="gray50",
            wraplength=400,
        ).pack(anchor="w", pady=(2, 0))

        # ── Directory list ───────────────────────────────────
        list_frame = ctk.CTkFrame(
            self, fg_color=("gray92", "gray17"), corner_radius=10
        )
        list_frame.pack(fill="both", expand=True, padx=24, pady=(16, 0))

        self.dir_listbox = ctk.CTkTextbox(
            list_frame,
            state="disabled",
            font=ctk.CTkFont(family="Consolas", size=12),
            fg_color="transparent",
            wrap="none",
        )
        self.dir_listbox.pack(fill="both", expand=True, padx=12, pady=12)
        self._refresh_list()

        # ── Action buttons ───────────────────────────────────
        btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        btn_frame.pack(fill="x", padx=24, pady=(12, 0))

        add_btn = ctk.CTkButton(
            btn_frame,
            text="+ Add Folder",
            command=self._add_directory,
            width=140,
            height=36,
            corner_radius=8,
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color=("#2b7a0b", "#2b7a0b"),
            hover_color=("#1e5a08", "#3a9c14"),
        )
        add_btn.pack(side="left", padx=(0, 8))

        remove_btn = ctk.CTkButton(
            btn_frame,
            text="Remove Last",
            command=self._remove_last,
            width=120,
            height=36,
            corner_radius=8,
            font=ctk.CTkFont(size=13),
            fg_color=("#c0392b", "#c0392b"),
            hover_color=("#96281b", "#e74c3c"),
        )
        remove_btn.pack(side="left")

        # ── Bottom buttons ───────────────────────────────────
        bottom_frame = ctk.CTkFrame(self, fg_color="transparent")
        bottom_frame.pack(fill="x", padx=24, pady=(16, 24))

        cancel_btn = ctk.CTkButton(
            bottom_frame,
            text="Cancel",
            command=self._cancel,
            width=100,
            height=38,
            corner_radius=8,
            font=ctk.CTkFont(size=13),
            fg_color="transparent",
            border_width=1,
            border_color="gray40",
            hover_color=("gray80", "gray30"),
            text_color=("gray30", "gray70"),
        )
        cancel_btn.pack(side="right", padx=(8, 0))

        ok_btn = ctk.CTkButton(
            bottom_frame,
            text="Start Indexing",
            command=self._ok,
            width=160,
            height=38,
            corner_radius=8,
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color=("#1a73e8", "#1a73e8"),
            hover_color=("#155ab6", "#2b85f0"),
        )
        ok_btn.pack(side="right")

    def _add_directory(self):
        directory = filedialog.askdirectory(title="Select a folder to index")
        if directory and directory not in self.selected_dirs:
            self.selected_dirs.append(directory)
            self._refresh_list()

    def _remove_last(self):
        if self.selected_dirs:
            self.selected_dirs.pop()
            self._refresh_list()

    def _refresh_list(self):
        self.dir_listbox.configure(state="normal")
        self.dir_listbox.delete("1.0", "end")
        if self.selected_dirs:
            for i, d in enumerate(self.selected_dirs, 1):
                self.dir_listbox.insert("end", f"  {i}.  {d}\n")
        else:
            self.dir_listbox.insert(
                "end",
                "  No folders selected yet.\n\n"
                "  Click '+ Add Folder' to select folders\n"
                "  containing your documents.",
            )
        self.dir_listbox.configure(state="disabled")

    def _ok(self):
        self.result = self.selected_dirs
        self.destroy()

    def _cancel(self):
        self.result = None
        self.destroy()

    def get_result(self) -> Optional[List[str]]:
        """Wait for dialog to close and return result."""
        self.wait_window()
        return self.result
