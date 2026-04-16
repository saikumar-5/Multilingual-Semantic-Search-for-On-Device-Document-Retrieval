"""
Settings Panel - Clean sidebar with search mode and directory management.
"""

import customtkinter as ctk
from typing import Callable, List


class SettingsFrame(ctk.CTkFrame):
    """Sidebar settings panel."""

    def __init__(
        self,
        parent,
        indexed_dirs: List[str],
        on_add_directory: Callable,
        on_reindex: Callable,
        on_mode_change: Callable,
        current_mode: str = "hybrid",
    ):
        super().__init__(parent, fg_color=("gray94", "gray14"), corner_radius=12)

        self.on_add_directory = on_add_directory
        self.on_reindex = on_reindex
        self.on_mode_change = on_mode_change

        self._create_widgets(indexed_dirs, current_mode)

    def _create_widgets(self, indexed_dirs: List[str], current_mode: str):
        # ── Search Mode ─────────────────────────────────────
        mode_header = ctk.CTkLabel(
            self,
            text="Search Mode",
            font=ctk.CTkFont(size=14, weight="bold"),
        )
        mode_header.pack(pady=(18, 8), padx=16, anchor="w")

        self.mode_var = ctk.StringVar(value=current_mode)

        modes = [
            ("Hybrid", "hybrid", "Keyword + Semantic combined"),
            ("Keyword", "keyword", "TF-IDF exact matching"),
            ("Semantic", "semantic", "AI meaning-based search"),
        ]

        for label, value, tooltip in modes:
            btn_frame = ctk.CTkFrame(self, fg_color="transparent")
            btn_frame.pack(fill="x", padx=16, pady=1)

            rb = ctk.CTkRadioButton(
                btn_frame,
                text=label,
                variable=self.mode_var,
                value=value,
                command=lambda: self.on_mode_change(self.mode_var.get()),
                font=ctk.CTkFont(size=13),
                radiobutton_width=18,
                radiobutton_height=18,
            )
            rb.pack(side="left")

            hint = ctk.CTkLabel(
                btn_frame,
                text=tooltip,
                font=ctk.CTkFont(size=9),
                text_color="gray50",
            )
            hint.pack(side="right", padx=(0, 4))

        # ── Separator ────────────────────────────────────────
        ctk.CTkFrame(self, height=1, fg_color="gray35").pack(
            fill="x", padx=16, pady=14
        )

        # ── Indexed Directories ──────────────────────────────
        dir_header = ctk.CTkLabel(
            self,
            text="Indexed Folders",
            font=ctk.CTkFont(size=14, weight="bold"),
        )
        dir_header.pack(pady=(0, 6), padx=16, anchor="w")

        self.dir_list = ctk.CTkTextbox(
            self,
            height=110,
            state="disabled",
            font=ctk.CTkFont(family="Consolas", size=10),
            fg_color=("gray88", "gray18"),
            corner_radius=8,
        )
        self.dir_list.pack(fill="x", padx=16, pady=(0, 8))
        self.update_directories(indexed_dirs)

        # Buttons row
        btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        btn_frame.pack(fill="x", padx=16, pady=(0, 6))

        add_btn = ctk.CTkButton(
            btn_frame,
            text="+ Add Folder",
            command=self.on_add_directory,
            height=32,
            corner_radius=8,
            font=ctk.CTkFont(size=12, weight="bold"),
            fg_color=("#2b7a0b", "#2b7a0b"),
            hover_color=("#1e5a08", "#3a9c14"),
        )
        add_btn.pack(side="left", fill="x", expand=True, padx=(0, 4))

        reindex_btn = ctk.CTkButton(
            btn_frame,
            text="Re-Index",
            command=self.on_reindex,
            height=32,
            corner_radius=8,
            font=ctk.CTkFont(size=12, weight="bold"),
            fg_color=("#e67e22", "#e67e22"),
            hover_color=("#d35400", "#f0932b"),
        )
        reindex_btn.pack(side="right", fill="x", expand=True, padx=(4, 0))

        # ── Separator ────────────────────────────────────────
        ctk.CTkFrame(self, height=1, fg_color="gray35").pack(
            fill="x", padx=16, pady=14
        )

        # ── Stats ────────────────────────────────────────────
        self.stats_label = ctk.CTkLabel(
            self,
            text="",
            font=ctk.CTkFont(size=11),
            text_color="gray50",
            justify="left",
        )
        self.stats_label.pack(padx=16, anchor="w")

        # ── About (bottom) ──────────────────────────────────
        spacer = ctk.CTkFrame(self, fg_color="transparent")
        spacer.pack(fill="both", expand=True)

        about_frame = ctk.CTkFrame(self, fg_color="transparent")
        about_frame.pack(fill="x", padx=16, pady=(0, 14))

        ctk.CTkLabel(
            about_frame,
            text="DocSearch v1.0",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="gray45",
        ).pack(anchor="w")
        # ctk.CTkLabel(
        #     about_frame,
        #     text="CSE3707 - Information Retrieval\nBMU, Batch 2023",
        #     font=ctk.CTkFont(size=10),
        #     text_color="gray40",
        #     justify="left",
        # ).pack(anchor="w")

    def update_directories(self, dirs: List[str]):
        """Update the displayed list of indexed directories."""
        self.dir_list.configure(state="normal")
        self.dir_list.delete("1.0", "end")
        if dirs:
            for d in dirs:
                # Show just the last part of path for brevity
                short = d.replace("\\", "/").split("/")[-1] or d
                self.dir_list.insert("end", f" {short}\n   {d}\n\n")
        else:
            self.dir_list.insert("end", " No folders indexed yet.\n Click '+ Add Folder' to begin.")
        self.dir_list.configure(state="disabled")

    def update_stats(self, num_docs: int, num_terms: int):
        """Update index statistics display."""
        self.stats_label.configure(
            text=f"Documents: {num_docs}\nTerms: {num_terms:,}"
        )
