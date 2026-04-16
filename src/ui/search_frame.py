"""
Search Frame - Modern search interface with live as-you-type search.
"""

import os
import threading
import customtkinter as ctk
from typing import Callable, List
import subprocess
import sys


class SearchFrame(ctk.CTkFrame):
    """Main search UI with live debounced search."""

    def __init__(self, parent, on_search: Callable):
        super().__init__(parent, fg_color="transparent")

        self.on_search = on_search
        self.current_results: List[dict] = []
        self._debounce_id = None
        self._last_query = ""
        self._searching = False

        self._create_widgets()

    def _create_widgets(self):
        # ── App Title Bar ────────────────────────────────────
        header = ctk.CTkFrame(self, fg_color="transparent", height=50)
        header.pack(fill="x", padx=20, pady=(18, 0))

        title = ctk.CTkLabel(
            header,
            text="DocSearch",
            font=ctk.CTkFont(family="Segoe UI", size=26, weight="bold"),
            text_color=("#1a73e8", "#58a6ff"),
        )
        title.pack(side="left")

        subtitle = ctk.CTkLabel(
            header,
            text="Multilingual Semantic Search",
            font=ctk.CTkFont(size=12),
            text_color="gray50",
        )
        subtitle.pack(side="left", padx=(10, 0), pady=(8, 0))

        # ── Search Bar (Google-style) ────────────────────────
        search_container = ctk.CTkFrame(
            self, fg_color=("gray92", "gray17"), corner_radius=25, height=52
        )
        search_container.pack(fill="x", padx=20, pady=(14, 0))
        search_container.pack_propagate(False)

        # Search icon (magnifying glass unicode)
        icon = ctk.CTkLabel(
            search_container,
            text="\u2315",
            font=ctk.CTkFont(size=20),
            text_color="gray50",
            width=30,
        )
        icon.pack(side="left", padx=(16, 4))

        self.search_entry = ctk.CTkEntry(
            search_container,
            placeholder_text="Search documents in English, Hindi, or Telugu...",
            font=ctk.CTkFont(family="Segoe UI", size=15),
            height=44,
            border_width=0,
            fg_color="transparent",
            placeholder_text_color="gray50",
        )
        self.search_entry.pack(side="left", fill="both", expand=True, padx=(0, 4))

        # Bind live search events
        self.search_entry.bind("<KeyRelease>", self._on_key_release)
        self.search_entry.bind("<Return>", lambda e: self._do_search_now())
        self.search_entry.bind("<Escape>", lambda e: self._clear_search())

        # Clear button
        self.clear_btn = ctk.CTkButton(
            search_container,
            text="\u2715",
            width=32,
            height=32,
            corner_radius=16,
            font=ctk.CTkFont(size=14),
            fg_color="transparent",
            hover_color=("gray80", "gray30"),
            text_color="gray50",
            command=self._clear_search,
        )
        # Hidden initially, shown when there's text
        self.clear_btn.pack_forget()

        # Search button
        self.search_btn = ctk.CTkButton(
            search_container,
            text="\u2315 Search",
            width=90,
            height=36,
            corner_radius=18,
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color=("#1a73e8", "#1a73e8"),
            hover_color=("#155ab6", "#2b85f0"),
            text_color="white",
            command=self._do_search_now,
        )
        self.search_btn.pack(side="right", padx=(4, 8))

        # ── Status bar ───────────────────────────────────────
        status_row = ctk.CTkFrame(self, fg_color="transparent", height=24)
        status_row.pack(fill="x", padx=24, pady=(6, 0))

        self.status_label = ctk.CTkLabel(
            status_row,
            text="Ready  |  Type to search instantly",
            font=ctk.CTkFont(size=11),
            text_color="gray50",
        )
        self.status_label.pack(side="left")

        self.spinner_label = ctk.CTkLabel(
            status_row,
            text="",
            font=ctk.CTkFont(size=11),
            text_color=("#1a73e8", "#58a6ff"),
        )
        self.spinner_label.pack(side="right")

        # ── Results Area ─────────────────────────────────────
        self.results_frame = ctk.CTkScrollableFrame(
            self,
            fg_color="transparent",
            scrollbar_button_color=("gray75", "gray30"),
            scrollbar_button_hover_color=("gray65", "gray40"),
        )
        self.results_frame.pack(fill="both", expand=True, padx=16, pady=(8, 12))

    def _on_key_release(self, event):
        """Live search with debounce - triggers search 400ms after user stops typing."""
        query = self.search_entry.get().strip()

        # Show/hide clear button
        if query:
            self.clear_btn.pack(side="right", padx=(0, 2))
        else:
            self.clear_btn.pack_forget()

        # Don't re-search same query
        if query == self._last_query:
            return

        # Cancel previous debounce timer
        if self._debounce_id:
            self.after_cancel(self._debounce_id)

        if not query:
            self._last_query = ""
            self._show_placeholder()
            self.status_label.configure(text="Ready  |  Type to search instantly")
            return

        # Show typing indicator
        self.status_label.configure(text=f"Typing...", text_color="gray50")

        # Schedule search after 400ms of no typing
        self._debounce_id = self.after(400, lambda: self._do_search_debounced(query))

    def _do_search_debounced(self, query: str):
        """Execute debounced search in background thread."""
        if self._searching:
            return
        self._do_search(query)

    def _do_search_now(self):
        """Immediate search (on Enter key or button click)."""
        if self._debounce_id:
            self.after_cancel(self._debounce_id)
        query = self.search_entry.get().strip()
        if query:
            self._do_search(query)

    def _do_search(self, query: str = None):
        """Run search in background thread to keep UI responsive."""
        if query is None:
            query = self.search_entry.get().strip()
        if not query or self._searching:
            return

        self._searching = True
        self._last_query = query
        self.spinner_label.configure(text="Searching...")
        self.search_btn.configure(state="disabled")
        self.update_idletasks()

        def worker():
            try:
                results = self.on_search(query)
                self.after(0, lambda: self._show_results(results, query))
            except Exception as e:
                self.after(0, lambda: self.status_label.configure(
                    text=f"Error: {e}", text_color="#e74c3c"
                ))
            finally:
                self.after(0, lambda: self._search_done())

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()

    def _search_done(self):
        self._searching = False
        self.spinner_label.configure(text="")
        self.search_btn.configure(state="normal")

    def _clear_search(self):
        """Clear search and show placeholder."""
        self.search_entry.delete(0, "end")
        self.clear_btn.pack_forget()
        self._last_query = ""
        self._show_placeholder()
        self.status_label.configure(
            text="Ready  |  Type to search instantly", text_color="gray50"
        )
        self.search_entry.focus_set()

    def _show_placeholder(self):
        """Show welcome/placeholder content."""
        for widget in self.results_frame.winfo_children():
            widget.destroy()

        placeholder_frame = ctk.CTkFrame(
            self.results_frame, fg_color="transparent"
        )
        placeholder_frame.pack(expand=True, pady=40)

        icon = ctk.CTkLabel(
            placeholder_frame,
            text="\u2315",
            font=ctk.CTkFont(size=48),
            text_color="gray40",
        )
        icon.pack(pady=(0, 10))

        msg = ctk.CTkLabel(
            placeholder_frame,
            text="Search your documents instantly",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color="gray45",
        )
        msg.pack()

        suggestions = [
            ("network security", "Keyword match"),
            ("marksheet", "Find your marksheets"),
            ("\u0c35\u0c3f\u0c26\u0c4d\u0c2f (Telugu)", "Cross-language search"),
            ("house permission", "Semantic meaning match"),
            ("comp*", "Wildcard pattern"),
        ]

        tips_frame = ctk.CTkFrame(placeholder_frame, fg_color="transparent")
        tips_frame.pack(pady=(20, 0))

        for query, desc in suggestions:
            row = ctk.CTkFrame(tips_frame, fg_color="transparent")
            row.pack(anchor="w", pady=2)

            chip = ctk.CTkButton(
                row,
                text=query,
                font=ctk.CTkFont(size=12),
                fg_color=("gray85", "gray25"),
                hover_color=("gray75", "gray35"),
                text_color=("#1a73e8", "#58a6ff"),
                corner_radius=14,
                height=28,
                command=lambda q=query: self._search_suggestion(q),
            )
            chip.pack(side="left", padx=(0, 8))

            hint = ctk.CTkLabel(
                row,
                text=desc,
                font=ctk.CTkFont(size=11),
                text_color="gray50",
            )
            hint.pack(side="left")

    def _search_suggestion(self, query: str):
        """Fill and execute a suggestion search."""
        self.search_entry.delete(0, "end")
        self.search_entry.insert(0, query)
        self._do_search_now()

    def _show_results(self, results: List[dict], query: str):
        """Display search results."""
        for widget in self.results_frame.winfo_children():
            widget.destroy()

        self.current_results = results

        if not results:
            self.status_label.configure(
                text=f'No results for "{query}"', text_color="#e67e22"
            )
            no_frame = ctk.CTkFrame(self.results_frame, fg_color="transparent")
            no_frame.pack(expand=True, pady=40)
            ctk.CTkLabel(
                no_frame,
                text="No results found",
                font=ctk.CTkFont(size=16, weight="bold"),
                text_color="gray45",
            ).pack()
            ctk.CTkLabel(
                no_frame,
                text="Try different keywords, another language, or use wildcard (e.g., comp*)",
                font=ctk.CTkFont(size=12),
                text_color="gray50",
            ).pack(pady=(4, 0))
            return

        self.status_label.configure(
            text=f'{len(results)} results for "{query}"',
            text_color=("#2b7a0b", "#4caf50"),
        )

        for result in results:
            self._create_result_card(result)

    def display_results(self, results: List[dict], query: str):
        """Public method for backward compatibility."""
        self._show_results(results, query)

    def _create_result_card(self, result: dict):
        """Create a modern result card."""
        card = ctk.CTkFrame(
            self.results_frame,
            corner_radius=10,
            fg_color=("gray95", "gray20"),
            border_width=1,
            border_color=("gray85", "gray28"),
        )
        card.pack(fill="x", pady=4, padx=4)

        inner = ctk.CTkFrame(card, fg_color="transparent")
        inner.pack(fill="x", padx=14, pady=10)

        # ── Row 1: File info ─────────────────────────────────
        row1 = ctk.CTkFrame(inner, fg_color="transparent")
        row1.pack(fill="x")

        # Rank circle
        rank_colors = {1: "#1a73e8", 2: "#2b7a0b", 3: "#e67e22"}
        rank_bg = rank_colors.get(result["rank"], "gray40")
        rank = ctk.CTkLabel(
            row1,
            text=str(result["rank"]),
            width=26,
            height=26,
            corner_radius=13,
            fg_color=rank_bg,
            text_color="white",
            font=ctk.CTkFont(size=11, weight="bold"),
        )
        rank.pack(side="left", padx=(0, 8))

        # File type badge
        type_config = {
            "pdf": ("#e74c3c", "PDF"),
            "docx": ("#2980b9", "DOC"),
            "text": ("#27ae60", "TXT"),
            "excel": ("#1e8449", "XLS"),
            "image": ("#8e44ad", "IMG"),
        }
        ft = result.get("file_type", "unknown")
        badge_color, badge_text = type_config.get(ft, ("gray50", ft.upper()[:3]))

        type_badge = ctk.CTkLabel(
            row1,
            text=f" {badge_text} ",
            font=ctk.CTkFont(size=9, weight="bold"),
            fg_color=badge_color,
            corner_radius=4,
            text_color="white",
            height=20,
        )
        type_badge.pack(side="left", padx=(0, 8))

        # Filename (clickable)
        name_btn = ctk.CTkButton(
            row1,
            text=result["file_name"],
            font=ctk.CTkFont(family="Segoe UI", size=14, weight="bold"),
            fg_color="transparent",
            hover_color=("gray85", "gray28"),
            text_color=("#1a73e8", "#58a6ff"),
            anchor="w",
            height=26,
            command=lambda p=result["file_path"]: self._open_file(p),
        )
        name_btn.pack(side="left", fill="x", expand=True)

        # Score pill
        score = result["score"]
        score_color = "#2b7a0b" if score > 0.7 else "#e67e22" if score > 0.3 else "gray50"
        score_label = ctk.CTkLabel(
            row1,
            text=f"{score:.1%}",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=score_color,
        )
        score_label.pack(side="right", padx=(8, 0))

        # Language badge
        lang = result.get("language", "English")
        lang_config = {
            "Hindi": ("#ff9800", "HI"),
            "Telugu": ("#9c27b0", "TE"),
            "English": ("#607d8b", "EN"),
        }
        lang_color, lang_short = lang_config.get(lang, ("gray50", "??"))
        lang_badge = ctk.CTkLabel(
            row1,
            text=f" {lang_short} ",
            font=ctk.CTkFont(size=9, weight="bold"),
            fg_color=lang_color,
            corner_radius=4,
            text_color="white",
            height=20,
        )
        lang_badge.pack(side="right", padx=(0, 4))

        # ── Row 2: Snippet ───────────────────────────────────
        snippet = result.get("snippet", "")
        if snippet:
            snippet_text = snippet[:280].replace("\n", " ")
            snippet_label = ctk.CTkLabel(
                inner,
                text=snippet_text,
                font=ctk.CTkFont(size=11),
                text_color=("gray40", "gray60"),
                justify="left",
                anchor="w",
                wraplength=680,
            )
            snippet_label.pack(fill="x", pady=(4, 0))

        # ── Row 3: Path ──────────────────────────────────────
        path = result.get("file_path", "")
        if path:
            # Shorten long paths
            display_path = path if len(path) < 80 else "..." + path[-75:]
            path_label = ctk.CTkLabel(
                inner,
                text=display_path,
                font=ctk.CTkFont(size=9),
                text_color="gray45",
                anchor="w",
            )
            path_label.pack(fill="x", pady=(3, 0))

    def _open_file(self, file_path: str):
        """Open a file with the default system application."""
        try:
            if sys.platform == "win32":
                os.startfile(file_path)
            elif sys.platform == "darwin":
                subprocess.run(["open", file_path])
            else:
                subprocess.run(["xdg-open", file_path])
        except Exception as e:
            self.status_label.configure(
                text=f"Could not open file: {e}", text_color="#e74c3c"
            )

    def set_loading(self, message: str = "Loading..."):
        """Show a loading state."""
        for widget in self.results_frame.winfo_children():
            widget.destroy()
        frame = ctk.CTkFrame(self.results_frame, fg_color="transparent")
        frame.pack(expand=True, pady=40)
        ctk.CTkLabel(
            frame, text="\u231B", font=ctk.CTkFont(size=36), text_color="gray40"
        ).pack()
        ctk.CTkLabel(
            frame,
            text=message,
            font=ctk.CTkFont(size=14),
            text_color="gray50",
        ).pack(pady=(8, 0))
        self.update_idletasks()
