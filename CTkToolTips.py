# based on: https://github.com/TomSchimansky/CustomTkinter/discussions/740#discussioncomment-4625758

import tkinter as tk
import customtkinter as ctk
from customtkinter import ThemeManager
import sys

class CTkToolTip(object):
    """
    Create a tooltip for a given widget. By default, CustomTkinter theme colours are used for the background and text.
    You must import ThemeManager from CustomTkinter for this widget.

    This is customised fpr CustomTkinter, but based on a Tkinter solution found here:
    https://stackoverflow.com/questions/3221956/how-do-i-display-tooltips-in-tkinter
    """

    def __init__(self, widget, text='widget info', fg_color=None):
        """
        Class initialisation.

        :param widget: The widget object to assign the tooltip to.
        :param text: The text to be displayed as the tooltip.
        :param fg_color:  Hex colour code (#RRGGBB), defining the colour of the tooltip. 
        """
        if fg_color is None:
            self.fg_color = self.get_color_from_name('CTkFrame', 'fg_color')
        else:
            self.fg_color = fg_color

        self.wait_time = 400  # milli-seconds
        self.wrap_length = 300  # pixels
        self._widget = widget
        self.text = text
        self.x_offset: int = +20
        self.y_offset: int = +10
        self.corner_radius: int = 10
        self.border_width: int = 1
        self.padding: tuple = (10, 2)       
        # Bind to the primary widget; use add="+" to avoid clobbering existing bindings
        self._widget.bind("<Enter>", self.on_enter, add="+")
        self._widget.bind("<Leave>", self.on_leave, add="+")
        self._widget.bind("<ButtonPress>", self.on_leave, add="+")

        # Also bind to an inner canvas if the widget is a composite (e.g., ProgressFrame)
        try:
            inner_targets = []
            # Prefer a known attribute used by ProgressFrame
            if hasattr(self._widget, 'progress_canvas'):
                inner_targets.append(self._widget.progress_canvas)
            else:
                # Fallback: bind to any direct tk.Canvas children
                for child in getattr(self._widget, 'winfo_children', lambda: [])():
                    try:
                        if isinstance(child, tk.Canvas) or getattr(child, 'winfo_class', lambda: '')() == 'Canvas':
                            inner_targets.append(child)
                    except Exception:
                        pass
            for target in inner_targets:
                try:
                    target.bind("<Enter>", self.on_enter, add="+")
                    target.bind("<Leave>", self.on_leave, add="+")
                    target.bind("<ButtonPress>", self.on_leave, add="+")
                except Exception:
                    pass
        except Exception:
            pass
        self._id = None
        self._tw = None

    def on_enter(self, event=None):
        self._schedule(event)

    def on_leave(self, event=None):
        self._unschedule()
        self.hide_tooltip()
        
    def set_text(self, text):
        self.text = text

    def _schedule(self, event=None):
        self._unschedule()
        # Schedule showing the tooltip after delay without calling immediately
        self._id = self._widget.after(self.wait_time, lambda e=event: self.show_tooltip(e))

    def _unschedule(self):
        id = self._id
        self._id = None
        if id:
            self._widget.after_cancel(id)

    def show_tooltip(self, event=None):
        # creates a toplevel window
        self._tw = tk.Toplevel(self._widget)

        # Leaves only the label and removes the app window
        self._tw.wm_overrideredirect(True)
        
        if sys.platform.startswith("win"):
            self._tw.transparent_color = self._widget._apply_appearance_mode(
                ThemeManager.theme["CTkToplevel"]["fg_color"])
            self._tw.attributes("-transparentcolor", self._tw.transparent_color)
            self._tw.transient()
        elif sys.platform.startswith("darwin"):
            self._tw.transparent_color = 'systemTransparent'
            self._tw.attributes("-transparent", True)
            self._tw.transient(self._tw.master)
        else:
            self._tw.transparent_color = '#000001'
            self.corner_radius = 0
            self._tw.transient()

        # self.resizable(width=True, height=True)

        # Make the background transparent
        self._tw.config(background=self._tw.transparent_color)        
        
        #self._tw.wm_attributes("-transparentcolor", "white")  # Set transparent color
        # self._tw.wm_geometry("+%d+%d" % (x, y))
        
        # create frame and label
        self.frame = ctk.CTkFrame(
            self._tw,
            corner_radius=self.corner_radius,
            border_width=self.border_width,
            fg_color=self.fg_color,
        )
        self.message_label = ctk.CTkLabel(self.frame, text=self.text)
        self.message_label.pack(
            fill="both",
            padx=self.padding[0] + self.border_width,
            pady=self.padding[1] + self.border_width,
            expand=True,
        )
        # Pack the frame into the Toplevel so the label becomes visible
        self.frame.pack(fill="both", expand=True)
        
        # Determine current pointer position for robust placement (works with synthetic events)
        try:
            pointer_x, pointer_y = self._widget.winfo_pointerxy()
        except Exception:
            # Fallback to widget origin if pointer position is unavailable
            pointer_x = self._widget.winfo_rootx()
            pointer_y = self._widget.winfo_rooty()

        # Calculate available space on the right side of the pointer relative to the screen
        root_width = self._tw.winfo_screenwidth()
        space_on_right = root_width - pointer_x

        # Calculate the width of the tooltip's text based on the message label
        text_width = self.message_label.winfo_reqwidth()

        # Calculate the offset based on available space and text width to avoid going off-screen on the right side
        offset_x = self.x_offset
        if space_on_right < text_width + 20:  # Adjust the threshold as needed
            offset_x = -text_width - 20  # Negative offset when space is limited on the right side

        # Position the tooltip near the pointer
        self._tw.geometry(f"+{pointer_x + offset_x}+{pointer_y + self.y_offset}")
        
        # label = tk.Label(self._tw, text=self._text, justify='left', fg=self._fg_color,
        #                 bg=self._bg_colour, relief='solid', borderwidth=1,
        #                 wraplength=self._wrap_length)
        # label.pack(ipadx=1)

    def hide_tooltip(self):
        tw = self._tw
        self._tw = None
        if tw:
            tw.destroy()
            
    @staticmethod
    def get_color_from_name(widget, name: str):
        """Gets the colour code associated with the supplied widget property,
        as defined by the currently active CustomTkinter theme."""
        mode = ctk.get_appearance_mode()
        if mode == 'Light':
            mode = 0
        else:
            mode = 1
        # colour = ThemeManager.theme["color"][name][mode]
        colour = ThemeManager.theme[widget][name] #[mode]
        return colour
