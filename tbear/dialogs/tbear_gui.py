import os
import tkinter as tk
from tkinter import ttk
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

FILE_PATH = os.path.dirname(os.path.abspath(__file__))
ICON_PATH = os.path.realpath(os.path.join(FILE_PATH, r'icons8-bear-16.ico'))
LARGE_FONT = ('Roboto', 13)


class TBEARApp(tk.Tk):
    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        tk.Tk.iconbitmap(self, default=ICON_PATH)
        tk.Tk.wm_title(self, 'T-BEAR')
        container = tk.Frame(self)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)
        container.pack(side='top', fill='both', expand=True)

        self.frames = {}

        for F in (HomePage, PageOne):
            frame = F(container, self)
            self.frames[F] = frame
            frame.grid(row=0, column=0, sticky='nsew')

        self.show_frame(HomePage)

    def show_frame(self, controller):
        frame = self.frames[controller]
        frame.tkraise()


class HomePage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text='Home', font=LARGE_FONT)
        label.pack(padx=10, pady=10)

        button = ttk.Button(self, text='Inspect POI',
                            command=lambda: controller.show_frame(PageOne))
        button.pack()


class PageOne(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text='Artifact Rejection', font=LARGE_FONT)
        label.pack(padx=10, pady=10)

        button1 = ttk.Button(self, text='Home',
                             command=lambda: controller.show_frame(HomePage))
        button1.pack()

        button2 = ttk.Button(self, text='Back',
                             command=lambda: controller.show_frame(HomePage))
        button2.pack()

        button3 = ttk.Button(self, text='Next',
                             command=lambda: controller.show_frame(HomePage))
        button3.pack()


class PageTwo(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text='Page Two', font=LARGE_FONT)
        label.pack(pady=10, padx=10)

        button1 = ttk.Button(self, text='Back to Home',
                             command=lambda: controller.show_frame(HomePage))
        button1.pack()

        button2 = ttk.Button(self, text='Page One',
                             command=lambda: controller.show_frame(PageOne))
        button2.pack()


if __name__ == '__main__':
    app = TBEARApp()
    app.mainloop()
