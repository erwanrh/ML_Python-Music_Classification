# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 20:31:58 2021

@author: lilia
"""

import tkinter as tk

class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.pack()
        self.create_widgets()

    def create_widgets(self):
        self.hi_there = tk.Button(self)
        self.hi_there["text"] = "Lancer l'algorithme de détection"
        self.hi_there["command"] = self.lancer
        self.hi_there.pack(side="top")

        self.quit = tk.Button(self, text="Quitter", fg="red",
                              command=self.master.destroy)
        self.quit.pack(side="bottom")
        
    def lancer(self):
        print("Prédiction!")

root = tk.Tk()
app = Application(master=root)
app.mainloop()