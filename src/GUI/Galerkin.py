from tkinter import ttk
from tkinter import *

class DisG():

  def __init__(self,window):
    self.wind = window
    self.wind.title("GALERKIN SOLVER")

    frame = LabelFrame(self.wind, text = 'Select the equation to solve.')
    frame.grid(row = 0, column = 0, columnspan = 3, pady = 20, bg = 'red')

    Label(frame, text = 'Advection').grid(row = 1, column = 0)
if __name__ == '__main__':
  window = Tk()
  application = DisG(window)
  window.mainloop()