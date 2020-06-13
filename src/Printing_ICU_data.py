import numpy as np
import matplotlib.pyplot as plt

def main():
    my_data = np.genfromtxt('./data/New_UCI_June10.csv', delimiter=',')
    my_data = my_data[1:]
    plt.figure()
    plt.title("ICU data from Stockholm")
    plt.xlabel("Days, Day 0 = March 10")
    plt.ylabel("Number of new patients")
    plt.plot(my_data)
    plt.show()

if __name__ == "__main__":
    main()






