import matplotlib.pyplot as plt
import numpy as np

# 1

x = np.linspace(-10, 10, 500)
y = x**2 * np.sin(x)

plt.figure(figsize=(10, 6))
plt.plot(x, y, color='blue')

plt.axhline(0, color='black', linewidth=1) 
plt.axvline(0, color='black', linewidth=1) 

plt.grid(True, linestyle='-', alpha=0.7)
plt.title('Графік функції $f(x) = x^2 \sin(x)$')

plt.xlabel('x')
plt.ylabel('f(x)')

plt.show()

# 2

data = np.random.normal(5, 2, 1000)

plt.figure(figsize=(8, 5))
plt.hist(data, bins=30, color='lightblue', edgecolor='black', alpha=0.8)

plt.title(f"Нормальний розподіл (n=1000, μ={2}, σ={5})")
plt.grid(axis='y', linestyle='--', alpha=0.6)

plt.xlabel("Значення")
plt.ylabel("Кількість (Частота)")

plt.show()

# 3

labels = ['Програмування', 'Ігри', 'Слухання  музики', 'Футбол']
sizes = [30, 30, 25, 15]

plt.figure(figsize=(8, 8))
plt.pie(sizes, labels=labels, autopct='%1.0f%%', startangle=90, colors=['#ff9999','#66b3ff','#99ff99','#ffcc99'])
plt.title("Популярність мов програмування")
plt.axis('equal') 
plt.show()

# 4

apple = np.random.normal(150, 15, 100)
pineapple = np.random.normal(120, 10, 100)
pear = np.random.normal(180, 20, 100)
peaches = np.random.normal(150, 12, 100)

data = [apple, pineapple, pear, peaches]

plt.figure(figsize=(10, 6))

bp = plt.boxplot(data, labels=["Apple", "Pineapple", "Pear", "Peaches"])

plt.title("Розподіл маси фруктів")
plt.xlabel("Фрукти")
plt.ylabel("Маса (г)")
plt.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.show()