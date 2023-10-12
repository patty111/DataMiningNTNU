import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset into a Pandas DataFrame
df = pd.read_csv('duke_gpa.csv')

# Group the data by sleep night and calculate the mean GPA for each group
gpa_by_sleepnight = df.groupby('sleepnight')['gpa'].mean()
sleepnight_by_out = df.groupby('gender')['out'].mean()

# print(gpa_by_sleepnight)
# Plot a bar chart of the GPA by sleep night
sleepnight_by_out.plot(kind='bar')
plt.xlabel('Out days')
plt.ylabel('sleepnight')
plt.title('sleepnight by out')
plt.show()