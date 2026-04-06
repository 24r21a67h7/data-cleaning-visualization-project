import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)

data = pd.DataFrame({
    "Name": ["Student " + str(i) for i in range(1, 101)],
    "Age": np.random.randint(17, 25, 100),
    "Gender": np.random.choice(["Male", "Female"], 100),
    "Course": np.random.choice(["BTech", "BSc", "BCom", "BA"], 100),
    "Marks": np.random.randint(30, 110, 100),
    "Attendance (%)": np.random.randint(50, 101, 100),
    "City": np.random.choice(["Hyderabad", "Hyd", "Delhi", "Mumbai", None], 100)
})

data.loc[np.random.choice(data.index, 10), 'Marks'] = np.nan
data.loc[np.random.choice(data.index, 5), 'Age'] = np.nan
data.loc[np.random.choice(data.index, 5), 'Gender'] = None

print(data.head())
print(data.info())
print(data.describe())
print(data.isnull().sum())

data['Age'] = data['Age'].fillna(data['Age'].mean())
data['Marks'] = data['Marks'].fillna(data['Marks'].mean())
data['Attendance (%)'] = data['Attendance (%)'].fillna(data['Attendance (%)'].mean())

data['Gender'] = data['Gender'].fillna(data['Gender'].mode()[0])
data['Course'] = data['Course'].fillna(data['Course'].mode()[0])
data['City'] = data['City'].fillna(data['City'].mode()[0])
data['Name'] = data['Name'].fillna("Unknown")

data = data.drop_duplicates()

data['Name'] = data['Name'].str.strip().str.title()
data['City'] = data['City'].str.strip().str.title()

data['City'] = data['City'].replace({'Hyd': 'Hyderabad'})

data.loc[data['Marks'] > 100, 'Marks'] = data['Marks'].median()

data.to_csv("cleaned_student_dataset.csv", index=False)

plt.figure()
sns.histplot(data['Marks'], kde=True)
plt.title("Marks Distribution")
plt.show()

plt.figure()
sns.countplot(x='Course', data=data)
plt.xticks(rotation=45)
plt.title("Students per Course")
plt.show()

plt.figure()
sns.scatterplot(x='Attendance (%)', y='Marks', data=data)
plt.title("Attendance vs Marks")
plt.show()

plt.figure()
sns.boxplot(x=data['Marks'])
plt.title("Outliers in Marks")
plt.show()

plt.figure()
sns.heatmap(data.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

sns.pairplot(data)
plt.show()

print("Average Marks:", data['Marks'].mean())
print("Highest Marks:", data['Marks'].max())
print("Lowest Marks:", data['Marks'].min())
print("Most Common Course:", data['Course'].mode()[0])