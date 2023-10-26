from tkinter import *
from tkinter import messagebox
from tkinter import ttk
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score

df = pd.read_csv('winequality-red-1.csv')
data = np.array(df[['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol']].values)
dt_Train,dt_Test = train_test_split(data,test_size = 0.1 ,shuffle = True)
X_train = dt_Train[:,:11]
X_test = dt_Test[:,:11]

# form
form = Tk()
form.title("Dự đoán nồng độ cồn của rượu:")
form.geometry("1000x500")

lable_title = Label(form, text="Nhập thông tin phân cụm loại rượu:", font=("Arial Bold", 10), fg="red")
lable_title.grid(row=1, column=1, padx=40, pady=10)

lable_Fixedacidity = Label(form, text=" Fixed acidity:")
lable_Fixedacidity.grid(row=2, column=1, padx=40, pady=10)
textbox_Fixedacidity = Entry(form)
textbox_Fixedacidity.grid(row=2, column=2)

lable_Volatileacidity = Label(form, text="Volatile acidity:")
lable_Volatileacidity.grid(row=3, column=1, pady=10)
textbox_Volatileacidity = Entry(form)
textbox_Volatileacidity.grid(row=3, column=2)

lable_Citricacid = Label(form, text="Citric acid:")
lable_Citricacid.grid(row=4, column=1, pady=10)
textbox_Citricacid = Entry(form)
textbox_Citricacid.grid(row=4, column=2)

lable_Residualsugar = Label(form, text="Residual sugar:")
lable_Residualsugar.grid(row=5, column=1, pady=10)
textbox_Residualsugar = Entry(form)
textbox_Residualsugar.grid(row=5, column=2)

lable_Chlorides = Label(form, text="Chlorides:")
lable_Chlorides.grid(row=6, column=1, pady=10)
textbox_Chlorides = Entry(form)
textbox_Chlorides.grid(row=6, column=2)

lable_Freesulfurdioxide = Label(form, text="Free sulfur dioxide:")
lable_Freesulfurdioxide.grid(row=7, column=1, pady=10)
textbox_Freesulfurdioxide = Entry(form)
textbox_Freesulfurdioxide.grid(row=7, column=2)

lable_Totalsulfurdioxide = Label(form, text="Total sulfur dioxide:")
lable_Totalsulfurdioxide.grid(row=2, column=3, pady=10)
textbox_Totalsulfurdioxide = Entry(form)
textbox_Totalsulfurdioxide.grid(row=2, column=4)

lable_density = Label(form, text="Density:")
lable_density.grid(row=3, column=3, pady=10)
textbox_density = Entry(form)
textbox_density.grid(row=3, column=4)

lable_pH = Label(form, text="pH:")
lable_pH.grid(row=4, column=3, pady=10)
textbox_pH = Entry(form)
textbox_pH.grid(row=4, column=4)

lable_sulphates = Label(form, text="Sulphates:")
lable_sulphates.grid(row=5, column=3, pady=10)
textbox_sulphates = Entry(form)
textbox_sulphates.grid(row=5, column=4)

lable_alcohol = Label(form, text="Alcohol:")
lable_alcohol.grid(row=6, column=3, pady=10)
textbox_alcohol = Entry(form)
textbox_alcohol.grid(row=6, column=4)


kmeans = KMeans(n_clusters=3,init='random',n_init=10).fit(X_train)
y_pre = kmeans.predict(X_test)
lbl1 = Label(form)
lbl1.grid(column=1, row=9)
lbl1.configure(text="Các độ đo của Kmean: " + '\n'
                    + "Silhouette: " + str(silhouette_score(X_test,y_pre)) + '\n'
                    + "Davies Bouldin: " + str(davies_bouldin_score(X_test,y_pre)) + '\n',
               font=("Arial Bold", 10), fg="red")

def kmean():
    Fixedacidit = textbox_Fixedacidity.get()
    Volatileacidity = textbox_Volatileacidity.get()
    Citricacid = textbox_Citricacid.get()
    Residualsugar = textbox_Residualsugar.get()
    Chlorides = textbox_Chlorides.get()
    Freesulfurdioxide = textbox_Freesulfurdioxide.get()
    Totalsulfurdioxide = textbox_Totalsulfurdioxide.get()
    Density = textbox_density.get()
    pH = textbox_pH.get()
    Sulphates = textbox_sulphates.get()
    Alcohol = textbox_alcohol.get()
    if ((Fixedacidit == '') or (Volatileacidity == '') or (Citricacid == '') or (Residualsugar == '') or (Chlorides == '') or (Freesulfurdioxide == '') or (Totalsulfurdioxide == '')  or (Density == '') or (pH == '') or (Sulphates == '') or (Alcohol == '') ):
        messagebox.showinfo("Thông báo", "Bạn cần nhập đầy đủ thông tin!")
    else:
        X_dudoan = np.array([Fixedacidit,Volatileacidity,Citricacid,Residualsugar,Chlorides,Freesulfurdioxide,Totalsulfurdioxide,Density,pH,Sulphates,Alcohol]).reshape(1, -1)
        y_pre = kmeans.predict(X_dudoan)
        lbl.configure(text=y_pre[0])

button_cart = Button(form, text='Kết quả dự đoán Kmean', command=kmean)
button_cart.grid(row=11, column=1, pady=20)
lbl1 = Label(form, text="Dữ liệu trên thuộc nhóm: ")
lbl1.grid(column=2, row=11)
lbl = Label(form, text="...")
lbl.grid(column=3, row=11)

form.mainloop()
