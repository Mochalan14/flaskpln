from knn_model import KNN
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import joblib

# preprocessing
df = pd.read_csv('data/dataset.csv')
df = df[['daya_lama', 'jam_nyala', 'pemakaian_kwh', 'histori']]
df['daya_lama'].value_counts()
options = [450, 900, 1300, 2200]
df_baru = df[df['daya_lama'].isin(options)]
duplicateRows = df_baru[df_baru.duplicated()]
duplicateRows.to_csv('data/duplicate.csv', index=True)
# df_baru.to_csv('data/dataset_preprocess.csv', index=False)
# print(df_baru.sample(n=11))

# proses training
X = df_baru.iloc[:, :-1].values
y = df_baru.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# print(len(X_train))
# print(len(X_test))

# k_values = []
# accuracies = []
# presisis = []
# recalls = []
# f1_scores = []

# for i in range(15):
#     nilai_k = i+1
#     model = KNN(k=nilai_k)
#     model.fit(X_train, y_train)
#     prediksi = model.predict(X_test)
#     # joblib.dump(model, 'knn.mdl')

#     TP = 0
#     TN = 0
#     FP = 0
#     FN = 0

#     for i in range(len(y_test)):
#         if y_test[i] == 1 and prediksi[i] == 1:
#             TP += 1
#         elif y_test[i] == 0 and prediksi[i] == 0:
#             TN += 1
#         elif y_test[i] == 0 and prediksi[i] == 1:
#             FP += 1
#         elif y_test[i] == 1 and prediksi[i] == 0:
#             FN += 1

#     confusion_matrix = [[TN, FP], [FN, TP]]
#     accuracy = (TP + TN) / (TP + TN + FP + FN)
#     precision = TP / (TP + FP)
#     recall = TP / (TP + FN)
#     f1_score = 2 * (precision * recall) / (precision + recall)

#     k_values.append(nilai_k)
#     accuracies.append(accuracy)
#     presisis.append(precision)
#     recalls.append(recall)
#     f1_scores.append(f1_score)
#     # print(
#     #     f'Nilai K = {nilai_k} Akurasi: {accuracy} , Presisi: {precision}, Recall: {recall}, F1score: {f1_score}')

# # joblib.dump(model, 'modelnew.mdl')

# plt.figure(figsize=(10, 6), dpi=80)
# plt.plot(k_values, accuracies, marker='o', linestyle='-')
# plt.title('Grafik Akurasi vs. Nilai k')
# plt.xlabel('Nilai k')
# plt.ylabel('Akurasi')
# plt.grid(True)
# plt.savefig("grafik_akurasi.jpg", format="jpeg")

# plt.figure(figsize=(10, 6), dpi=80)
# plt.plot(k_values, recalls, marker='o', linestyle='-')
# plt.title('Grafik Recall vs. Nilai k')
# plt.xlabel('Nilai k')
# plt.ylabel('Recall')
# plt.grid(True)
# plt.savefig("grafik_recall.jpg", format="jpeg")

# plt.figure(figsize=(10, 6), dpi=80)
# plt.plot(k_values, presisis, marker='o', linestyle='-')
# plt.title('Grafik Presisi vs. Nilai k')
# plt.xlabel('Nilai k')
# plt.ylabel('Presisi')
# plt.grid(True)
# plt.savefig("grafik_presisi.jpg", format="jpeg")

# plt.figure(figsize=(10, 6), dpi=80)
# plt.plot(k_values, f1_scores, marker='o', linestyle='-')
# plt.title('Grafik F1 Score vs. Nilai k')
# plt.xlabel('Nilai k')
# plt.ylabel('F1 Score')
# plt.grid(True)
# plt.savefig("grafik_f1score.jpg", format="jpeg")

# initial_data = {'akurasi': accuracies, 'presisi': presisis,
#                 'recall': recalls, 'f1_score': f1_scores}

# dfEvaluasi = pd.DataFrame(initial_data, columns=[
#                           'akurasi', 'presisi', 'recall', 'f1_score'])
# dfEvaluasi.index = k_values

# dfEvaluasi.to_csv('data/evaluasi.csv', index=True)
# joblib.dump(model, 'modelnew.mdl')
