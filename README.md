Sebelum menjalankan sistemnya, diperlukan untuk install library terlebih dahulu untuk mendukung proses pelatihan model.

pip install pandas 
pip install scikit-learn 
pip install matplotlib 
pip install seaborn 
pip install xgboost 
pip install joblib 
pip install flask

Library yang dibutuhkan untuk mambantu preprocessing data :

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
import matplotlib
matplotlib.use('Agg')  # Menggunakan backend non-GUI
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import joblib
import os

Library yang dibutuhkan untuk membantu integrasi data dari sistem ke web :
from flask import Flask, render_template, jsonify, request, redirect, url_for, send_from_directory
import pandas as pd
import (masing-masing nama file .py yang digunakan) misalnya pada sistem ini terdapat 9 macam pelatihan,maka akan import semuanya ke flask dengan cara dibawah ini
import Main3k1, Main3k2, Main3k3, Main5k1, Main5k2, Main5k3, Main10k1, Main10k2, Main10k3, Main100

untuk menjalankan flask, maka dibutuhkan untuk menjalankan prompt dibawah ini :
1. CD skripsi = untuk memasuk ke dalam folder skripsi
2. Python app.py = untuk menjalankan flask yang nantinya akan mendapatkan link url local yang dapat menuju ke halaman web


