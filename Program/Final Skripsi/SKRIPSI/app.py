from flask import Flask, render_template, jsonify, request, redirect, url_for, send_from_directory
import pandas as pd
import Main3k1, Main3k2, Main3k3, Main5k1, Main5k2, Main5k3, Main10k1, Main10k2, Main10k3, Main100

app = Flask(__name__)

# Fungsi untuk membaca CSV
def read_csv(choice):
    if choice == 1:
        df = pd.read_csv('C:\\Users\\User\\Desktop\\Final Skripsi\\SKRIPSI\\Mental_Health100.csv')
    elif choice == 2:
        df = pd.read_csv('C:\\Users\\User\\Desktop\\Final Skripsi\\SKRIPSI\\Mental_Health_3k(1).csv')
    elif choice == 3:
        df = pd.read_csv('C:\\Users\\User\\Desktop\\Final Skripsi\\SKRIPSI\\Mental_Health_3k(2).csv')
    elif choice == 4:
        df = pd.read_csv('C:\\Users\\User\\Desktop\\Final Skripsi\\SKRIPSI\\Mental_Health_3k(3).csv')
    elif choice == 5:
        df = pd.read_csv('C:\\Users\\User\\Desktop\\Final Skripsi\\SKRIPSI\\Mental_Health_5k(1).csv')
    elif choice == 6:
        df = pd.read_csv('C:\\Users\\User\\Desktop\\Final Skripsi\\SKRIPSI\\Mental_Health_5k(2).csv')
    elif choice == 7:
        df = pd.read_csv('C:\\Users\\User\\Desktop\\Final Skripsi\\SKRIPSI\\Mental_Health_5k(3).csv')
    elif choice == 8:
        df = pd.read_csv('C:\\Users\\User\\Desktop\\Final Skripsi\\SKRIPSI\\Mental_Health_10k(1).csv')
    elif choice == 9:
        df = pd.read_csv('C:\\Users\\User\\Desktop\\Final Skripsi\\SKRIPSI\\Mental_Health_10k(2).csv')
    elif choice == 10:
        df = pd.read_csv('C:\\Users\\User\\Desktop\\Final Skripsi\\SKRIPSI\\Mental_Health_10k(3).csv')
    else:
        raise ValueError("Invalid choice. Please choose 1 or 2 or 3 or 4 or 5 or 6 or 7 or 8 or 9 or 10.")
    return df

# Variabel untuk menyimpan akurasi dari model agar tidak perl   u melatih dua kali
cached_accuracy = {'accuracy_rf_1': None, 'accuracy_xgb_1': None,
                'accuracy_rf_2': None, 'accuracy_xgb_2': None,
                'accuracy_rf_3': None, 'accuracy_xgb_3': None,
                'accuracy_rf_4': None, 'accuracy_xgb_4': None,
                'accuracy_rf_5': None, 'accuracy_xgb_5': None,
                'accuracy_rf_6': None, 'accuracy_xgb_6': None,
                'accuracy_rf_7': None, 'accuracy_xgb_7': None,
                'accuracy_rf_8': None, 'accuracy_xgb_8': None,
                'accuracy_rf_9': None, 'accuracy_xgb_9': None,
                'accuracy_rf_10': None, 'accuracy_xgb_10': None}

cached_precision = {'precision_rf_1': None, 'precision_xgb_1': None,
                'precision_rf_2': None, 'precision_xgb_2': None,
                'precision_rf_3': None, 'precision_xgb_3': None,
                'precision_rf_4': None, 'precision_xgb_4': None,
                'precision_rf_5': None, 'precision_xgb_5': None,
                'precision_rf_6': None, 'precision_xgb_6': None,
                'precision_rf_7': None, 'precision_xgb_7': None,
                'precision_rf_8': None, 'precision_xgb_8': None,
                'precision_rf_9': None, 'precision_xgb_9': None,
                'precision_rf_10': None, 'precision_xgb_10': None}

cached_recall = {'recall_rf_1': None, 'recall_xgb_1': None,
                'recall_rf_2': None, 'recall_xgb_2': None,
                'recall_rf_3': None, 'recallxgb_3': None,
                'recall_rf_4': None, 'recall_xgb_4': None,
                'recall_rf_5': None, 'recall_xgb_5': None,
                'recall_rf_6': None, 'recall_xgb_6': None,
                'recall_rf_7': None, 'recall_xgb_7': None,
                'recall_rf_8': None, 'recall_xgb_8': None,
                'recall_rf_9': None, 'recall_xgb_9': None,
                'recall_rf_10': None, 'recall_xgb_10': None}

cached_f1 = {'f1_rf_1': None, 'f1_xgb_1': None,
            'f1_rf_2': None, 'f1_xgb_2': None,
            'f1_rf_3': None, 'f1_xgb_3': None,
            'f1_rf_4': None, 'f1_xgb_4': None,
            'f1_rf_5': None, 'f1_xgb_5': None,
            'f1_rf_6': None, 'f1_xgb_6': None,
            'f1_rf_7': None, 'f1_xgb_7': None,
            'f1_rf_8': None, 'f1_xgb_8': None,
            'f1_rf_9': None, 'f1_xgb_9': None,
            'f1_rf_10': None, 'f1_xgb_10': None}

cached_values_count = {'yes_count_1': None, 'no_count_1': None,
                    'yes_count_2': None, 'no_count_2': None,
                    'yes_count_3': None, 'no_count_3': None, 
                    'yes_count_4': None, 'no_count_4': None,
                    'yes_count_5': None, 'no_count_5': None, 
                    'yes_count_6': None, 'no_count_6': None,
                    'yes_count_7': None, 'no_count_7': None,
                    'yes_count_8': None, 'no_count_8': None,
                    'yes_count_9': None, 'no_count_9': None,
                    'yes_count_10': None, 'no_count_10': None}

cached_TP = {'TP_rf_1': None, 'TP_xgb_1': None,
            'TP_rf_2': None, 'TP_xgb_2': None,
            'TP_rf_3': None, 'TP_xgb_3': None,
            'TP_rf_4': None, 'TP_xgb_4': None,
            'TP_rf_5': None, 'TP_xgb_5': None,
            'TP_rf_6': None, 'TP_xgb_6': None,
            'TP_rf_7': None, 'TP_xgb_7': None,
            'TP_rf_8': None, 'TP_xgb_8': None,
            'TP_rf_9': None, 'TP_xgb_9': None,
            'TP_rf_10': None, 'TP_xgb_10': None}

cached_TN = {'TN_rf_1': None, 'TN_xgb_1': None,
            'TN_rf_2': None, 'TN_xgb_2': None,
            'TN_rf_3': None, 'TN_xgb_3': None,
            'TN_rf_4': None, 'TN_xgb_4': None,
            'TN_rf_5': None, 'TN_xgb_5': None,
            'TN_rf_6': None, 'TN_xgb_6': None,
            'TN_rf_7': None, 'TN_xgb_7': None,
            'TN_rf_8': None, 'TN_xgb_8': None,
            'TN_rf_9': None, 'TN_xgb_9': None,
            'TN_rf_10': None, 'TN_xgb_10': None}

cached_FP = {'FP_rf_1': None, 'FP_xgb_1': None,
            'FP_rf_2': None, 'FP_xgb_2': None,
            'FP_rf_3': None, 'FP_xgb_3': None,
            'FP_rf_4': None, 'FP_xgb_4': None,
            'FP_rf_5': None, 'FP_xgb_5': None,
            'FP_rf_6': None, 'FP_xgb_6': None,
            'FP_rf_7': None, 'FP_xgb_7': None,
            'FP_rf_8': None, 'FP_xgb_8': None,
            'FP_rf_9': None, 'FP_xgb_9': None,
            'FP_rf_10': None, 'FP_xgb_10': None}

cached_FN = {'FN_rf_1': None, 'FN_xgb_1': None,
            'FN_rf_2': None, 'FN_xgb_2': None,
            'FN_rf_3': None, 'FN_xgb_3': None,
            'FN_rf_4': None, 'FN_xgb_4': None,
            'FN_rf_5': None, 'FN_xgb_5': None,
            'FN_rf_6': None, 'FN_xgb_6': None,
            'FN_rf_7': None, 'FN_xgb_7': None,
            'FN_rf_8': None, 'FN_xgb_8': None,
            'FN_rf_9': None, 'FN_xgb_9': None,
            'FN_rf_10': None, 'FN_xgb_10': None}

cached_TP_train = {'TP_train_rf_1': None, 'TP_train_xgb_1': None,
                'TP_train_rf_2': None, 'TP_train_xgb_2': None,
                'TP_train_rf_3': None, 'TP_train_xgb_3': None,
                'TP_train_rf_4': None, 'TP_train_xgb_4': None,
                'TP_train_rf_5': None, 'TP_train_xgb_5': None,
                'TP_train_rf_6': None, 'TP_train_xgb_6': None,
                'TP_train_rf_7': None, 'TP_train_xgb_7': None,
                'TP_train_rf_8': None, 'TP_train_xgb_8': None,
                'TP_train_rf_9': None, 'TP_train_xgb_9': None,
                'TP_train_rf_10': None, 'TP_train_xgb_10': None}

cached_TN_train = {'TN_train_rf_1': None, 'TN_train_xgb_1': None,
                'TN_train_rf_2': None, 'TN_train_xgb_2': None,
                'TN_train_rf_3': None, 'TN_train_xgb_3': None,
                'TN_train_rf_4': None, 'TN_train_xgb_4': None,
                'TN_train_rf_5': None, 'TN_train_xgb_5': None,
                'TN_train_rf_6': None, 'TN_train_xgb_6': None,
                'TN_train_rf_7': None, 'TN_train_xgb_7': None,
                'TN_train_rf_8': None, 'TN_train_xgb_8': None,
                'TN_train_rf_9': None, 'TN_train_xgb_9': None,
                'TN_train_rf_10': None, 'TN_train_xgb_10': None}

cached_FP_train = {'FP_train_rf_1': None, 'FP_train_xgb_1': None,
            'FP_train_rf_2': None, 'FP_train_xgb_2': None,
            'FP_train_rf_3': None, 'FP_train_xgb_3': None,
            'FP_train_rf_4': None, 'FP_train_xgb_4': None,
            'FP_train_rf_5': None, 'FP_train_xgb_5': None,
            'FP_train_rf_6': None, 'FP_train_xgb_6': None,
            'FP_train_rf_7': None, 'FP_train_xgb_7': None,
            'FP_train_rf_8': None, 'FP_train_xgb_8': None,
            'FP_train_rf_9': None, 'FP_train_xgb_9': None,
            'FP_train_rf_10': None, 'FP_train_xgb_10': None}

cached_FN_train = {'FN_train_rf_1': None, 'FN_train_xgb_1': None,
            'FN_train_rf_2': None, 'FN_train_xgb_2': None,
            'FN_train_rf_3': None, 'FN_train_xgb_3': None,
            'FN_train_rf_4': None, 'FN_train_xgb_4': None,
            'FN_train_rf_5': None, 'FN_train_xgb_5': None,
            'FN_train_rf_6': None, 'FN_train_xgb_6': None,
            'FN_train_rf_7': None, 'FN_train_xgb_7': None,
            'FN_train_rf_8': None, 'FN_train_xgb_8': None,
            'FN_train_rf_9': None, 'FN_train_xgb_9': None,
            'FN_train_rf_10': None, 'FN_train_xgb_10': None}

# Route untuk halaman utama dengan pagination
@app.route("/")
@app.route("/data1")
def index():
    x = request.args.get('x', 1, type=int)
    # Ambil parameter page dari URL, default ke 1 jika tidak ada
    page = request.args.get('page', 1, type=int)

    # Ambil data dari CSV
    data = read_csv(x)  
    df = data
    per_page = 10
    total_pages = (len(data) + per_page - 1) // per_page  # Total halaman

    # Pilih kolom yang ingin dianalisis
    column_name = "Diagnosed_by_professional"

    # Pisahkan data berdasarkan nilai
    yes_data = df[df[column_name] == "Yes"]
    no_data = df[df[column_name] == "No"]

    # Hitung jumlah data untuk setiap nilai
    yes_count1 = len(yes_data)
    no_count1 = len(no_data)

    # Cek apakah page kurang dari 1 atau lebih dari total_pages
    if page < 1 or page > total_pages:
        return redirect(url_for('index', page=1))

    start_idx = (page - 1) * per_page
    end_idx = min(start_idx + per_page, len(data))  # Hitung end_idx

    # Ambil data untuk halaman saat ini
    page_data = data[start_idx:end_idx]

    return render_template('index100.html', data=page_data, page=page, total_pages=total_pages, x=x, yes_count1 = yes_count1, no_count1 = no_count1)

@app.route("/data2")
def index2():
    x = request.args.get('x', 2, type=int)
    # Ambil parameter page dari URL, default ke 1 jika tidak ada
    page = request.args.get('page', 1, type=int)

    # Ambil data dari CSV
    data = read_csv(x)  
    df = data
    per_page = 10
    total_pages = (len(data) + per_page - 1) // per_page  # Total halaman

    # Pilih kolom yang ingin dianalisis
    column_name = "Diagnosed_by_professional"

    # Pisahkan data berdasarkan nilai
    yes_data = df[df[column_name] == "Yes"]
    no_data = df[df[column_name] == "No"]

    # Hitung jumlah data untuk setiap nilai
    yes_count2 = len(yes_data)
    no_count2 = len(no_data)

    # Cek apakah page kurang dari 1 atau lebih dari total_pages
    if page < 1 or page > total_pages:
        return redirect(url_for('index2', page=1))

    start_idx = (page - 1) * per_page
    end_idx = min(start_idx + per_page, len(data))  # Hitung end_idx

    # Ambil data untuk halaman saat ini
    page_data = data[start_idx:end_idx]

    return render_template('index3k(1).html', data=page_data, page=page, total_pages=total_pages, x=x, yes_count2 = yes_count2, no_count2 = no_count2)

@app.route("/data3")
def index3():
    x = request.args.get('x', 3, type=int)
    # Ambil parameter page dari URL, default ke 1 jika tidak ada
    page = request.args.get('page', 1, type=int)

    # Ambil data dari CSV
    data = read_csv(x)
    df = data  
    per_page = 10
    total_pages = (len(data) + per_page - 1) // per_page  # Total halaman

    # Pilih kolom yang ingin dianalisis
    column_name = "Diagnosed_by_professional"

    # Pisahkan data berdasarkan nilai
    yes_data = df[df[column_name] == "Yes"]
    no_data = df[df[column_name] == "No"]

    # Hitung jumlah data untuk setiap nilai
    yes_count3 = len(yes_data)
    no_count3 = len(no_data)

    # Cek apakah page kurang dari 1 atau lebih dari total_pages
    if page < 1 or page > total_pages:
        return redirect(url_for('index3', page=1))

    start_idx = (page - 1) * per_page
    end_idx = min(start_idx + per_page, len(data))  # Hitung end_idx

    # Ambil data untuk halaman saat ini
    page_data = data[start_idx:end_idx]

    return render_template('index3k(2).html', data=page_data, page=page, total_pages=total_pages, x=x, yes_count3 = yes_count3, no_count3 = no_count3)

@app.route("/data4")
def index4():
    x = request.args.get('x', 4, type=int)
    # Ambil parameter page dari URL, default ke 1 jika tidak ada
    page = request.args.get('page', 1, type=int)

    # Ambil data dari CSV
    data = read_csv(x) 
    df = data 
    per_page = 10
    total_pages = (len(data) + per_page - 1) // per_page  # Total halaman

    # Pilih kolom yang ingin dianalisis
    column_name = "Diagnosed_by_professional"

    # Pisahkan data berdasarkan nilai
    yes_data = df[df[column_name] == "Yes"]
    no_data = df[df[column_name] == "No"]

    # Hitung jumlah data untuk setiap nilai
    yes_count4 = len(yes_data)
    no_count4 = len(no_data)

    # Cek apakah page kurang dari 1 atau lebih dari total_pages
    if page < 1 or page > total_pages:
        return redirect(url_for('index4', page=1))

    start_idx = (page - 1) * per_page
    end_idx = min(start_idx + per_page, len(data))  # Hitung end_idx

    # Ambil data untuk halaman saat ini
    page_data = data[start_idx:end_idx]

    return render_template('index3k(3).html', data=page_data, page=page, total_pages=total_pages, x=x, yes_count4 = yes_count4, no_count4 = no_count4)

@app.route("/data5")
def index5():
    x = request.args.get('x', 5, type=int)
    # Ambil parameter page dari URL, default ke 1 jika tidak ada
    page = request.args.get('page', 1, type=int)

    # Ambil data dari CSV
    data = read_csv(x)
    df = data  
    per_page = 10
    total_pages = (len(data) + per_page - 1) // per_page  # Total halaman

    # Pilih kolom yang ingin dianalisis
    column_name = "Diagnosed_by_professional"

    # Pisahkan data berdasarkan nilai
    yes_data = df[df[column_name] == "Yes"]
    no_data = df[df[column_name] == "No"]

    # Hitung jumlah data untuk setiap nilai
    yes_count5 = len(yes_data)
    no_count5 = len(no_data)

    # Cek apakah page kurang dari 1 atau lebih dari total_pages
    if page < 1 or page > total_pages:
        return redirect(url_for('index5', page=1))

    start_idx = (page - 1) * per_page
    end_idx = min(start_idx + per_page, len(data))  # Hitung end_idx

    # Ambil data untuk halaman saat ini
    page_data = data[start_idx:end_idx]

    return render_template('index5k(1).html', data=page_data, page=page, total_pages=total_pages, x=x, yes_count5 = yes_count5, no_count5 = no_count5) 

@app.route("/data6")
def index6():
    x = request.args.get('x', 6, type=int)
    # Ambil parameter page dari URL, default ke 1 jika tidak ada
    page = request.args.get('page', 1, type=int)

    # Ambil data dari CSV
    data = read_csv(x) 
    df = data 
    per_page = 10
    total_pages = (len(data) + per_page - 1) // per_page  # Total halaman

    # Pilih kolom yang ingin dianalisis
    column_name = "Diagnosed_by_professional"

    # Pisahkan data berdasarkan nilai
    yes_data = df[df[column_name] == "Yes"]
    no_data = df[df[column_name] == "No"]

    # Hitung jumlah data untuk setiap nilai
    yes_count6 = len(yes_data)
    no_count6 = len(no_data)

    # Cek apakah page kurang dari 1 atau lebih dari total_pages
    if page < 1 or page > total_pages:
        return redirect(url_for('index6', page=1))

    start_idx = (page - 1) * per_page
    end_idx = min(start_idx + per_page, len(data))  # Hitung end_idx

    # Ambil data untuk halaman saat ini
    page_data = data[start_idx:end_idx]

    return render_template('index5k(2).html', data=page_data, page=page, total_pages=total_pages, x=x, yes_count6 = yes_count6, no_count6 = no_count6) 

@app.route("/data7")
def index7():
    x = request.args.get('x', 7, type=int)
    # Ambil parameter page dari URL, default ke 1 jika tidak ada
    page = request.args.get('page', 1, type=int)

    # Ambil data dari CSV
    data = read_csv(x)  
    df = data
    per_page = 10
    total_pages = (len(data) + per_page - 1) // per_page  # Total halaman

    # Pilih kolom yang ingin dianalisis
    column_name = "Diagnosed_by_professional"

    # Pisahkan data berdasarkan nilai
    yes_data = df[df[column_name] == "Yes"]
    no_data = df[df[column_name] == "No"]

    # Hitung jumlah data untuk setiap nilai
    yes_count7 = len(yes_data)
    no_count7 = len(no_data)

    # Cek apakah page kurang dari 1 atau lebih dari total_pages
    if page < 1 or page > total_pages:
        return redirect(url_for('index7', page=1))

    start_idx = (page - 1) * per_page
    end_idx = min(start_idx + per_page, len(data))  # Hitung end_idx

    # Ambil data untuk halaman saat ini
    page_data = data[start_idx:end_idx]

    return render_template('index5k(3).html', data=page_data, page=page, total_pages=total_pages, x=x, yes_count7 = yes_count7, no_count7 = no_count7) 

@app.route("/data8")
def index8():
    x = request.args.get('x', 8, type=int)
    # Ambil parameter page dari URL, default ke 1 jika tidak ada
    page = request.args.get('page', 1, type=int)

    # Ambil data dari CSV
    data = read_csv(x) 
    df = data 
    per_page = 10
    total_pages = (len(data) + per_page - 1) // per_page  # Total halaman

    # Pilih kolom yang ingin dianalisis
    column_name = "Diagnosed_by_professional"

    # Pisahkan data berdasarkan nilai
    yes_data = df[df[column_name] == "Yes"]
    no_data = df[df[column_name] == "No"]

    # Hitung jumlah data untuk setiap nilai
    yes_count8 = len(yes_data)
    no_count8 = len(no_data)

    # Cek apakah page kurang dari 1 atau lebih dari total_pages
    if page < 1 or page > total_pages:
        return redirect(url_for('index8', page=1))

    start_idx = (page - 1) * per_page
    end_idx = min(start_idx + per_page, len(data))  # Hitung end_idx

    # Ambil data untuk halaman saat ini
    page_data = data[start_idx:end_idx]

    return render_template('index10k(1).html', data=page_data, page=page, total_pages=total_pages, x=x, yes_count8 = yes_count8, no_count8 = no_count8) 

@app.route("/data9")
def index9():
    x = request.args.get('x', 9, type=int)
    # Ambil parameter page dari URL, default ke 1 jika tidak ada
    page = request.args.get('page', 1, type=int)

    # Ambil data dari CSV
    data = read_csv(x) 
    df = data 
    per_page = 10
    total_pages = (len(data) + per_page - 1) // per_page  # Total halaman

    # Pilih kolom yang ingin dianalisis
    column_name = "Diagnosed_by_professional"

    # Pisahkan data berdasarkan nilai
    yes_data = df[df[column_name] == "Yes"]
    no_data = df[df[column_name] == "No"]

    # Hitung jumlah data untuk setiap nilai
    yes_count9 = len(yes_data)
    no_count9 = len(no_data)

    # Cek apakah page kurang dari 1 atau lebih dari total_pages
    if page < 1 or page > total_pages:
        return redirect(url_for('index9', page=1))

    start_idx = (page - 1) * per_page
    end_idx = min(start_idx + per_page, len(data))  # Hitung end_idx

    # Ambil data untuk halaman saat ini
    page_data = data[start_idx:end_idx]

    return render_template('index10k(2).html', data=page_data, page=page, total_pages=total_pages, x=x, yes_count9 = yes_count9, no_count9 = no_count9) 

@app.route("/data10")
def index10():
    x = request.args.get('x', 10, type=int)
    # Ambil parameter page dari URL, default ke 1 jika tidak ada
    page = request.args.get('page', 1, type=int)

    # Ambil data dari CSV
    data = read_csv(x) 
    df = data 
    per_page = 10
    total_pages = (len(data) + per_page - 1) // per_page  # Total halaman

    # Pilih kolom yang ingin dianalisis
    column_name = "Diagnosed_by_professional"

    # Pisahkan data berdasarkan nilai
    yes_data = df[df[column_name] == "Yes"]
    no_data = df[df[column_name] == "No"]

    # Hitung jumlah data untuk setiap nilai
    yes_count10 = len(yes_data)
    no_count10 = len(no_data)

    # Cek apakah page kurang dari 1 atau lebih dari total_pages
    if page < 1 or page > total_pages:
        return redirect(url_for('index9', page=1))

    start_idx = (page - 1) * per_page
    end_idx = min(start_idx + per_page, len(data))  # Hitung end_idx

    # Ambil data untuk halaman saat ini
    page_data = data[start_idx:end_idx]

    return render_template('index10k(3).html', data=page_data, page=page, total_pages=total_pages, x=x, yes_count10 = yes_count10, no_count10 = no_count10) 

# Route untuk menjalankan model dan mendapatkan hasil akurasi
@app.route('/run-models', methods=['GET'])
def run_models():
    choice = int(request.args.get('choice', 0))

    # Pilih model berdasarkan choice
    if choice == 1:
        model_function = Main100.train_models
        cache_key_rf = 'accuracy_rf_1'
        cache_key_xgb = 'accuracy_xgb_1'
        cache_key_rf_precision = 'precision_rf_1'
        cache_key_xgb_precision = 'precision_xgb_1'
        cache_key_rf_recall = 'recall_rf_1'
        cache_key_xgb_recall = 'recall_xgb_1'
        cache_key_rf_f1 = 'f1_rf_1'
        cache_key_xgb_f1 = 'f1_xgb_1'
        cache_key_rf_TP = 'TP_rf_1'
        cache_key_rf_TN = 'TN_rf_1'
        cache_key_rf_FP = 'FP_rf_1'
        cache_key_rf_FN = 'FN_rf_1'
        cache_key_xgb_TP = 'TP_xgb_1'
        cache_key_xgb_TN = 'TN_xgb_1'
        cache_key_xgb_FP = 'FP_xgb_1'
        cache_key_xgb_FN = 'FN_xgb_1'
        cache_key_rf_TP_train = 'TP_train_rf_1'
        cache_key_rf_TN_train = 'TN_train_rf_1'
        cache_key_rf_FP_train = 'FP_train_rf_1'
        cache_key_rf_FN_train = 'FN_train_rf_1'
        cache_key_xgb_TP_train = 'TP_train_xgb_1'
        cache_key_xgb_TN_train = 'TN_train_xgb_1'
        cache_key_xgb_FP_train = 'FP_train_xgb_1'
        cache_key_xgb_FN_train = 'FN_train_xgb_1'
    elif choice == 2:
        model_function = Main3k1.train_models
        cache_key_rf = 'accuracy_rf_2'
        cache_key_xgb = 'accuracy_xgb_2'
        cache_key_rf_precision = 'precision_rf_2'
        cache_key_xgb_precision = 'precision_xgb_2'
        cache_key_rf_recall = 'recall_rf_2'
        cache_key_xgb_recall = 'recall_xgb_2'
        cache_key_rf_f1 = 'f1_rf_2'
        cache_key_xgb_f1 = 'f1_xgb_2'
        cache_key_rf_TP = 'TP_rf_2'
        cache_key_rf_TN = 'TN_rf_2'
        cache_key_rf_FP = 'FP_rf_2'
        cache_key_rf_FN = 'FN_rf_2'
        cache_key_xgb_TP = 'TP_xgb_2'
        cache_key_xgb_TN = 'TN_xgb_2'
        cache_key_xgb_FP = 'FP_xgb_2'
        cache_key_xgb_FN = 'FN_xgb_2'
        cache_key_rf_TP_train = 'TP_train_rf_2'
        cache_key_rf_TN_train = 'TN_train_rf_2'
        cache_key_rf_FP_train = 'FP_train_rf_2'
        cache_key_rf_FN_train = 'FN_train_rf_2'
        cache_key_xgb_TP_train = 'TP_train_xgb_2'
        cache_key_xgb_TN_train = 'TN_train_xgb_2'
        cache_key_xgb_FP_train = 'FP_train_xgb_2'
        cache_key_xgb_FN_train = 'FN_train_xgb_2'
    elif choice == 3:
        model_function = Main3k2.train_models
        cache_key_rf = 'accuracy_rf_3'
        cache_key_xgb = 'accuracy_xgb_3'
        cache_key_rf_precision = 'precision_rf_3'
        cache_key_xgb_precision = 'precision_xgb_3'
        cache_key_rf_recall = 'recall_rf_3'
        cache_key_xgb_recall = 'recall_xgb_3'
        cache_key_rf_f1 = 'f1_rf_3'
        cache_key_xgb_f1 = 'f1_xgb_3'
        cache_key_rf_TP = 'TP_rf_3'
        cache_key_rf_TN = 'TN_rf_3'
        cache_key_rf_FP = 'FP_rf_3'
        cache_key_rf_FN = 'FN_rf_3'
        cache_key_xgb_TP = 'TP_xgb_3'
        cache_key_xgb_TN = 'TN_xgb_3'
        cache_key_xgb_FP = 'FP_xgb_3'
        cache_key_xgb_FN = 'FN_xgb_3'
        cache_key_rf_TP_train = 'TP_train_rf_3'
        cache_key_rf_TN_train = 'TN_train_rf_3'
        cache_key_rf_FP_train = 'FP_train_rf_3'
        cache_key_rf_FN_train = 'FN_train_rf_3'
        cache_key_xgb_TP_train = 'TP_train_xgb_3'
        cache_key_xgb_TN_train = 'TN_train_xgb_3'
        cache_key_xgb_FP_train = 'FP_train_xgb_3'
        cache_key_xgb_FN_train = 'FN_train_xgb_3'
    elif choice == 4:
        model_function = Main3k3.train_models
        cache_key_rf = 'accuracy_rf_4'
        cache_key_xgb = 'accuracy_xgb_4'
        cache_key_rf_precision = 'precision_rf_4'
        cache_key_xgb_precision = 'precision_xgb_4'
        cache_key_rf_recall = 'recall_rf_4'
        cache_key_xgb_recall = 'recall_xgb_4'
        cache_key_rf_f1 = 'f1_rf_4'
        cache_key_xgb_f1 = 'f1_xgb_4'
        cache_key_rf_TP = 'TP_rf_4'
        cache_key_rf_TN = 'TN_rf_4'
        cache_key_rf_FP = 'FP_rf_4'
        cache_key_rf_FN = 'FN_rf_4'
        cache_key_xgb_TP = 'TP_xgb_4'
        cache_key_xgb_TN = 'TN_xgb_4'
        cache_key_xgb_FP = 'FP_xgb_4'
        cache_key_xgb_FN = 'FN_xgb_4'
        cache_key_rf_TP_train = 'TP_train_rf_4'
        cache_key_rf_TN_train = 'TN_train_rf_4'
        cache_key_rf_FP_train = 'FP_train_rf_4'
        cache_key_rf_FN_train = 'FN_train_rf_4'
        cache_key_xgb_TP_train = 'TP_train_xgb_4'
        cache_key_xgb_TN_train = 'TN_train_xgb_4'
        cache_key_xgb_FP_train = 'FP_train_xgb_4'
        cache_key_xgb_FN_train = 'FN_train_xgb_4'
    elif choice == 5:
        model_function = Main5k1.train_models
        cache_key_rf = 'accuracy_rf_5'
        cache_key_xgb = 'accuracy_xgb_5'
        cache_key_rf_precision = 'precision_rf_5'
        cache_key_xgb_precision = 'precision_xgb_5'
        cache_key_rf_recall = 'recall_rf_5'
        cache_key_xgb_recall = 'recall_xgb_5'
        cache_key_rf_f1 = 'f1_rf_5'
        cache_key_xgb_f1 = 'f1_xgb_5'
        cache_key_rf_TP = 'TP_rf_5'
        cache_key_rf_TN = 'TN_rf_5'
        cache_key_rf_FP = 'FP_rf_5'
        cache_key_rf_FN = 'FN_rf_5'
        cache_key_xgb_TP = 'TP_xgb_5'
        cache_key_xgb_TN = 'TN_xgb_5'
        cache_key_xgb_FP = 'FP_xgb_5'
        cache_key_xgb_FN = 'FN_xgb_5'
        cache_key_rf_TP_train = 'TP_train_rf_5'
        cache_key_rf_TN_train = 'TN_train_rf_5'
        cache_key_rf_FP_train = 'FP_train_rf_5'
        cache_key_rf_FN_train = 'FN_train_rf_5'
        cache_key_xgb_TP_train = 'TP_train_xgb_5'
        cache_key_xgb_TN_train = 'TN_train_xgb_5'
        cache_key_xgb_FP_train = 'FP_train_xgb_5'
        cache_key_xgb_FN_train = 'FN_train_xgb_5'
    elif choice == 6:
        model_function = Main5k2.train_models
        cache_key_rf = 'accuracy_rf_6'
        cache_key_xgb = 'accuracy_xgb_6'
        cache_key_rf_precision = 'precision_rf_6'
        cache_key_xgb_precision = 'precision_xgb_6'
        cache_key_rf_recall = 'recall_rf_6'
        cache_key_xgb_recall = 'recall_xgb_6'
        cache_key_rf_f1 = 'f1_rf_6'
        cache_key_xgb_f1 = 'f1_xgb_6'
        cache_key_rf_TP = 'TP_rf_6'
        cache_key_rf_TN = 'TN_rf_6'
        cache_key_rf_FP = 'FP_rf_6'
        cache_key_rf_FN = 'FN_rf_6'
        cache_key_xgb_TP = 'TP_xgb_6'
        cache_key_xgb_TN = 'TN_xgb_6'
        cache_key_xgb_FP = 'FP_xgb_6'
        cache_key_xgb_FN = 'FN_xgb_6'
        cache_key_rf_TP_train = 'TP_train_rf_6'
        cache_key_rf_TN_train = 'TN_train_rf_6'
        cache_key_rf_FP_train = 'FP_train_rf_6'
        cache_key_rf_FN_train = 'FN_train_rf_6'
        cache_key_xgb_TP_train = 'TP_train_xgb_6'
        cache_key_xgb_TN_train = 'TN_train_xgb_6'
        cache_key_xgb_FP_train = 'FP_train_xgb_6'
        cache_key_xgb_FN_train = 'FN_train_xgb_6'
    elif choice == 7:
        model_function = Main5k3.train_models
        cache_key_rf = 'accuracy_rf_7'
        cache_key_xgb = 'accuracy_xgb_7'
        cache_key_rf_precision = 'precision_rf_7'
        cache_key_xgb_precision = 'precision_xgb_7'
        cache_key_rf_recall = 'recall_rf_7'
        cache_key_xgb_recall = 'recall_xgb_7'
        cache_key_rf_f1 = 'f1_rf_7'
        cache_key_xgb_f1 = 'f1_xgb_7'
        cache_key_rf_TP = 'TP_rf_7'
        cache_key_rf_TN = 'TN_rf_7'
        cache_key_rf_FP = 'FP_rf_7'
        cache_key_rf_FN = 'FN_rf_7'
        cache_key_xgb_TP = 'TP_xgb_7'
        cache_key_xgb_TN = 'TN_xgb_7'
        cache_key_xgb_FP = 'FP_xgb_7'
        cache_key_xgb_FN = 'FN_xgb_7'
        cache_key_rf_TP_train = 'TP_train_rf_7'
        cache_key_rf_TN_train = 'TN_train_rf_7'
        cache_key_rf_FP_train = 'FP_train_rf_7'
        cache_key_rf_FN_train = 'FN_train_rf_7'
        cache_key_xgb_TP_train = 'TP_train_xgb_7'
        cache_key_xgb_TN_train = 'TN_train_xgb_7'
        cache_key_xgb_FP_train = 'FP_train_xgb_7'
        cache_key_xgb_FN_train = 'FN_train_xgb_7'
    elif choice == 8:
        model_function = Main10k1.train_models
        cache_key_rf = 'accuracy_rf_8'
        cache_key_xgb = 'accuracy_xgb_8'
        cache_key_rf_precision = 'precision_rf_8'
        cache_key_xgb_precision = 'precision_xgb_8'
        cache_key_rf_recall = 'recall_rf_8'
        cache_key_xgb_recall = 'recall_xgb_8'
        cache_key_rf_f1 = 'f1_rf_8'
        cache_key_xgb_f1 = 'f1_xgb_8'
        cache_key_rf_TP = 'TP_rf_8'
        cache_key_rf_TN = 'TN_rf_8'
        cache_key_rf_FP = 'FP_rf_8'
        cache_key_rf_FN = 'FN_rf_8'
        cache_key_xgb_TP = 'TP_xgb_8'
        cache_key_xgb_TN = 'TN_xgb_8'
        cache_key_xgb_FP = 'FP_xgb_8'
        cache_key_xgb_FN = 'FN_xgb_8'
        cache_key_rf_TP_train = 'TP_train_rf_8'
        cache_key_rf_TN_train = 'TN_train_rf_8'
        cache_key_rf_FP_train = 'FP_train_rf_8'
        cache_key_rf_FN_train = 'FN_train_rf_8'
        cache_key_xgb_TP_train = 'TP_train_xgb_8'
        cache_key_xgb_TN_train = 'TN_train_xgb_8'
        cache_key_xgb_FP_train = 'FP_train_xgb_8'
        cache_key_xgb_FN_train = 'FN_train_xgb_8'
    elif choice == 9:
        model_function = Main10k2.train_models
        cache_key_rf = 'accuracy_rf_9'
        cache_key_xgb = 'accuracy_xgb_9'
        cache_key_rf_precision = 'precision_rf_9'
        cache_key_xgb_precision = 'precision_xgb_9'
        cache_key_rf_recall = 'recall_rf_9'
        cache_key_xgb_recall = 'recall_xgb_9'
        cache_key_rf_f1 = 'f1_rf_9'
        cache_key_xgb_f1 = 'f1_xgb_9'
        cache_key_rf_TP = 'TP_rf_9'
        cache_key_rf_TN = 'TN_rf_9'
        cache_key_rf_FP = 'FP_rf_9'
        cache_key_rf_FN = 'FN_rf_9'
        cache_key_xgb_TP = 'TP_xgb_9'
        cache_key_xgb_TN = 'TN_xgb_9'
        cache_key_xgb_FP = 'FP_xgb_9'
        cache_key_xgb_FN = 'FN_xgb_9'
        cache_key_rf_TP_train = 'TP_train_rf_9'
        cache_key_rf_TN_train = 'TN_train_rf_9'
        cache_key_rf_FP_train = 'FP_train_rf_9'
        cache_key_rf_FN_train = 'FN_train_rf_9'
        cache_key_xgb_TP_train = 'TP_train_xgb_9'
        cache_key_xgb_TN_train = 'TN_train_xgb_9'
        cache_key_xgb_FP_train = 'FP_train_xgb_9'
        cache_key_xgb_FN_train = 'FN_train_xgb_9'
    elif choice == 10:
        model_function = Main10k3.train_models
        cache_key_rf = 'accuracy_rf_10'
        cache_key_xgb = 'accuracy_xgb_10'
        cache_key_rf_precision = 'precision_rf_10'
        cache_key_xgb_precision = 'precision_xgb_10'
        cache_key_rf_recall = 'recall_rf_10'
        cache_key_xgb_recall = 'recall_xgb_10'
        cache_key_rf_f1 = 'f1_rf_10'
        cache_key_xgb_f1 = 'f1_xgb_10'
        cache_key_rf_TP = 'TP_rf_10'
        cache_key_rf_TN = 'TN_rf_10'
        cache_key_rf_FP = 'FP_rf_10'
        cache_key_rf_FN = 'FN_rf_10'
        cache_key_xgb_TP = 'TP_xgb_10'
        cache_key_xgb_TN = 'TN_xgb_10'
        cache_key_xgb_FP = 'FP_xgb_10'
        cache_key_xgb_FN = 'FN_xgb_10'
        cache_key_rf_TP_train = 'TP_train_rf_10'
        cache_key_rf_TN_train = 'TN_train_rf_10'
        cache_key_rf_FP_train = 'FP_train_rf_10'
        cache_key_rf_FN_train = 'FN_train_rf_10'
        cache_key_xgb_TP_train = 'TP_train_xgb_10'
        cache_key_xgb_TN_train = 'TN_train_xgb_10'
        cache_key_xgb_FP_train = 'FP_train_xgb_10'
        cache_key_xgb_FN_train = 'FN_train_xgb_10'
    else:
        return jsonify({"error": "Invalid dataset choice"}), 400

    # Jalankan model hanya jika akurasi belum ada di cache
    if cached_accuracy[cache_key_rf] is None or cached_accuracy[cache_key_xgb] is None:
        accuracy_rf, accuracy_xgb, precision_rf, precision_xgb, recall_rf, recall_xgb, f1_rf, f1_xgb, TN_xgb, FP_xgb, FN_xgb, TP_xgb, TN_rf, FP_rf, FN_rf, TP_rf, TN_train_rf, FP_train_rf, FN_train_rf, TP_train_rf, TN_train_xgb, FP_train_xgb, FN_train_xgb, TP_train_xgb = model_function()
        cached_accuracy[cache_key_rf] = accuracy_rf
        cached_accuracy[cache_key_xgb] = accuracy_xgb
        cached_precision[cache_key_rf_precision] = precision_rf
        cached_precision[cache_key_xgb_precision] = precision_xgb
        cached_recall[cache_key_rf_recall] = recall_rf
        cached_recall[cache_key_xgb_recall] = recall_xgb
        cached_f1[cache_key_rf_f1] = f1_rf
        cached_f1[cache_key_xgb_f1] = f1_xgb
        cached_TP[cache_key_rf_TP] = TP_rf
        cached_TN[cache_key_rf_TN] = TN_rf
        cached_FP[cache_key_rf_FP] = FP_rf
        cached_FN[cache_key_rf_FN] = FN_rf
        cached_TP[cache_key_xgb_TP] = TP_xgb
        cached_TN[cache_key_xgb_TN] = TN_xgb
        cached_FP[cache_key_xgb_FP] = FP_xgb
        cached_FN[cache_key_xgb_FN] = FN_xgb
        cached_TP_train[cache_key_rf_TP_train] = TP_train_rf
        cached_TN_train[cache_key_rf_TN_train] = TN_train_rf
        cached_FP_train[cache_key_rf_FP_train] = FP_train_rf
        cached_FN_train[cache_key_rf_FN_train] = FN_train_rf
        cached_TP_train[cache_key_xgb_TP_train] = TP_train_xgb
        cached_TN_train[cache_key_xgb_TN_train] = TN_train_xgb
        cached_FP_train[cache_key_xgb_FP_train] = FP_train_xgb
        cached_FN_train[cache_key_xgb_FN_train] = FN_train_xgb


    #coba itung Comparison    
    if cached_accuracy[cache_key_rf] > cached_accuracy[cache_key_xgb]:
        result_text = "Hasil dari perbandingan grafik, menyatakan bahwa model Random Forest lebih unggul dibandingkan model XGBoost dalam hal klasifikasi gangguan kesehatan mental.";
    elif cached_accuracy[cache_key_rf] < cached_accuracy[cache_key_xgb] :
        result_text = "Hasil dari perbandingan grafik, menyatakan bahwa model XGBoost lebih unggul dibandingkan model Random Forest dalam hal klasifikasi gangguan kesehatan mental.";
    elif cached_accuracy[cache_key_rf] == cached_accuracy[cache_key_xgb] :
        result_text = "Hasil dari perbandingan grafik, menyatakan bahwa model Random Forest dan model XGBoost sama unggul dalam hal klasifikasi gangguan kesehatan mental.";

    print("Result Text:", result_text)

    
    return jsonify({
        "accuracy_rf" : cached_accuracy[cache_key_rf],
        "accuracy_xgb" : cached_accuracy[cache_key_xgb],
        "precision_rf" : cached_precision[cache_key_rf_precision],
        "precision_xgb" : cached_precision[cache_key_xgb_precision],
        "recall_rf" : cached_recall[cache_key_rf_recall],
        "recall_xgb" : cached_recall[cache_key_xgb_recall],
        "f1_rf": cached_f1[cache_key_rf_f1],
        "f1_xgb": cached_f1[cache_key_xgb_f1],
        "TP_rf" : cached_TP[cache_key_rf_TP],
        "TP_xgb" : cached_TP[cache_key_xgb_TP],
        "TN_rf" : cached_TN[cache_key_rf_TN],
        "TN_xgb" : cached_TN[cache_key_xgb_TN],
        "FP_rf" : cached_FP[cache_key_rf_FP],
        "FP_xgb" : cached_FP[cache_key_xgb_FP],
        "FN_rf" : cached_FN[cache_key_rf_FN],
        "FN_xgb" : cached_FN[cache_key_xgb_FN],
        "TP_train_rf" : cached_TP_train[cache_key_rf_TP_train],
        "TP_train_xgb" : cached_TP_train[cache_key_xgb_TP_train],
        "TN_train_rf" : cached_TN_train[cache_key_rf_TN_train],
        "TN_train_xgb" : cached_TN_train[cache_key_xgb_TN_train],
        "FP_train_rf" : cached_FP_train[cache_key_rf_FP_train],
        "FP_train_xgb" : cached_FP_train[cache_key_xgb_FP_train],
        "FN_train_rf" : cached_FN_train[cache_key_rf_FN_train],
        "FN_train_xgb" : cached_FN_train[cache_key_xgb_FN_train],
        "Yes_train" : cached_TP_train[cache_key_rf_TP_train] + cached_FN_train[cache_key_rf_FN_train],
        "No_train" : cached_TN_train[cache_key_xgb_TN_train] + cached_FP_train[cache_key_xgb_FP_train],
        "Yes" : cached_TP[cache_key_rf_TP] + cached_FN[cache_key_rf_FN],
        "No" : cached_TN[cache_key_xgb_TN] + cached_FP[cache_key_xgb_FP],
        "result_text": result_text
    })

@app.route('/style.css')
def css():
    return send_from_directory('templates', 'style.css')

@app.route('/script.js')
def javascript():
    return send_from_directory('templates', 'script.js')

if __name__ == '__main__':
    app.run(debug=True)




