# Siaga Malaria Nusantara: Deteksi Dini Malaria Berbasis AI 🩺🔬

Selamat datang di Siaga Malaria Nusantara! 🌟 Proyek ini menghadirkan solusi cerdas untuk mendeteksi malaria secara dini menggunakan _Convolutional Neural Network_ (CNN) yang terintegrasi dalam prototipe web telemedicine modern. Bayangkan sebuah alat yang membantu dokter dan tenaga medis di pelosok Nusantara untuk mendiagnosis malaria dengan cepat dan akurat! 🚀


## Pendahuluan

Malaria tetap menjadi ancaman di banyak wilayah tropis, termasuk Indonesia. Penyakit yang ditularkan oleh nyamuk ini membutuhkan deteksi cepat untuk menyelamatkan nyawa dan mencegah penyebaran lebih lanjut. 🦟 Di sinilah Siaga Malaria Nusantara berperan: kami memanfaatkan kecerdasan buatan untuk menganalisis gambar mikroskopis sel darah dan menentukan apakah seseorang terinfeksi malaria. 🔍

Dengan dua pendekatan pemodelan, yaitu _Custom CNN_ dan _Transfer Learning_ berbasis _EfficientNetB0_, kami membangun sistem yang tidak hanya akurat, tetapi juga praktis untuk digunakan melalui aplikasi web interaktif. 💻


## Struktur Folder 📂

Proyek ini tersusun rapi dalam beberapa folder utama
- `data/`: Berisi referensi ke dataset malaria.
- `models`: Menyimpan model yang telah dilatih dalam berbagai format, yaitu _SavedModel_, _Pickle_, _TFLite_ (standar dan kuantisasi), serta _TFJS_ untuk kebutuhan web. 🗃️
- `notebooks`: Koleksi _Jupyter Notebook_ untuk analisis data, pelatihan model, dan evaluasi. 📓
- `web_app`: Berisi referensi ke _prototipe_ aplikasi web yang siap digunakan untuk prediksi malaria. 🌐


## Exploratory Data Analysis (EDA) 📊

Sebelum melangkah ke pemodelan, kami menggali dataset malaria untuk memahami karakteristiknya. Dataset ini berisi gambar sel darah dengan dua label: _Parasitized_ (terinfeksi malaria) dan _Uninfected_ (sehat). ✅

Langkah-langkah _Exploratory Data Analysis_ (EDA) yang dilakukan:

- **Pemeriksaan gambar _corrupt_**: Seluruh gambar pada direktori _train_, _valid_, dan _test_ diperiksa satu per satu untuk memastikan tidak ada file rusak atau tidak terbaca. Hasilnya, tidak ditemukan gambar _corrupt_. ✔️

- **Distribusi kelas**: Jumlah gambar di masing-masing kelas seimbang, baik untuk data _train_, _valid_, maupun _test_. Hal ini memastikan model tidak bias terhadap salah satu kelas. ⚖️

- **Visualisasi sampel gambar**: Beberapa contoh gambar dari masing-masing kelas divisualisasikan untuk melihat perbedaan visual antara sel darah yang terinfeksi dan tidak terinfeksi. 🖼️

- **Ukuran gambar**: Semua gambar memiliki ukuran yang sama, yaitu 224x224 piksel, sehingga tidak diperlukan proses _resize_ tambahan sebelum _training_. 📏

- **Pemeriksaan duplikasi gambar**: Data _train_, _valid_, dan _test_ diperiksa untuk memastikan tidak ada gambar yang identik secara konten (bukan hanya nama file). Hasilnya, tidak ditemukan gambar duplikat di ketiga subset tersebut. 🔎

EDA ini memastikan kualitas data sebelum proses pemodelan dimulai, sehingga hasil pemodelan menjadi lebih akurat dan dapat diandalkan. 🌟


## Pemodelan 🤖

Kami mengembangkan dua model utama untuk tugas deteksi malaria. Berikut rinciannya:

**A. Struktur Direktori**
- TRAIN_DIR = '/kaggle/input/malaria/Malaria Dataset/train'
- VALID_DIR = '/kaggle/input/malaria/Malaria Dataset/valid'
- TEST_DIR = '/kaggle/input/malaria/Malaria Dataset/test'

**B. Parameter**
- IMG_SIZE = (224, 224)
- BATCH_SIZE = 32
- EPOCHS_CNN = 50
- EPOCHS_EFFNET = 50

### Custom CNN

Model CNN kustom ini dirancang untuk mempelajari fitur dari awal (from scratch) berdasarkan dataset yang diberikan. Arsitekturnya terdiri dari tiga blok konvolusi yang bertujuan untuk mengekstraksi fitur dari gambar, diikuti oleh lapisan klasifikasi untuk menghasilkan prediksi biner. 🧠

Setiap blok konvolusi memiliki struktur sebagai berikut:
- **Lapisan Konvolusi**: Menggunakan _Conv2D_ dengan aktivasi ReLU untuk mengekstraksi fitur. Ukuran kernel diatur pada (3,3) dengan _padding 'same'_ agar dimensi input tetap terjaga.
- **Lapisan _Pooling_**: Menggunakan _MaxPooling2D_ dengan ukuran (2,2) untuk mengurangi dimensi spasial dan mempertahankan fitur penting. 📉

Rincian setiap blok:
- **Blok 1**: Lapisan _Conv2D_ memiliki 32 filter, diikuti oleh _MaxPooling2D_.
- **Blok 2**: Lapisan _Conv2D_ memiliki 64 filter, diikuti oleh _MaxPooling2D_.
- **Blok 3**: Lapisan _Conv2D_ memiliki 128 filter, diikuti oleh _MaxPooling2D_.

Setelah proses konvolusi selesai, fitur yang dihasilkan diratakan (flattened) menjadi vektor satu dimensi. Vektor ini kemudian diproses melalui lapisan klasifikasi:
- **Lapisan _Dense_**: Terdiri dari 128 unit dengan aktivasi ReLU untuk mempelajari kombinasi fitur yang lebih kompleks.
- **_Dropout_**: Diterapkan dengan tingkat 0,5 untuk mengurangi risiko _overfitting_ dengan secara acak menonaktifkan sebagian neuron selama pelatihan. 🛡️
- **Lapisan _Output_**: Lapisan _Dense_ dengan 1 unit dan aktivasi _sigmoid_ untuk menghasilkan probabilitas klasifikasi biner (terinfeksi malaria atau tidak).

Model ini dikompilasi dengan konfigurasi berikut:
- **_Optimizer_**: _Adam_ dengan _learning rate_ 0,001, yang dipilih karena kemampuannya untuk menyesuaikan langkah pembelajaran secara adaptif.
- **_Loss_**: _binary_crossentropy_, sesuai untuk tugas klasifikasi biner.
- **_Metrics_**: _accuracy_, untuk mengukur performa model dalam mengklasifikasikan data.

Input untuk model ini adalah gambar _grayscale_ dengan ukuran (224, 224, 1).

### Transfer Model (EfficientNetB0)

Model ini menggunakan pendekatan _transfer learning_ dengan memanfaatkan _EfficientNetB0_, sebuah arsitektur yang telah dilatih sebelumnya pada dataset _ImageNet_. Pendekatan ini dipilih untuk memanfaatkan fitur yang telah dipelajari dari dataset besar, yang dapat meningkatkan performa pada dataset yang lebih kecil seperti dataset malaria ini. 🌍

Langkah-langkah pembangunan model:
- **_Base Model_**: _EfficientNetB0_ dimuat dengan bobot pra-latih dari _ImageNet_. Semua lapisan pada _base model_ diatur sebagai _trainable = False_, sehingga bobotnya tidak diperbarui selama pelatihan dan fitur yang telah dipelajari tetap dipertahankan.
- **Pengolahan _Output Base Model_**: _Output_ dari _EfficientNetB0_ diproses melalui lapisan _GlobalAveragePooling2D_, yang mengurangi dimensi spasial menjadi vektor fitur dengan merata-ratakan nilai pada setiap saluran.
- **Lapisan Tambahan**:
    - **_Dense_**: Lapisan _fully connected_ dengan 512 unit dan aktivasi ReLU ditambahkan untuk mempelajari kombinasi fitur spesifik dari dataset malaria.
    - **_Dropout_**: Tingkat 0,5 diterapkan untuk mencegah _overfitting_. 🛡️
    - **_Output_**: Lapisan _Dense_ dengan 1 unit dan aktivasi _sigmoid_ untuk klasifikasi biner.

Model ini dikompilasi dengan konfigurasi berikut:
- **_Optimizer_**: _Adam_ dengan _learning rate_ 0,001.
- **_Loss_**: _binary_crossentropy_.
- **_Metrics_**: _accuracy_.

Input untuk model ini adalah gambar RGB dengan ukuran (224, 224, 3), sesuai dengan kebutuhan _EfficientNetB0_.


## Perbandingan Model 📈

Setelah dilatih, kami menguji kedua model pada data _test_. Hasilnya, model yang dibangun secara kustom menggunakan CNN murni jauh lebih unggul. 🏆


## Environment 🛠️

### Library

- Python: >= 3.11
- os
- cv2 (opencv-python)
- random
- matplotlib.pyplot
- warnings
- numpy
- seaborn
- tensorflow
- pickle
- tensorflowjs
- ai-edge-litert
- sklearn.metrics

### Library Tambahan TensorFlow

- tensorflow.keras.models
- tensorflow.keras.layers
- tensorflow.keras.preprocessing.image
- tensorflow.keras.optimizers
- tensorflow.keras.applications
- tensorflow.keras.callbacks
- tensorflow.keras.utils


## Hasil 🎉

### Inferensi

**A. Custom CNN**

!['Custom CNN'](models/__results___files/__results___28_1.png)

**B. Transfer Model (EfficientNetB0)**

!['Transfer Model (EfficientNetB0)'](models/__results___files/__results___29_1.png)

### Prototipe

Prototipe web kami memungkinkan pengguna mengunggah gambar sel darah dan mendapatkan prediksi instan. Dibangun dengan _TensorFlow.js_, semua proses berjalan di browser, aman, dan cepat. ⚡

Coba sendiri di: [Web Telemedicine](https://siaga-malaria-nusantara.vercel.app/) 🌐


# Rencana ke Depan 🚀

- Memperluas dataset dengan berbagai varian karena proyek ini hanya menggunakan gambar _grayscale_.
- Mengeksplorasi model lain untuk performa optimal. 🔍
