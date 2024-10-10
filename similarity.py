import pandas as pd
import time
import numpy as np
import helper.helper as hp
import cmath
import helper.helper as help
import meanCentered as mc
import prediction as pc

class Pearson(mc.MeanCentered, pc.Prediction):
    """
    Kelas Pearson digunakan untuk menghitung Pearson Correlation Coefficient sebagai
    ukuran kesamaan (similarity) antara dua vektor data. Kelas ini merupakan turunan dari
    kelas MeanCentered dan Prediction.

    Attributes:
    -----------
    data : list of list
        Matriks data masukan untuk perhitungan similarity.
    opsional : int
        0 untuk pendekatan item-based, 1 untuk pendekatan user-based.
    k : int
        Jumlah tetangga (neighbors) yang akan dipertimbangkan dalam prediksi.
    result : list of list
        Hasil perhitungan similarity.

    Methods:
    --------
    numerator(data1, data2)
        Menghitung nilai numerator (pembilang) untuk perhitungan similarity.
    
    denominator(data1, data2)
        Menghitung nilai denominator (penyebut) untuk perhitungan similarity.

    measureSimilarity(u, v, data, meanC=[])
        Menghitung Pearson similarity antara dua vektor u dan v.

    mainSimilarityMeasure()
        Menghitung similarity untuk semua pasangan vektor dalam data yang diberikan.

    getSimilarityArray()
        Mengembalikan hasil perhitungan similarity dalam bentuk array numpy.

    getSimilarityDataFrame()
        Mengembalikan hasil perhitungan similarity dalam bentuk DataFrame pandas.
    """
    
    def __init__(self, data, *, opsional, k):
        """
        Inisialisasi objek Pearson.

        Parameters:
        -----------
        data : list of list
            Matriks data masukan untuk perhitungan similarity.
        opsional : int
            0 untuk pendekatan item-based, 1 untuk pendekatan user-based.
        k : int
            Jumlah tetangga (neighbors) yang akan dipertimbangkan dalam prediksi.
        """
        super().__init__(data, opsional=opsional)
        self.start = time.time()
        self.result = self.mainSimilarityMeasure()
        pc.Prediction.__init__(self, self.mean_centered_result, self.result, data, meanList=self.meanList, opsional=opsional, k=k)

    @staticmethod
    def numerator(data1, data2):
        """
        Menghitung nilai numerator (pembilang) dari dua vektor.

        Parameters:
        -----------
        data1 : list of float
            Vektor pertama yang akan digunakan dalam perhitungan.
        data2 : list of float
            Vektor kedua yang akan digunakan dalam perhitungan.

        Returns:
        --------
        float
            Nilai hasil perkalian dot product dari dua vektor.
        """
        return sum((data1[i] * data2[i]) for i in range(len(data1)))

    @staticmethod
    def denominator(data1, data2):
        """
        Menghitung nilai denominator (penyebut) dari dua vektor.

        Parameters:
        -----------
        data1 : list of float
            Vektor pertama yang akan digunakan dalam perhitungan.
        data2 : list of float
            Vektor kedua yang akan digunakan dalam perhitungan.

        Returns:
        --------
        float
            Nilai hasil perkalian magnitudo (panjang) dari dua vektor.
        """
        return cmath.sqrt(sum((data1[i]**2) for i in range(len(data1)))) * cmath.sqrt(sum((data2[i]**2) for i in range(len(data2))))

    def measureSimilarity(self, u, v, data, meanC=[]):
        """
        Menghitung Pearson similarity antara dua vektor u dan v.

        Parameters:
        -----------
        u : int
            Indeks vektor pertama dalam data.
        v : int
            Indeks vektor kedua dalam data.
        data : list of list
            Matriks data yang akan digunakan untuk perhitungan similarity.
        meanC : list of list, optional
            Hasil mean-centered dari data (default: []).

        Returns:
        --------
        float
            Nilai Pearson similarity antara dua vektor.
        """
        tempMc1 = [meanC[u][mc1] for mc1 in range(len(data[u]))]
        tempMc2 = [meanC[v][mc2] for mc2 in range(len(data[v]))]

        tempMc1 = np.delete(tempMc1, help.indexOfZero(data[u], data[v]))
        tempMc2 = np.delete(tempMc2, help.indexOfZero(data[u], data[v]))
        denom = self.denominator(tempMc1, tempMc2)
        return (self.numerator(tempMc1, tempMc2) / denom) if denom != 0 else 0

    def mainSimilarityMeasure(self):
        """
        Menghitung similarity untuk semua pasangan vektor dalam data yang diberikan.

        Returns:
        --------
        list of list
            Matriks similarity yang berisi nilai Pearson similarity antara setiap pasangan vektor.
        """
        result = [[] for _ in range(len(self.data))]
        for i in range(len(self.data)): 
            for j in range(i, len(self.data)):
                if i == j:
                    result[i].append(1)
                    continue
                similarity_result = self.measureSimilarity(i, j, self.data, self.mean_centered_result)
                result[i].append(similarity_result)
                result[j].append(similarity_result)
        return result
    
    def getSimilarityArray(self):
        """
        Mengembalikan hasil perhitungan similarity dalam bentuk array numpy.

        Returns:
        --------
        numpy.ndarray
            Array numpy yang berisi nilai Pearson similarity.
        """
        return np.array(self.result)

    def getSimilarityDataFrame(self):
        """
        Mengembalikan hasil perhitungan similarity dalam bentuk DataFrame pandas.

        Returns:
        --------
        pandas.DataFrame
            DataFrame yang berisi nilai Pearson similarity.
        """
        return pd.DataFrame(self.result)


class Cosine(Pearson):
    """
    Kelas Cosine digunakan menghitung Cosine sebagai
    ukuran kesamaan (similarity) antara dua vektor data. Kelas ini merupakan turunan dari Kelas Pearson
    dan kelas Pearson memiliki parent kelas MeanCentered dan Prediction.

    Attributes:
    -----------
    data : list of list
        Matriks data masukan untuk perhitungan similarity.
    opsional : str
        item-based, user-based.
    k : int
        Jumlah tetangga (neighbors) yang akan dipertimbangkan dalam prediksi.
    result : list of list
        Hasil perhitungan similarity.

    Methods:
    --------
    numerator(data1, data2)
        Menghitung nilai numerator (pembilang) untuk perhitungan similarity.
    
    denominator(data1, data2)
        Menghitung nilai denominator (penyebut) untuk perhitungan similarity.

    measureSimilarity(u, v, data, meanC=[])
        Menghitung Pearson similarity antara dua vektor u dan v.

    mainSimilarityMeasure()
        Menghitung similarity untuk semua pasangan vektor dalam data yang diberikan.

    getSimilarityArray()
        Mengembalikan hasil perhitungan similarity dalam bentuk array numpy.

    getSimilarityDataFrame()
        Mengembalikan hasil perhitungan similarity dalam bentuk DataFrame pandas.
    """
    def __init__(self, data, *, opsional, k):
        """
        Inisialisasi objek Pearson.

        Parameters:
        -----------
        data : list of list
            Matriks data masukan untuk perhitungan similarity.
        opsional : str
            item-based, user-based.
        k : int
            Jumlah tetangga (neighbors) yang akan dipertimbangkan dalam prediksi.
        """
        super().__init__(data, opsional=opsional, k=k)
    
    def measureSimilarity(self,u,v,data):
        """
        Menghitung Cosine similarity turunan dari Pearson antara dua vektor u dan v.
        Yang berbeda hanya pada vektor yang dibandingkan
        Pearson menggunakan Mean Centered sedangkan Cosine menggunakan data rating
        Parameters:
        -----------
        u : int
            Indeks vektor pertama dalam data.
        v : int
            Indeks vektor kedua dalam data.
        data : list of list
            Matriks data yang akan digunakan untuk perhitungan similarity.
        meanC : list of list, optional
            Hasil mean-centered dari data (default: []).

        Returns:
        --------
        float
            Nilai Cosine similarity antara dua vektor.
        """
        tempMc1 = [data[u][mc1]for mc1 in range(len(data[u]))]
        tempMc2 = [data[v][mc2]for mc2 in range(len(data[v]))]

        tempMc1= np.delete(tempMc1,help.indexOfZero(data[u],data[v]))
        tempMc2 = np.delete(tempMc2,help.indexOfZero(data[u],data[v]))

        denom = self.denominator(data[v],data[u])
        return (self.numerator(tempMc1,tempMc2) / denom) if denom != 0 else 0

    def mainSimilarityMeasure(self) :
        """
        Menghitung similarity untuk semua pasangan vektor dalam data yang diberikan.

        Returns:
        --------
        list of list
            Matriks similarity yang berisi nilai Pearson similarity antara setiap pasangan vektor.
        """
        result = [[] for _ in range(len(self.data))]
        for i in range(len(self.data)) :
            for j in range(i,len(self.data)):
                if i == j :
                    result[i].append(1)
                    continue
                similarity_result = self.measureSimilarity(i,j,self.data)
                result[i].append(similarity_result)
                result[j].append(similarity_result)

        return np.transpose(result)

class ACosine(mc.MeanCentered, pc.Prediction):
    """
    Kelas ACosine digunakan untuk menghitung Asymmetric Cosine Similarity sebagai
    ukuran kesamaan (similarity) antara dua vektor data dengan memperhitungkan 
    mean-centered dari pasangan vektor yang berbeda (twins approach).

    Kelas ini menggunakan metode perhitungan mean-centered terbalik antara
    pendekatan item-based dan user-based.

    Attributes:
    -----------
    data : list of list
        Matriks data masukan untuk perhitungan similarity.
    opsional : str
        item-based, user-based.
    k : int
        Jumlah tetangga (neighbors) yang akan dipertimbangkan dalam prediksi.
    result : list of list
        Hasil perhitungan similarity.
    meanListBrother : list
        Daftar nilai mean yang digunakan sebagai mean-centered 'kembar' dari data asli.
    mean_centered_result_brother : list of list
        Hasil mean-centered dari meanListBrother yang digunakan untuk perhitungan similarity.

    Methods:
    --------
    numerator(data1, data2)
        Menghitung nilai numerator (pembilang) untuk perhitungan similarity.

    denominator(data1, data2)
        Menghitung nilai denominator (penyebut) untuk perhitungan similarity.

    mean_centered_measure(data, meanList)
        Menghasilkan mean-centered data berdasarkan meanList yang diberikan.

    measureSimilarity(u, v, data, meanC=[])
        Menghitung Asymmetric Cosine similarity antara dua vektor u dan v.

    mainSimilarityMeasure()
        Menghitung similarity untuk semua pasangan vektor dalam data yang diberikan.

    getSimilarityArray()
        Mengembalikan hasil perhitungan similarity dalam bentuk array numpy.

    getSimilarityDataFrame()
        Mengembalikan hasil perhitungan similarity dalam bentuk DataFrame pandas.
    """
    
    def __init__(self, data, *, opsional, k):
        """
        Inisialisasi objek ACosine dengan mean-centered dari pasangan data yang berbeda.

        Parameters:
        -----------
        data : list of list
            Matriks data masukan untuk perhitungan similarity.
        opsional : str
            item-based, user-based.
        k : int
            Jumlah tetangga (neighbors) yang akan dipertimbangkan dalam prediksi.
        """
        super().__init__(data, opsional="item-based" if opsional == "user-based" else "user-based", twins=True)
        self.result = self.mainSimilarityMeasure()
        pc.Prediction.__init__(self, self.mean_centered_result, self.result, data, 
                               meanList=self.meanList, meanListBrother=self.meanListBrother,
                               mean_centered_result_brother=self.mean_centered_result_brother, 
                               opsional=opsional, k=k, twins=True)

    @staticmethod
    def numerator(data1, data2):
        """
        Menghitung nilai numerator (pembilang) dari dua vektor.

        Parameters:
        -----------
        data1 : list of float
            Vektor pertama yang akan digunakan dalam perhitungan.
        data2 : list of float
            Vektor kedua yang akan digunakan dalam perhitungan.

        Returns:
        --------
        float
            Nilai hasil perkalian dot product dari dua vektor.
        """
        return sum((data1[i] * data2[i]) for i in range(len(data1)))

    @staticmethod
    def denominator(data1, data2):
        """
        Menghitung nilai denominator (penyebut) dari dua vektor.

        Parameters:
        -----------
        data1 : list of float
            Vektor pertama yang akan digunakan dalam perhitungan.
        data2 : list of float
            Vektor kedua yang akan digunakan dalam perhitungan.

        Returns:
        --------
        float
            Nilai hasil perkalian magnitudo (panjang) dari dua vektor.
        """
        return cmath.sqrt(sum((data1[i]**2) for i in range(len(data1)))) * cmath.sqrt(sum((data2[i]**2) for i in range(len(data1))))
    
    def mean_centered_measure(self, data, meanList):
        """
        Menghasilkan mean-centered data berdasarkan meanList yang diberikan.

        Parameters:
        -----------
        data : list of list
            Matriks data masukan yang akan di mean-centered.
        meanList : list of float
            Daftar mean untuk setiap vektor dalam data.

        Returns:
        --------
        list of list
            Matriks mean-centered yang telah disesuaikan dengan meanList yang diberikan.
        """
        return [[(data[i][j] - meanList[i] if data[i][j] != 0 else 0) for j in range(len(data[i]))] for i in range(len(data))] if self.opsional == "item-based" else hp.reverseMatrix([[(data[i][j] - meanList[i] if data[i][j] != 0 else 0) for j in range(len(data[i]))] for i in range(len(data))]) 
    
    def measureSimilarity(self, u, v, data, meanC=[]):
        """
        Menghitung Asymmetric Cosine similarity antara dua vektor u dan v.

        Parameters:
        -----------
        u : int
            Indeks vektor pertama dalam data.
        v : int
            Indeks vektor kedua dalam data.
        data : list of list
            Matriks data yang akan digunakan untuk perhitungan similarity.
        meanC : list of list, optional
            Hasil mean-centered dari data (default: []).

        Returns:
        --------
        float
            Nilai Asymmetric Cosine similarity antara dua vektor.
        """
        tempMc1 = [meanC[u][mc1] for mc1 in range(len(data[u]))]
        tempMc2 = [meanC[v][mc2] for mc2 in range(len(data[v]))]

        tempMc1 = np.delete(tempMc1, help.indexOfZero(data[u], data[v]))
        tempMc2 = np.delete(tempMc2, help.indexOfZero(data[u], data[v]))
        denom = self.denominator(tempMc1, tempMc2).real
        return (self.numerator(tempMc1, tempMc2) / denom) if denom != 0 else 0
    
    def mainSimilarityMeasure(self):
        """
        Menghitung similarity untuk semua pasangan vektor dalam data yang diberikan.

        Returns:
        --------
        list of list
            Matriks similarity yang berisi nilai Asymmetric Cosine similarity 
            antara setiap pasangan vektor.
        """
        result = [[] for _ in range(len(self.data[0]))]
        for i in range(len(self.data[0])):
            for j in range(i, len(self.data[0])):
                if i == j:
                    result[i].append(1)
                    continue
                similarity_result = self.measureSimilarity(i, j, hp.reverseMatrix(self.data),
                self.mean_centered_result if self.opsional == "user-based" else hp.reverseMatrix(self.mean_centered_result))
                result[i].append(similarity_result)
                result[j].append(similarity_result)
        return result
    
    def getSimilarityArray(self):
        """
        Mengembalikan hasil perhitungan similarity dalam bentuk array numpy.

        Returns:
        --------
        numpy.ndarray
            Array numpy yang berisi nilai Asymmetric Cosine similarity.
        """
        return np.array(self.result)
    
    def getSimilarityDataFrame(self):
        """
        Mengembalikan hasil perhitungan similarity dalam bentuk DataFrame pandas.

        Returns:
        --------
        pandas.DataFrame
            DataFrame yang berisi nilai Asymmetric Cosine similarity.
        """
        return pd.DataFrame(self.result)

class BC(mc.MeanCentered, pc.Prediction):
    """
    Kelas BC digunakan untuk menghitung similarity matrix menggunakan metode Bhattacharyya Coefficient (BC). 
    Metode ini mengukur kesamaan antara dua distribusi probabilitas pada masing-masing kolom data.

    Kelas ini merupakan turunan dari `MeanCentered` dan `Prediction`, di mana:
    - `MeanCentered` digunakan untuk menghitung mean-centered data dari matriks yang diberikan.
    - `Prediction` digunakan untuk menghasilkan prediksi berdasarkan similarity matrix.

    Attributes:
    -----------
    result : list of list
        Matriks hasil perhitungan similarity antara semua kolom data.
    data : list of list
        Matriks data input yang akan diproses.
    meanList : list of float
        Daftar mean dari masing-masing vektor (baris atau kolom) dalam data.
    mean_centered_result : list of list
        Hasil mean-centered dari data asli.

    Methods:
    --------
    probabilitas(kolom, target)
        Menghitung probabilitas kemunculan nilai target dalam sebuah kolom.

    bunchOfProbabilitas(kolom)
        Menghasilkan daftar probabilitas untuk setiap nilai dalam rentang yang ada pada kolom.

    measureSimilarity(u, j)
        Mengukur similarity antara dua kolom berdasarkan Bhattacharyya Coefficient.

    mainSimilarityMeasure()
        Menghasilkan similarity matrix untuk semua kolom dalam data.

    getSimilarityArray()
        Mengembalikan similarity matrix dalam bentuk array numpy.

    getSimilarityDataFrame()
        Mengembalikan similarity matrix dalam bentuk DataFrame pandas.
    """

    def __init__(self, data, *, opsional, k):
        """
        Inisialisasi objek BC.

        Parameters:
        -----------
        data : list of list
            Matriks data masukan yang akan diproses.
        opsional : str
            item-based, user-based.
        k : int
            Jumlah tetangga terdekat yang akan digunakan dalam prediksi (untuk superclass `Prediction`).
        """
        super().__init__(data, opsional=opsional)
        self.result = self.mainSimilarityMeasure()
        pc.Prediction.__init__(self, self.mean_centered_result, self.result, data, meanList=self.meanList, opsional=opsional, k=k)

    def probabilitas(self, kolom, target):
        """
        Menghitung probabilitas kemunculan nilai `target` pada kolom tertentu.

        Parameters:
        -----------
        kolom : int
            Indeks kolom pada data yang akan dihitung probabilitasnya.
        target : int
            Nilai yang ingin dihitung probabilitas kemunculannya.

        Returns:
        --------
        float
            Probabilitas kemunculan `target` pada kolom yang diberikan.
        """
        return self.data[kolom].count(target) / (len(self.data[kolom]) - self.data[kolom].count(0))

    def bunchOfProbabilitas(self, kolom):
        """
        Menghasilkan daftar probabilitas kemunculan setiap nilai pada rentang yang ada di kolom tertentu.

        Parameters:
        -----------
        kolom : int
            Indeks kolom pada data yang akan dihitung probabilitas kemunculan setiap nilai dalam rentangnya.

        Returns:
        --------
        list of float
            Daftar probabilitas untuk setiap nilai dalam rentang pada kolom tersebut.
        """
        # Mendapatkan nilai maksimum dan minimum pada setiap kolom, mengabaikan nol
        highestNumber = int(max([max(number) for number in self.data])) + 1
        lowestNumber = int(min([min(hp.remove_items(number, 0)) for number in self.data]))
        return [self.probabilitas(kolom, i) for i in range(lowestNumber, highestNumber)]

    def measureSimilarity(self, u, j):
        """
        Mengukur similarity antara dua kolom berdasarkan Bhattacharyya Coefficient.

        Parameters:
        -----------
        u : int
            Indeks kolom pertama yang akan dibandingkan.
        j : int
            Indeks kolom kedua yang akan dibandingkan.

        Returns:
        --------
        float
            Nilai similarity antara kedua kolom yang dihitung menggunakan Bhattacharyya Coefficient.
        """
        target1 = self.bunchOfProbabilitas(u)
        target2 = self.bunchOfProbabilitas(j)
        result = sum([cmath.sqrt(x * y).real for x, y in zip(target1, target2)])
        return result

    def mainSimilarityMeasure(self):
        """
        Menghasilkan similarity matrix untuk semua kolom dalam data.

        Returns:
        --------
        list of list
            Matriks similarity antara semua kolom data.
        """
        result = [[] for _ in range(len(self.data))]

        for i in range(len(self.data)):
            for j in range(i, len(self.data)):
                if i == j:
                    result[i].append(1)  # Similaritas terhadap dirinya sendiri adalah 1
                    continue
                similarity_result = self.measureSimilarity(i, j)
                result[i].append(similarity_result)
                result[j].append(similarity_result)
        return result

    def getSimilarityArray(self):
        """
        Mengembalikan similarity matrix dalam bentuk array numpy.

        Returns:
        --------
        numpy.ndarray
            Array numpy yang berisi similarity matrix.
        """
        return np.array(self.result)

    def getSimilarityDataFrame(self):
        """
        Mengembalikan similarity matrix dalam bentuk DataFrame pandas.

        Returns:
        --------
        pandas.DataFrame
            DataFrame yang berisi similarity matrix.
        """
        return pd.DataFrame(self.result)

