import numpy as np
import pandas as pd
import helper.helper as hp

class Prediction:
    """
    Kelas Prediction digunakan untuk menghasilkan prediksi pada sistem rekomendasi berbasis Collaborative Filtering
    dengan menggunakan data mean-centered dan similarity matrix. Kelas ini mendukung pendekatan user-based atau item-based,
    serta opsi untuk menggunakan mean-centered dari "brother dataset" jika tersedia.

    Attributes:
    -----------
    similarity : list of list
        Matriks similarity yang digunakan untuk perhitungan prediksi.
    mean_centered : list of list
        Hasil mean-centered dari data asli yang diberikan.
    data : list of list
        Matriks data input yang akan digunakan untuk menghitung prediksi.
    k : int
        Jumlah tetangga terdekat (neighbors) yang akan digunakan dalam perhitungan prediksi.
    meanList : list of float
        Daftar mean dari masing-masing vektor (baris atau kolom) dalam data.
    opsional : int
        0 untuk item-based, 1 untuk user-based.
    prediction : list of list
        Matriks hasil prediksi yang dihasilkan oleh algoritma.
    twins : bool, optional
        Menentukan apakah menggunakan dataset kembar (brother dataset) untuk perhitungan mean centered.

    Methods:
    --------
    numerator(similarity, meanCentered)
        Menghitung nilai pembilang (numerator) dari formula prediksi.

    denominator(similarity)
        Menghitung nilai penyebut (denominator) dari formula prediksi.

    selectedNeighborhood(neighborhood, index, indexUser, k, data, meanCentered, *, opsional, twins)
        Memilih `k` tetangga terdekat dari similarity matrix untuk perhitungan prediksi.

    prediction_measure(userTarget, item)
        Menghasilkan nilai prediksi untuk user dan item tertentu.

    main_prediction_measure(data)
        Menghasilkan matriks prediksi untuk seluruh data.

    getPredictionArray()
        Mengembalikan hasil prediksi dalam bentuk array numpy.

    getPredictionDataFrame()
        Mengembalikan hasil prediksi dalam bentuk DataFrame pandas.
    """

    def __init__(self, meanC, similarity, data, *, meanList, meanListBrother=[], mean_centered_result_brother=[], opsional, k, twins=False):
        """
        Inisialisasi objek Prediction.

        Parameters:
        -----------
        meanC : list of list
            Mean-centered matrix dari data yang diberikan.
        similarity : list of list
            Matriks similarity yang dihitung sebelumnya (dari kelas BC atau sejenis).
        data : list of list
            Matriks data input yang akan diprediksi.
        meanList : list of float
            Daftar mean dari masing-masing vektor (baris atau kolom) dalam data.
        meanListBrother : list of float, optional
            Daftar mean dari brother dataset, jika menggunakan dataset kembar.
        mean_centered_result_brother : list of list, optional
            Mean-centered dari brother dataset, jika menggunakan dataset kembar.
        opsional : int
            0 untuk pendekatan item-based, 1 untuk pendekatan user-based.
        k : int
            Jumlah tetangga terdekat yang akan digunakan dalam prediksi.
        twins : bool, optional
            Menentukan apakah menggunakan dataset kembar (brother dataset) untuk perhitungan mean-centered.
        """
        self.similarity = similarity
        self.twins = twins
        if twins:
            self.meanListBrother = meanListBrother
            self.mean_centered_result_brother = mean_centered_result_brother
        self.mean_centered = meanC
        self.data = data
        self.k = k
        self.meanList = meanList
        self.opsional = opsional
        self.prediction = self.main_prediction_measure(self.data)

    @staticmethod
    def numeratorPrediction(similarity, meanCentered):
        """
        Menghitung nilai pembilang (numerator) dari formula prediksi.

        Parameters:
        -----------
        similarity : list of float
            Daftar nilai similarity untuk tetangga terdekat.
        meanCentered : list of float
            Mean-centered nilai untuk tetangga terdekat.

        Returns:
        --------
        float
            Hasil pembilang dari formula prediksi.
        """
        return sum(sim * meanC for sim, meanC in zip(similarity, meanCentered))

    @staticmethod
    def denominatorPrediction(similarity):
        """
        Menghitung nilai penyebut (denominator) dari formula prediksi.

        Parameters:
        -----------
        similarity : list of float
            Daftar nilai similarity untuk tetangga terdekat.

        Returns:
        --------
        float
            Hasil penyebut dari formula prediksi.
        """
        return sum(abs(sim) for sim in similarity)

    @staticmethod
    def selectedNeighborhood(neighborhood, index, indexUser, k, data, meanCentered, *, opsional, twins) -> list:
        """
        Memilih `k` tetangga terdekat dari similarity matrix untuk perhitungan prediksi.

        Parameters:
        -----------
        neighborhood : list of list
            Matriks similarity yang akan dipilih tetangga terdekatnya.
        index : int
            Indeks item atau user yang menjadi referensi.
        indexUser : int
            Indeks user yang sedang diproses.
        k : int
            Jumlah tetangga terdekat yang akan dipilih.
        data : list of list
            Matriks data yang sedang diproses.
        meanCentered : list of list
            Mean-centered data dari matriks asli.
        opsional : int
            0 untuk item-based, 1 untuk user-based.
        twins : bool
            Apakah menggunakan brother dataset untuk mean-centered.

        Returns:
        --------
        list of list
            Daftar `k` tetangga terdekat beserta mean-centered data dari tetangga tersebut.
        """
        # Untuk pendekatan item-based (opsional = 0), data di-transpose
        data = hp.reverseMatrix(data)
        meanCentered = hp.reverseMatrix(meanCentered) if not twins or opsional == 0 else meanCentered

        # Dapatkan indeks dari data yang bernilai 0
        indexZero = hp.checkIndexZeroOfData(data=data, index=indexUser if opsional == 0 else index, indexUser=indexUser if opsional == 1 else index)
        # Membuat Index Similarity
        indexOfNeighborhood = list(np.delete(hp.createList(0, len(neighborhood[indexUser]) - 1), indexZero).tolist())
        # Index Neighborhood Item based = Index
        # Index Neighborhood User based = IndexUser
        print(f"Similarity[{indexUser if opsional == 1 else index}] :",neighborhood[indexUser if opsional == 1 else index])
        neighborhood = list(np.delete(neighborhood[indexUser if opsional == 1 else index], indexZero).tolist())

        # Mengurutkan similarity dan tetangganya
        lengthLoop = len(neighborhood)
        for i in range(lengthLoop - 2, -1, -1):
            indexFlag = i
            prevNeighborhood = np.real(neighborhood[i])
            prevIndexList = indexOfNeighborhood[i]
            innerCondition = True
            j = i + 1
            while innerCondition and j < lengthLoop and prevNeighborhood < np.real(neighborhood[j]):
                if prevNeighborhood < np.real(neighborhood[j]):
                    indexFlag = j
                    neighborhood[j - 1] = np.real(neighborhood[j])
                    indexOfNeighborhood[j - 1] = indexOfNeighborhood[j]
                    j += 1
                else:
                    j += 1
                    innerCondition = False
            neighborhood[indexFlag] = prevNeighborhood
            indexOfNeighborhood[indexFlag] = prevIndexList
        # Index Mean Centered Item Based = IndexUser
        # Index Mean Centered User Based = Index
        print(f"MeanCentered[{indexUser if opsional ==  0 else index}] : {meanCentered[indexUser if opsional ==  0 else index]}")
        indexOfNeighborhood = [
            (meanCentered[indexUser if opsional ==  0 else index][i]) 
            for i in indexOfNeighborhood[0:k]
        ]
        return [neighborhood[0:k], indexOfNeighborhood]

    def prediction_measure(self, userTarget, item) -> float:
        """
        Menghasilkan nilai prediksi untuk user dan item tertentu.

        Parameters:
        -----------
        userTarget : int
            Indeks user yang sedang diprediksi.
        item : int
            Indeks item yang sedang diprediksi.

        Returns:
        --------
        float
            Nilai prediksi berdasarkan formula Collaborative Filtering.
        """
        print("User Index :",userTarget,"Item Index :",item)
        print("Similarity :", self.similarity)
        print("Main Centered :", hp.reverseMatrix(self.mean_centered) if not self.twins else hp.reverseMatrix(self.mean_centered_result_brother))
        target = self.selectedNeighborhood(self.similarity, item, userTarget, self.k, self.data, self.mean_centered if not self.twins else hp.reverseMatrix(self.mean_centered_result_brother), opsional=self.opsional, twins=self.twins)
        print(target)
        return (self.meanList[userTarget if self.opsional == 1 else item] 
                if not self.twins else 
                self.meanListBrother[userTarget if self.opsional == 1 else item]) + self.numeratorPrediction(target[0], target[1]) / self.denominatorPrediction(target[0])

    def main_prediction_measure(self, data):
        """
        Menghasilkan matriks prediksi untuk seluruh data.

        Parameters:
        -----------
        data : list of list
            Matriks data yang akan dihitung prediksinya.

        Returns:
        --------
        list of list
            Matriks prediksi untuk seluruh data.
        """
        return [
            [
                (self.prediction_measure(i, j) if data[i][j] == 0 else data[i][j]) 
                for j in range(len(data[0]))
            ]
            for i in range(len(data))
        ]

    def getPredictionArray(self):
        """
        Mengembalikan hasil prediksi dalam bentuk array numpy.

        Returns:
        --------
        numpy.ndarray
            Array numpy yang berisi hasil prediksi.
        """
        return np.array(self.prediction)

    def getPredictionDataFrame(self):
        """
        Mengembalikan hasil prediksi dalam bentuk DataFrame pandas.

        Returns:
        --------
        pandas.DataFrame
            DataFrame yang berisi hasil prediksi.
        """
        return pd.DataFrame(self.prediction)
