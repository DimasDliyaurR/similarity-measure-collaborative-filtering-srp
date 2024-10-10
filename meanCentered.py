import helper.helper as hp
import pandas as pd

class MeanCentered:
    """
    Kelas MeanCentered digunakan untuk menghitung mean-centered data dari matriks
    yang diberikan. Kelas ini dapat digunakan untuk melakukan mean-centered dengan 
    pendekatan item-based (opsional = 0) atau user-based (opsional = 1). Terdapat 
    juga opsi 'twins' yang memungkinkan perhitungan mean-centered dari 'saudara' 
    mean (mean kembar) berdasarkan data asli.

    Attributes:
    -----------
    data : list of list
        Matriks data input yang akan diproses.
    opsional : int
        0 untuk pendekatan item-based, 1 untuk pendekatan user-based.
    meanList : list of float
        Daftar mean dari masing-masing vektor (baris atau kolom) dalam data.
    meanListBrother : list of float
        Daftar mean dari mean-centered 'kembar' (jika twins diatur menjadi True).
    mean_centered_result : list of list
        Hasil mean-centered dari data asli.
    mean_centered_result_brother : list of list
        Hasil mean-centered dari meanListBrother (jika twins diatur menjadi True).

    Methods:
    --------
    mean(data)
        Menghitung mean dari vektor yang diberikan.

    mean_centered_measure(data, meanList)
        Menghasilkan mean-centered data berdasarkan meanList yang diberikan.

    getMeanCenteredArray()
        Mengembalikan mean-centered data dalam bentuk array numpy.

    getMeanCenteredDataFrame()
        Mengembalikan mean-centered data dalam bentuk DataFrame pandas.

    getMeanListArray()
        Mengembalikan daftar mean dalam bentuk array numpy.

    getMeanListDataFrame()
        Mengembalikan daftar mean dalam bentuk DataFrame pandas.
    """

    def __init__(self, data, *, opsional, twins=False):
        """
        Inisialisasi objek MeanCentered.

        Parameters:
        -----------
        data : list of list
            Matriks data masukan yang akan diproses.
        opsional : int
            0 untuk pendekatan item-based, 1 untuk pendekatan user-based.
        twins : bool, optional
            True jika ingin menghitung mean-centered dari meanList kembar (default: False).
        """
        self.data = data if opsional != "item-based" else hp.reverseMatrix(data)
        self.opsional = opsional
        self.meanList = [self.mean(i) for i in self.data]
        self.meanListBrother = [self.mean(i) for i in hp.reverseMatrix(self.data)] if twins else []
        self.mean_centered_result = self.mean_centered_measure(self.data, self.meanList)
        self.mean_centered_result_brother = self.mean_centered_measure(hp.reverseMatrix(self.data), self.meanListBrother) if twins else []

    @staticmethod
    def mean(data) -> float:
        """
        Menghitung mean dari vektor yang diberikan, mengabaikan elemen nol.

        Parameters:
        -----------
        data : list of float
            Vektor yang akan dihitung mean-nya.

        Returns:
        --------
        float
            Nilai mean dari vektor yang diberikan, mengabaikan elemen nol.
        """
        return sum([j for j in data if j != 0]) / len([j for j in data if j != 0])

    def mean_centered_measure(self, data, meanList) -> list[list]:
        """
        Menghasilkan mean-centered data berdasarkan meanList yang diberikan.

        Parameters:
        -----------
        data : list of list
            Matriks data yang akan di mean-centered.
        meanList : list of float
            Daftar mean untuk setiap vektor dalam data.

        Returns:
        --------
        list of list
            Matriks mean-centered yang telah disesuaikan dengan meanList yang diberikan.
        """
        return [[(data[i][j] - meanList[i] if data[i][j] != 0 else 0) for j in range(len(data[i]))] for i in range(len(data))]

    def getMeanCenteredArray(self):
        """
        Mengembalikan mean-centered data dalam bentuk array numpy.

        Returns:
        --------
        numpy.ndarray
            Array numpy yang berisi mean-centered data.
        """
        return self.mean_centered_result
   
    def getMeanCenteredDataFrame(self):
        """
        Mengembalikan mean-centered data dalam bentuk array numpy.

        Returns:
        --------
        numpy.ndarray
            Array numpy yang berisi mean-centered data.
        """
        return pd.DataFrame(self.mean_centered_result)

    def getMeanListArray(self):
        """
        Mengembalikan daftar mean dalam bentuk array numpy.

        Returns:
        --------
        numpy.ndarray
            Array numpy yang berisi mean dari setiap vektor dalam data.
        """
        return self.meanList
    
    def getMeanListDataFrame(self):
        """
        Mengembalikan daftar mean dalam bentuk array numpy.

        Returns:
        --------
        numpy.ndarray
            Array numpy yang berisi mean dari setiap vektor dalam data.
        """
        return pd.DataFrame(self.meanList)