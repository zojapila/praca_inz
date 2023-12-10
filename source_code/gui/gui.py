# pyuic5 -x gui_design.ui -o gui_code.py
from gui_code import *
import ga.genetic_algorithm as ga
import sys
import time
import data_preperation.data_preprocessing as preprocessing


class App(Ui_MainWindow):
    def __init__(self, window):
        self.setupUi(window)
        self.startGA.clicked.connect(self.runGA)

        self.timer = QtCore.QTimer(self.centralwidget)
        self.timer.timeout.connect(self.aktualizuj_czas)
        self.czas_start = 0

    def runGA(self):
        # population = self.populationSizeGA.toPlainText()
        self.czas_start = time.time()  # Początkowy czas
        self.timer.start(1000)
        # iter = self.maxIterGA.text()
        # regs = self.resultQuantityGA.text()
        # prob = self.mutationProbability.text()
        training_data = "D:/studia/inzynierka/unsw_nb15/UNSW_NB15_training-set.csv"
        processed_data = preprocessing.DataPreprocessing(training_data, alg_type='ga')

        # genetic_algorithm = ga.GeneticAlgorithm(processed_data)
        # result = genetic_algorithm.geneticAlgorithmLoop()

        # result = genetic_algorithm.geneticAlgorithmLoop()
        # self.textEdit_11.setPlainText(result.head())

    def aktualizuj_czas(self):
        obecny_czas = time.time()
        elapsed_time = obecny_czas - self.czas_start  # Czas wykonania
        self.textEdit_9.setPlainText("{:.1f}".format(elapsed_time))


app = QtWidgets.QApplication(sys.argv)
MainWindow = QtWidgets.QMainWindow()

ui = App(MainWindow)

MainWindow.show()
app.exec_()
