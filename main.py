from main_window import Ui_MainWindow
from PyQt6 import QtCore, QtGui, QtWidgets
from direction_calculator import DirectionCalculator
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.backends.qt_compat import QtWidgets
from matplotlib.figure import Figure
from my_math import ApproxMode, FilterAlgo
from matplotlib import ticker
import numpy as np
import time
import traceback

class CompletedMainWindow(QtWidgets.QMainWindow, Ui_MainWindow): 
    def __init__(self, parent=None):
        super(CompletedMainWindow, self).__init__(parent)
        self.setupUi(self)
        self.add_canvases()
        self.retranslateUi(self)
        self.connect_signals()
       
    def add_canvases(self):
        self.adfr_canvas = FigureCanvas(Figure(figsize=(5, 3), layout="constrained"))
        self.adfr_canvas.setParent(self.adfr_tab)
        self.verticalLayout_12.addWidget(self.adfr_canvas)
        self.verticalLayout_12.addWidget(NavigationToolbar(self.adfr_canvas, self.adfr_tab))
        
        self.amp_canvas = FigureCanvas(Figure(figsize=(5, 3), layout="constrained"))
        self.amp_canvas.setParent(self.amp_tab)
        self.verticalLayout_10.addWidget(self.amp_canvas)
        self.verticalLayout_10.addWidget(NavigationToolbar(self.amp_canvas, self.amp_tab))
        
        self.phase_canvas = FigureCanvas(Figure(figsize=(5, 3), layout="constrained"))
        self.phase_canvas.setParent(self.phase_tab)
        self.verticalLayout_9.addWidget(self.phase_canvas)
        self.verticalLayout_9.addWidget(NavigationToolbar(self.phase_canvas, self.phase_tab))
        
        self.ampphase_canvas = FigureCanvas(Figure(figsize=(5, 3), layout="constrained"))
        self.ampphase_canvas.setParent(self.ampphase_tab)
        self.verticalLayout_8.addWidget(self.ampphase_canvas)
        self.verticalLayout_8.addWidget(NavigationToolbar(self.ampphase_canvas, self.ampphase_tab))
        
        self.e_canvas = FigureCanvas(Figure(figsize=(5, 3)))
        self.e_canvas.setParent(self.e_tab)
        self.verticalLayout_11.addWidget(self.e_canvas)
        self.verticalLayout_11.addWidget(NavigationToolbar(self.e_canvas, self.e_tab))
        
    def connect_signals(self):
        self.submit_button.clicked.connect(self.calculate)
        self.sll1_button.clicked.connect(self.set_sll1_path)
        self.sll2_button.clicked.connect(self.set_sll2_path)
        self.faz_button.clicked.connect(self.set_faz_path)
    
        
    def set_sll1_path(self):
        dialog = QtWidgets.QFileDialog()
        dialog.setFileMode(QtWidgets.QFileDialog.FileMode.ExistingFile)
        dialog.setFilter(QtCore.QDir.Filter.Files)
        
        if dialog.exec():
            self.sll1_lineEdit.setText(dialog.selectedFiles()[0])
       
    def set_sll2_path(self):
        dialog = QtWidgets.QFileDialog()
        dialog.setFileMode(QtWidgets.QFileDialog.FileMode.ExistingFile)
        dialog.setFilter(QtCore.QDir.Filter.Files)
        
        if dialog.exec():
            self.sll2_lineEdit.setText(dialog.selectedFiles()[0])
            
    def set_faz_path(self):
        dialog = QtWidgets.QFileDialog()
        dialog.setFileMode(QtWidgets.QFileDialog.FileMode.ExistingFile)
        dialog.setFilter(QtCore.QDir.Filter.Files)
        
        if dialog.exec():
            self.faz_lineEdit.setText(dialog.selectedFiles()[0])    
            
    def _draw_ADFR(self, canvas_data):
        self.adfr_canvas.figure.clear()
        ax = self.adfr_canvas.figure.subplots()
        
        x_arr = canvas_data["x_arr"]
        y1_arr = canvas_data["y1_arr"]
        y2_arr = canvas_data["y2_arr"]
        pel_char = canvas_data["pel_char"]
        x_arr_trunc = canvas_data["x_arr_trunc"]
        approxed_pel = canvas_data["approxed_pel"]
                
        ax.plot(x_arr, y1_arr, label="ДНА 1-й антенны")
        ax.plot(x_arr, y2_arr, label="ДНА 2-й антенны")
        ax.plot(x_arr, pel_char, label="Амплитудная пел. хар-ка")
        ax.plot(x_arr_trunc, approxed_pel, label="Аппроксимированная пел.хар-ка")
        ax.set_xlabel('φ, [град.]')
        ax.set_ylabel('A, [Вт]')
        ax.set_title("Амплитудная пеленгационная хар-ка")
        ax.xaxis.set_major_locator(ticker.AutoLocator())
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.grid(True)
        ax.legend(loc='upper right')
        self.adfr_canvas.draw()
    
    def _draw_E(self, canvas_data):
        t_arr = canvas_data["t_arr"]
        E1_func = canvas_data["E1_func"]
        E11_func = canvas_data["E11_func"]
        E2_func = canvas_data["E2_func"]
        E22_func = canvas_data["E22_func"]
        
        self.e_canvas.figure.clear()
        ax = self.e_canvas.figure.subplots()
        
        ax.plot(t_arr, E1_func(t_arr), label="E1")
        ax.plot(t_arr, E11_func(t_arr), label="E11")
        ax.plot(t_arr, E2_func(t_arr), label="E2")
        ax.plot(t_arr, E22_func(t_arr), label="E22")
        ax.set_xlabel('t, [с]')
        ax.set_ylabel('E, [В]')
        ax.set_title("Квадратурные составляющие сигнала на входе приемника")
        ax.xaxis.set_major_locator(ticker.AutoLocator())
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.grid(True)
        ax.legend(loc='upper right')
        
        self.e_canvas.draw()
    
    def _draw_AMP(self, canvas_data):
        t_arr = canvas_data["t_arr"]
        A1_arr = canvas_data["A1_arr"]
        A2_arr = canvas_data["A2_arr"]
        phi_arr = canvas_data["phi_arr"]
        phi_approxed = canvas_data["phi_approxed"]
        
        phi_abs_max = max(abs(phi) for phi in phi_arr)
            
        self.amp_canvas.figure.clear()
        axs = self.amp_canvas.figure.subplots(1, 2)
        
        minA1, maxA1 = min(A1_arr), max(A1_arr)
        minA2, maxA2 = min(A2_arr), max(A2_arr)
        mixAax = min(minA1, minA2, 0)
        maxAax = max(maxA1, maxA2, 0)
    
        axs[0].plot(t_arr, A1_arr, label="A1")
        axs[0].plot(t_arr, A2_arr, label="A2")
        axs[0].set_xlabel('t, [c]')
        axs[0].set_ylabel('A, [В]')
        axs[0].set_title("Амплитуды")
        axs[0].set_xlim(0, t_arr[-1])
        axs[0].set_ylim(mixAax, maxAax*1.5)
        axs[0].xaxis.set_major_locator(ticker.AutoLocator())
        axs[0].yaxis.set_minor_locator(ticker.AutoMinorLocator())
        axs[0].grid(True)
        axs[0].legend(loc='upper right')
        
        axs[1].plot(t_arr, phi_arr, label="φ")
        axs[1].plot(t_arr, phi_approxed, label="φ_аппрокс.")
        axs[1].set_xlabel('t, [c]')
        axs[1].set_ylabel('φ, [град.]')
        axs[1].set_title("Вычисленный угол")
        axs[1].set_xlim(0, t_arr[-1])
        axs[1].set_ylim(min(-45, -phi_abs_max*1.5), max(45, phi_abs_max*1.5))
        axs[1].xaxis.set_major_locator(ticker.AutoLocator())
        axs[1].yaxis.set_minor_locator(ticker.AutoMinorLocator())
        axs[1].grid(True)
        axs[1].legend(loc='upper right')
        
        self.amp_canvas.draw()
    
    def _draw_PHASE(self, canvas_data):
        t_arr = canvas_data["t_arr"]
        phase1_arr = canvas_data["phase1_arr"]
        phase2_arr = canvas_data["phase2_arr"]
        delta_phase_arr = canvas_data["delta_phase_arr"]
        delta_phase_approxed_arr = canvas_data["delta_phase_approxed_arr"]
        deg_x_arr = canvas_data["deg_x_arr"]
        faz_arr = canvas_data["faz_arr"]
        faz_approxed_arr = canvas_data["faz_approxed_arr"]
        faz_approxed_normed_arr = canvas_data["faz_approxed_normed_arr"]
        delta_phase = canvas_data["delta_phase"]
        
        self.phase_canvas.figure.clear()
        axs = self.phase_canvas.figure.subplots(1, 2)
        
        axs[0].plot(t_arr, np.rad2deg(phase1_arr), label="Фаза1")
        axs[0].plot(t_arr, np.rad2deg(phase2_arr), label="Фаза2")
        axs[0].plot(t_arr, np.rad2deg(delta_phase_arr), label="Разность фаз")
        axs[0].plot(t_arr, np.rad2deg(delta_phase_approxed_arr), label="Аппрокс. разность фаз")
        
        axs[0].set_xlabel('t, [с]')
        axs[0].set_ylabel('φ, [град.]')
        axs[0].set_title("Разность фаз")
        axs[0].xaxis.set_major_locator(ticker.AutoLocator())
        axs[0].yaxis.set_minor_locator(ticker.AutoMinorLocator())
        axs[0].grid(True)
        axs[0].legend(loc='upper right')
        
        axs[1].plot(deg_x_arr, np.rad2deg(faz_arr), label="Фазовая пел. хар-ка")
        axs[1].plot(deg_x_arr, np.rad2deg(faz_approxed_arr), label="Аппрокс. фаз. пел. хар-ка")
        axs[1].plot(deg_x_arr, np.rad2deg(faz_approxed_normed_arr), label="Норм. аппрокс. фаз. пел. хар-ка")
        axs[1].plot(deg_x_arr, np.rad2deg([delta_phase for _ in deg_x_arr]), label="Разность фаз")
        axs[1].set_xlabel('Угол, [град.]')
        axs[1].set_ylabel('φ, [град.]')
        axs[1].set_title("Фазовая пел. хар-ка")
        axs[1].xaxis.set_major_locator(ticker.AutoLocator())
        axs[1].yaxis.set_minor_locator(ticker.AutoMinorLocator())
        axs[1].grid(True)
        axs[1].legend(loc='upper right')
    
    def _draw_AMPPHASE(self, canvas_data):
        x_arr = canvas_data["x_arr"]
        angles_phase = canvas_data["angles_phase"]
        best = canvas_data["best"]
        angle_amp = canvas_data["angle_amp"]
    
        self.ampphase_canvas.figure.clear()
        axs = self.ampphase_canvas.figure.subplots()
    
        i = 0
        for inter in angles_phase:
            width = 1
            if inter == best:
                width = 2
            axs.plot(x_arr, [inter for _ in x_arr], label="φ_фаз"+str(i+1), color='blue', linewidth=width)
            i += 1
    
        axs.plot(x_arr, [angle_amp for _ in x_arr], label="φ_амп", color='red')
        
        axs.set_xlabel('t, [с]')
        axs.set_ylabel('φ, [град.]')
        axs.set_title("Амплитудно-фазовый метод")
        axs.xaxis.set_major_locator(ticker.AutoLocator())
        axs.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        axs.grid(True)
        axs.legend(loc='upper right')
        
        self.ampphase_canvas.draw()
    
          
    def calculate(self):
        phi_0 = self.phi_0_spinBox.value()
        phi_n = self.phi_n_spinBox.value()
        phi_min = self.phi_min_spinBox.value()
        phi_max = self.phi_max_spinBox.value()
        f_c = self.f_c_spinBox.value()*(10**9)
        noise_enable = self.noise_checkBox.isChecked()
        q = self.q_spinBox.value()
        sll1_path = self.sll1_lineEdit.text()
        sll2_path = self.sll2_lineEdit.text()
        faz_path = self.faz_lineEdit.text()
        approx_mode = ApproxMode(self.approx_mode_combo_box.currentIndex())
        poly_degree = self.poly_degree_spinBox.value()
        phi_pel = self.phi_pel_spinBox.value()
        K_n = self.k_n_spinBox.value()
        freq_num = self.freq_num_spinBox.value()
        prefilter_en = self.filter_checkBox.isChecked()
        if prefilter_en:
            prefilter_algo = FilterAlgo(self.filter_algo_comboBox.currentIndex())
        else:
            prefilter_algo = FilterAlgo(0)
        t = self.t_spinBox.value()*(10**-9)
        f_discr = self.f_discr_spinBox.value()*10**9
        
        sims_num = self.sims_num_spinBox.value()
        
        try:
            times_arr = []
            divs_arr = []

            calc = None
            angle, accuracy = None, None

            for _ in range(sims_num):
                calc = DirectionCalculator(q, phi_0, phi_min, phi_max, f_c, sll1_path, sll2_path, faz_path, approx_mode,
                                    freq_num, phi_pel, K_n, phi_n, noise_enable, poly_degree, prefilter_en, prefilter_algo, t, f_discr)
                
                angle, accuracy = calc.calculate()
                approx_time = calc.get_approx_time()
                times_arr.append(approx_time)
                
                div_v = abs(phi_pel - angle)
                divs_arr.append(div_v)
            
            self.res_label.setText('{:.3f}°'.format(angle))
            self.accuracy_label.setText('±{:.3f}°'.format(accuracy))
            self.avg_time_label.setText('{:.3f} мс'.format(sum(times_arr) / len(times_arr) * 1000))
            self.avg_div_label.setText('{:.3f}°'.format(sum(divs_arr) / len(divs_arr)))
            
            cd = calc.get_canvas_data()
            
            self._draw_ADFR(cd["adfr"])
            self._draw_E(cd["e"])
            self._draw_AMP(cd["amp"])
            self._draw_PHASE(cd["phase"])
            self._draw_AMPPHASE(cd["ampphase"])
                
        except Exception as e:
            error_dialog = QtWidgets.QMessageBox()
            error_dialog.setWindowTitle("Ошибка")
            error_dialog.setText(f"Произошла ошибка: {e}")
            error_dialog.setIcon(QtWidgets.QMessageBox.Icon.Critical)
            error_dialog.exec()
            traceback.print_exc()
            
                      
if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = CompletedMainWindow()
    MainWindow.show()
    sys.exit(app.exec())
