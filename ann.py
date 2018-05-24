import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import timeit
import random


class yapay_sinir_agi():
    def __init__(self, katmanlar):
        self.katmanlar = katmanlar
        self.b = [np.random.randn(k, 1) for k in self.katmanlar[1:]]  # bias degerleri (ilk katman haric)
        self.W = [np.random.randn(k2, k1) for k1, k2 in zip(self.katmanlar[:-1], self.katmanlar[1:])]
        self.H = []  # hata

        self.onlyOnce = True

    def ag(self):
        return self.W, self.b

    def ileribesleme(self, a):
        """Katman katman yeni a degerleri hesaplaniyor"""
        a = self.checkDimension(a)
        for w, b in self.W, self.b:
            z = w * a + b
            a = self.sigmoid(z)
        return a

    def geribesleme(self, X, y):
        delta_b = [np.zeros(b.shape) for b in self.b]
        delta_w = [np.zeros(w.shape) for w in self.W]
        a = X
        A, Z = [a], []  # A, Z degerleri
        for w, b in self.W, self.b:  # z ve a degerlerini depolayalim
            z = w * a + b
            a = self.sigmoid(z)
            Z.append(z)
            A.append(a)

            self.printShape(b, "b", w, "w")

        hata = A[-1] - y  # En son katmandaki hata
        delta = hata * self.sigmoid_turevi(Z[-1])
        delta_b[-1] = delta  # Son katmanda W, b'deki degisim
        delta_w[-1] = delta * A[-2].T  # ERROR: np.dot(delta, A[-2].T)

        self.printShape(delta_b[-1], "delta_b[-1]", delta_w[-1], "delta_w[-1]")

        for k in range(2, len(self.katmanlar)):  # Hatanin geriye yayilimi
            delta = np.dot(self.W[-k + 1].T, delta) * self.sigmoid_turevi(Z[-k])
            delta_b[-k] = delta
            delta_w[-k] = delta * A[-k - 1].T  # ERROR: np.dot(delta, A[-k-1].T)

            self.printShape(delta_b[-k], "delta_b[-k]", delta_w[-k], "delta_w[-k]")
        self.onlyOnce = False

        return (delta_b, delta_w)

    def hata(self, X, y):
        a = self.ileribesleme(X)
        if a.shape != y.shape: print('hata')
        return np.sum(np.power(a - y, 2))

    def gradyan_inis(self, X_train, y_train, alpha, number_steps):
        print("X_train.shape", X_train.shape)
        print("y_train.shape", y_train.shape)
        for s in range(number_steps):
            i, m = 0, X_train.shape[1]
            X, y = X_train[:, [i]], y_train[:, [i]]
            tum_delta_b, tum_delta_w = self.geribesleme(X, y)
            hata = self.hata(X, y)

            for i in range(1, m):  # Tum X kolonlari icin
                X, y = X_train[:, [i]], y_train[:, [i]]
                delta_b, delta_w = self.geribesleme(X, y)
                tum_delta_b = [tdb + db for tdb, db in zip(tum_delta_b, delta_b)]
                tum_delta_w = [tdw + dw for tdw, dw in zip(tum_delta_w, delta_w)]
                hata += self.hata(X, y)

            tum_delta_b = [alpha * tdb for tdb in tum_delta_b]
            tum_delta_w = [alpha * tdw for tdw in tum_delta_w]

            self.W = [w - dw for w, dw in zip(self.W, tum_delta_w)]
            self.b = [b - db for b, db in zip(self.b, tum_delta_b)]
            self.H.append(hata / m)

    def fit(self, X_train, y_train, alpha=0.0000001, number_steps=1000):
        X_train = X_train.T  # X verileri kolon=gozlem, satir=oznitelik (alistigimizin tersi)
        y_train = self.checkOutputLayer(y_train)
        return self.gradyan_inis(X_train, y_train, alpha, number_steps)

    def predict(self, X_test):
        if self.katmanlar[-1] == 1:
            tahmin = self.ileribesleme(X_test.T) >= 0.5
            t = tahmin.astype('int')
            return t[0]
        return np.argmax(self.ileribesleme(X_test.T), axis=0)

    #### Yardimci Fonksiyonlar
    def sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    def sigmoid_turevi(self, z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))

    def checkDimension(self, x):
        if x.ndim == 1: return x.reshape(x.shape[0], 1)
        return x

    def checkOutputLayer(self, y):
        if len(set(y)) == 2:
            return y.reshape(1, y.shape[0])
        y_vec = np.zeros((len(set(y)), len(y)))
        for c, r in enumerate(y):
            r = int(r-1)
            y_vec[r, c] = 1
        return y_vec

    def printShape(self, b, bs, w, ws):
        if self.onlyOnce == True: print(bs, ".shape: ", b.shape, " ", ws, ".shape: ", w.shape)

    def runNN(self, r, X_train, X_test, y_train, y_test, alpha=0.001, number_steps=100):
        if r == 2: r = 1
        # Fitting Our Own Neural Network to the Training set
        start_time = timeit.default_timer()
        ysa = yapay_sinir_agi(katmanlar=[64, 12, r])
        ysa.fit(X_train, y_train, alpha, number_steps)

        tahmin = ysa.predict(X_test)
        print("Time: ", timeit.default_timer() - start_time)

        # Making the Confusion Matrix
        from sklearn.metrics import confusion_matrix, accuracy_score
        cm = confusion_matrix(y_test, tahmin)
        print("\t\t\t\t\t---Our Own Neural Network---")
        print("confusion_matrix:\n", cm)
        print("accuracy_score: ", accuracy_score(y_test, tahmin))
        plt.plot(ysa.H)
        print("\nMatrix Shape")
        for w, b in zip(ysa.W, ysa.b):
            print("b.shape: ", b.shape, " w.shape: ", w.shape)