
import numpy as Numpy;

import pandas as Pandas;

import matplotlib.pyplot as Matplotlib;

from sklearn.linear_model import LinearRegression;

from sklearn.metrics import mean_squared_error as MeanSquaredError;

from sklearn.metrics import mean_absolute_error as MeanAbsoluteError; 

from sklearn.metrics import r2_score as r2_Score;

from sklearn.model_selection import train_test_split as TrainTestSplit;


class LoadData:
    def __init__(self):
        pass;
    
    def DataLoader(self, filePath):
        fileData = Pandas.read_csv(filePath);
        fileData.columns = fileData.columns.str.strip()  # Eliminar espacios en blanco
        print("Columnas disponibles:", fileData.columns.tolist());  # Imprimir las columnas
        X = fileData[['hours']].values;
        y = fileData['scores'].values;
        return X, y;


class LinearRegressionModel:
    def __init__(self):
        self.model = LinearRegression();


    def TrainModel(self, X_train, y_train):
        self.model.fit(X_train, y_train)


    def PredictScores(self, X):
        return self.model.predict(X);


    def Evaluate(self, X_test, y_test):
        y_predictedValues = self.PredictScores(X_test);
        meanSquaredError = MeanSquaredError(y_test, y_predictedValues);
        meanAbsoluteError = MeanAbsoluteError(y_test, y_predictedValues);
        r2 = r2_Score(y_test, y_predictedValues);
        print(f'Error Cuadrático Medio (MSE): {meanSquaredError:.2f}');
        print(f'Error Absoluto Medio (MAE): {meanSquaredError:.2f}');
        print(f'Coeficiente de Determinación (R^2): {r2:.2f}');


    def VisualizeResults(self, X_train, y_train):
        Matplotlib.scatter(X_train, y_train, color='blue', label='Datos de Entrenamiento');
        Matplotlib.plot(X_train, self.PredictScores(X_train), color='red', label='Línea de Regresión');
        Matplotlib.xlabel('Horas de Estudio');
        Matplotlib.ylabel('Calificación');
        Matplotlib.title('Horas de Estudio vs Calificación');
        Matplotlib.legend();
        Matplotlib.show();   

if __name__ == "__main__":
    # Paso 1: Cargar los datos desde el archivo CSV
    dataLoader = LoadData();
    X, y = dataLoader.DataLoader('C:/Users/osiam/Desktop/AI/Predicción de calificaciones basada en horas de estudio/horasDeEstudio.csv');

    # Paso 2: Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = TrainTestSplit(X, y, test_size = 0.2, random_state = 53);

    # Paso 3: Crear y entrenar el modelo
    model = LinearRegressionModel();
    model.TrainModel(X_train, y_train);

    # Paso 4: Evaluar el modelo
    model.Evaluate(X_test, y_test);

    # Paso 5: Visualizar resultados del entrenamiento
    model.VisualizeResults(X_train, y_train);

    # Paso 6: Hacer una predicción para un nuevo valor
    newHours = [[10]];  # Horas de estudio de un nuevo estudiante
    predictedScores = model.PredictScores(newHours);
    print(f'Predicción para un estudiante que estudia {newHours[0][0]} horas: {predictedScores[0]:.2f}');
