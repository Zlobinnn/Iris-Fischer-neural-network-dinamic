#include <iostream>
#include <math.h>
#include <fstream>
#include <vector>
#include <string>
#include <ctime>

using namespace std;

class Neuron_Web {
public:
    vector <vector <vector <float>>> l;
    vector <vector <vector <float>>> w;
    vector <vector <float>> l_error;

    Neuron_Web(vector <int> new_layers) {
        for (int i = 0; i < new_layers.size(); i++) {
            vector <vector <float>> new_layer;
            for (int k = 0; k < new_layers[i]; k++) {
                new_layer.push_back({ 0, 0 });
            }
            if (i != new_layers.size() - 1) {
                new_layer.push_back({ 1, 0 });
            }
            l.push_back(new_layer);
        }
        l_print();
        w_rand();
    }

    void go(vector <float> a, vector <float> b) { // Выполняем проход по нейронной сети с входными параметрами (a) и со значениями, который должны быть на выходе (b)
        for (int i = 0; i < l[0].size() - 1; i++) {
            l[0][i][0] = a[i];
        }

        for (int i = 0; i < l.size() - 1; i++) {
            probeg(l[i], l[i + 1], w[i], l[i].size(), l[i + 1].size() - 1);
        }
        probeg(l[l.size() - 2], l[l.size() - 1], w[l.size() - 2], l[l.size() - 2].size(), l[l.size() - 1].size());

        for (int i = 0; i < l[l.size()-1].size(); i++) {
            l[l.size() - 1][i][1] = b[i] - l[l.size() - 1][i][0]; // Получаем значения ошибок на выходном слое
        }

        error(l[l.size() - 2], l[l.size() - 1], w[l.size() - 2], l[l.size() - 2].size(), l[l.size() - 1].size()); // Получаем значения ошибок на всех остальных слоях
        for (int i = l.size() - 3; i >= 0; i--) {
            error(l[i], l[i + 1], w[i], l[i].size(), l[i + 1].size() - 1);
        }

        //error(b);

        for (int i = 0; i < l.size() - 1; i++) {
            fix(l[i], l[i + 1], w[i], l[i].size(), l[i + 1].size() - 1);
        }
        fix(l[l.size() - 2], l[l.size() - 1], w[l.size() - 2], l[l.size() - 2].size(), l[l.size() - 1].size());
    }

    float go_check_error(vector <float> a, vector <float> b) { // То же самое, что и метод void go(float* a, float* b), только этот метод возвращает сумму квадратов ошибок на выходе
        for (int i = 0; i < l[0].size() - 1; i++) {
            l[0][i][0] = a[i];
        }

        for (int i = 0; i < l.size() - 1; i++) {
            probeg(l[i], l[i + 1], w[i], l[i].size(), l[i + 1].size() - 1);
        }
        probeg(l[l.size() - 2], l[l.size() - 1], w[l.size() - 2], l[l.size() - 2].size(), l[l.size() - 1].size());

        for (int i = 0; i < l[l.size() - 1].size(); i++) {
            l[l.size() - 1][i][1] = b[i] - l[l.size() - 1][i][0]; // Получаем значения ошибок на выходном слое
        }

        error(l[l.size() - 2], l[l.size() - 1], w[l.size() - 2], l[l.size() - 2].size(), l[l.size() - 1].size()); // Получаем значения ошибок на всех остальных слоях
        for (int i = l.size() - 3; i >= 0; i--) {
            error(l[i], l[i + 1], w[i], l[i].size(), l[i + 1].size() - 1);
        }

        //error(b);

        for (int i = 0; i < l.size() - 1; i++) {
            fix(l[i], l[i + 1], w[i], l[i].size(), l[i + 1].size() - 1);
        }
        fix(l[l.size() - 2], l[l.size() - 1], w[l.size() - 2], l[l.size() - 2].size(), l[l.size() - 1].size());

        float error = 0;
        for (int i = 0; i < l[l.size() - 1].size(); i++) {
            error += l[l.size() - 1][i][1] * l[l.size() - 1][i][1];
        }
        return error;
    }

    void go(vector <float> a) { // Выполняем проход по нейросети без исправления весов
        for (int i = 0; i < l[0].size() - 1; i++) {
            l[0][i][0] = a[i];
        }

        for (int i = 0; i < l.size()-1; i++) {
            probeg(l[i], l[i + 1], w[i], l[i].size(), l[i+1].size()-1);
        }
        probeg(l[l.size() - 2], l[l.size() - 1], w[l.size() - 2], l[l.size() - 2].size(), l[l.size() - 1].size());
    }

    void error(vector <vector <float>>& x1, vector <vector <float>>& x2, vector <vector <float>>& w, int size1, int size2) { // Вычисление ошибки для матрицы весов
        for (int i = 0; i < size1; i++) {
            x1[i][1] = 0;
            for (int k = 0; k < size2; k++) {
                x1[i][1] += x2[k][1] * w[i][k] * x1[i][0] * (1 - x1[i][0]);
            }
        }
    }

    void probeg(vector <vector <float>> &x1, vector <vector <float>> &x2, vector <vector <float>> &w, int size1, int size2) { // Вычисление значений для слоя
        for (int i = 0; i < size2; i++) {
            x2[i][0] = 0;
            for (int k = 0; k < size1; k++) {
                x2[i][0] += x1[k][0] * w[k][i];
            }
            x2[i][0] = 1 / (1 + exp(-1 * x2[i][0])); // Использую логистическую функцию активации
        }
    }

    void fix(vector <vector <float>> &x1, vector <vector <float>> &x2, vector <vector <float>> &w, int size1, int size2) { // Корректировка весов в зависимости от ошибки
        for (int i = 0; i < size1; i++) {
            for (int k = 0; k < size2; k++) {
                //w[k + i * size2] += 0.1 * x2[k * 2 + 1] * x1[i * 2] * x2[k * 2] * (1 - x2[k * 2]); // Корректировка весов для логистической функции активации
                w[i][k] += 0.1 * x2[k][1] * x1[i][0];
            }
        }
    }

    void w_print() {
        for (int i = 0; i < w.size(); i++) {
            for (int k = 0; k < w[i].size(); k++) {
                for (int j = 0; j < w[i][k].size(); j++) {
                    cout << w[i][k][j] << "\t";
                }
                cout << endl;
            }
            cout << endl << endl;
        }
    }

    void l_print() {
        for (int i = 0; i < l.size(); i++) {
            for (int k = 0; k < l[i].size(); k++) {
                cout << l[i][k][0] << " " << l[i][k][1] << "\t";
            }
            cout << endl;
        }
    }

    void w_rand() {
        for (int i = 0; i < l.size() - 1; i++) {
            vector <vector <float>> new_layer;
            for (int k = 0; k < l[i].size(); k++) {
                vector <float> new_l;
                for (int j = 0; j < l[i + 1].size() - 1; j++) {
                    float r = (rand() % 100);
                    new_l.push_back(r / 100);
                }
                if (i + 1 == l.size() - 1) {
                    float r = (rand() % 100);
                    new_l.push_back(r / 100);
                }
                new_layer.push_back(new_l);
            }
            w.push_back(new_layer);
        }
        w_print();
    }
};



int learning(Neuron_Web& web) { // Обучение нейросети
    ifstream file;
    file.open("file.txt"); // Подключаем датасет

    if (!file) {
        cout << "Ошибка открытия файла\n";
        return 0;
    }

    vector <vector <float>> data;
    string name;
    float value;

    while (!file.eof()) { // Сохраняем в матрицу data данные для обучения
        vector <float> values;
        value = 0;
        name = "";
        for (int i = 0; i < 4; i++) { // Считываем входные параметры
            file >> value;
            values.push_back(value);
        }
        file >> name;
        if (name == "setosa") { // Считываем вид Ириса для считанных входных параметров
            values.push_back(0.8);
            values.push_back(0.2);
            values.push_back(0.2);
        }
        else if (name == "versicolor") {
            values.push_back(0.2);
            values.push_back(0.8);
            values.push_back(0.2);
        }
        else if (name == "virginica") {
            values.push_back(0.2);
            values.push_back(0.2);
            values.push_back(0.8);
        }
        else if (name != "") {
            cout << "Ошибка в названии растения\n";
            return 0;
        }
        data.push_back(values);
    }
    file.close();
    for (int i = 0; i < data.size(); i++) { // Выводим на экран матрицу данных для обучения
        for (int k = 0; k < data[i].size(); k++) {
            cout << data[i][k] << "\t";
        }
        cout << "\n";
    }

    float error = 1;
    int j = 0;

    vector <float> a = { 0, 0, 0, 0 };
    vector <float> b = { 0.2, 0.2, 0.2 };

    while (error >= 0.0001) { // Обучаем нейросеть, пока сумма квадратов ошибок на выходных нейронах по всем примерам не меньше, чем заданное значение ошибки
        j++;
        error = 0;
        for (int i = 0; i < data.size(); i++) {
            for (int k = 0; k < 4; k++)
                a[k] = data[i][k];
            for (int k = 0; k < 3; k++)
                b[k] = data[i][k + 4];
            error += web.go_check_error(a, b);
        }

        if (j % 10000 == 0) cout << error << endl; // Выводим в консоль сумму квадратов ошибок каждой 100.000-ой эпохи
    }

    cout << "Кол-во эпох: " << j << endl;

    return 1;
}

int main()
{
    setlocale(LC_ALL, "RU");
    srand(time(0));

    Neuron_Web web({ 4, 4, 3, 3 }); // Создаём нейросеть

    vector <float> a = { 0, 0, 0, 0 };
    vector <float> b = { 0.2, 0.2, 0.2 };
    web.go(a, b); // Тестовый проход по нейросети

    if (!learning(web)) { // Обучаем нейросеть
        cout << "Обучение не было завершено\n";
        return 0;
    }

    web.w_print(); // Выводим значения весов и слоёв
    web.l_print();

    cout << "\nОбучение завершено\n\n\n\n\n\n\n";

    while (true) { // Пользуемся обученной нейросетью
        cout << "Введите данные: "; // В коносль вводим входные данные через пробел либо через tab, например: 4.8 3.4 1.6 0.2
        for (int i = 0; i < 4; i++) {
            cin >> a[i];
        }

        web.go(a); // Пробегаем по нейросети с введёнными входными данными
        cout << web.l[web.l.size() - 1][0][0] << " " << web.l[web.l.size() - 1][0][1] << endl;
        cout << web.l[web.l.size() - 1][1][0] << " " << web.l[web.l.size() - 1][1][1] << endl;
        cout << web.l[web.l.size() - 1][2][0] << " " << web.l[web.l.size() - 1][2][1] << endl;

        if (web.l[web.l.size() - 1][0][0] > 0.7) cout << "Setosa\n\n"; // Проверяем, под какой вид попадает предположение нейросети (я обучал её так, чтобы на выходном нейроне неправильного вида было значение 0.2, а правильного - 0.8)
        else if (web.l[web.l.size() - 1][1][0] > 0.7) cout << "Versicolor\n\n";
        else if (web.l[web.l.size() - 1][2][0] > 0.7) cout << "Virginica\n\n";
        else cout << "Не удалось однозначно определить вид Ириса\n\n";
    }
}