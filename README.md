# Нейронная сеть
Нейросеть с простой архитектурой.

## Состав
- Модуль для работы с данными: `data/DataManager.py`
- Нейросеть: `structures/NeuralNetwork.py`
- Некоторые функции активации: `structures/functions.py`
- Слои: `structures/layers/`

## Способ использования
```sh
nn = NeuralNetwork()

nn.add_layer(DenseLayer(784, 500, sigmoid, d_sigmoid, alpha=0.01))
nn.add_layer(DenseLayer(500, 100, sigmoid, d_sigmoid))
nn.add_layer(DenseLayer(100, 10, same, d_same))

nn.train(train_set, epochs=20, epoch_range=1)

res = nn.feed_forward(example)
```

## Кастомизация
Для добавления своего слоя, унаследуйте его от AbstractLayer. Тогда его можно будет добавить в текущую нейросеть.

## Планы
- [ ] Свёрточный слой
- [ ] LSTM