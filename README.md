# SyntheticAugmentationGenerator
**_Библиотека для аугментации изображений путём генерации заданных объектов при помощи диффузионных моделей_**

![](content/synt_generator.png)

Для установки библиотеки запустите данный код:
```console
git clone https://github.com/t041lk8/synthetic-augmentation-generator
cd synthetic-augmentation-generator
pip install .
cd ..
```

## Использование
SyntheticAugmentationGenerator позволяет собрать датасет в формате COCO, добавив на исходные изображения объекты, описанные при помощи текстовых подсказок, в случайно выбранных ббоксах.

Для работы с генератором необходимо создать JSON файл в формате:

    {
        "classes": [
            {
                "id": 0,
                "label": "label_0",
                "prompts": [
                    "prompt №1 for label_0",
                    "prompt №2 for label_0",
                    ...
                ]
            },
            ...
        ],
        "bboxs": [
            [x_min_0, y_min_0, x_max_0, y_max_0],
            [x_min_1, y_min_1, x_max_1, y_max_1],
            ...
        ]
    }

На выходе пользователь получит аугментированные изображения и разметку в формате COCO.

Данный генератор можно использовать, импортировав его из библиотеки syntgenerator:
```python
from syntgenerator import AugmentationGenerator

generator = AugmentationGenerator({initial_args})
generator({call_args})
```

Примеры использования генератора и дообучения Stable Diffusion Inpaint можно посмотреть в ноутбуках [GeneratorUsageExample](notebooks/GeneratorUsageExample.ipynb) и [StableDiffusionFinetuning](notebooks/StableDiffusionFinetuning.ipynb).

## Аргументы для использования
### Инициализация
|Аргумент|Тип|Описание|
|-|-|-|
|source_json|str, os.PathLike|Путь к JSON файлу с ббоксами и промптами|
|final_json|str, os.PathLike|Путь к JSON файлу в формате COCO|
|dir_images|str, os.PathLike|Путь к оригинальным изображениям|
|dir_dataset|str, os.PathLike|Путь к аугментированным изображениям|
|pipeline|StableDiffusionInpaintPipeline, AutoPipelineForInpainting|Inpaint пайплайн из библиотеки *diffusers* для модели|

### Вызов
|Аргумент|Тип|Значение по умолчанию|Описание|
|-|-|-|-|
|guidance_scale|float|10|Более высокое значение побуждает модель генерировать изображения, тесно связанные с текстовой подсказкой, за счет более низкого качества изображения|
|num_inference_steps|int|50|Количество шагов по снижению шума. Большее количество шагов по снижению шума обычно приводит к более высокому качеству изображения за счет более медленного инференса|
|bb_num|int|1|Количество ббоксов для каждого класса на одном изображении|
|negative_prompt|str|None|Подсказка, указывающая, что не следует включать в генерацию изображений|
|increase_scale|float|1.2|Этот параметр отвечает за значение, на которое умножается размер bbox для выделения области внимания. Этот параметр должен быть больше 1. Данный параметр проигнорируется, если параметр aa_size будет иметь значение отличное от None.|
|aa_size|int|None|Этот параметр отвечает за размер области внимания.|

## Аргументы для обучения
### Инициализация
|Аргумент|Тип|Описание|
|-|-|-|
|pretrained_model_name_or_path|str, os.PathLike|Название модели с huggingface или путь до весов для обучения.|
|output_dir|str, os.PathLike|Путь до места сохранения новых весов.|

### Вызов
|Аргумент|Тип|Значение по умолчанию|Описание|
|-|-|-|-|
|data_dir|str|None|Путь до обучающего датасета|
|train_batch_size|int|1|Размер батча|
|max_train_steps|int|400|Общее количество шагов обучения, которые необходимо выполнить|
|resolution|int|512|Разрешение входного изображения|
|lr|float|5e-6|Начальное значение learning rate для обучения|
|betas|tuple|(0.9, 0.999)|Параметры бета для оптимизатора Adam|
|eps|float|1e-08|Параметр эпсилон для оптимизатора Adam|
|checkpoint_save|int|500|Период сохранения чекпоинтов модели|