import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

# Импортируем твой класс
from src.masks import MaskCollatorMultiBlock


def visualize_masks(image_path="image_1.png"):
    # 1. Задаем параметры, идентичные твоему конфигу
    input_size = (32, 32)
    patch_size = 4
    nenc = 1
    npred = 2

    # Вычисляем размеры сетки патчей (для 32x32 и patch 4 -> 8x8)
    h_patches = input_size[0] // patch_size
    w_patches = input_size[1] // patch_size

    # 2. Инициализируем маск-коллатор
    collator = MaskCollatorMultiBlock(
        input_size=input_size,
        patch_size=patch_size,
        enc_mask_scale=(0.4, 0.6),  # твои обновленные масштабы
        pred_mask_scale=(0.15, 0.2),
        aspect_ratio=(0.75, 1.5),
        nenc=nenc,
        npred=npred,
        allow_overlap=False,  # можно поменять на False для теста
        min_keep=3,
    )

    # 3. Загружаем и подготавливаем изображение
    if not os.path.exists(image_path):
        # Если картинки нет, создадим случайный шум для теста, чтобы код не падал
        print(f"Файл {image_path} не найден. Генерируем случайный шум для теста.")
        img = Image.fromarray(
            np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        )
    else:
        img = Image.open(image_path).convert("RGB")

    # Преобразуем в тензор [3, 32, 32]
    transform = T.Compose([T.Resize(input_size), T.ToTensor()])
    img_tensor = transform(img)

    # Маск-коллатор ожидает на вход `batch`, то есть список.
    # Твой collator делает default_collate. Если в батче просто тензоры,
    # то результатом `collated_batch` будет сам этот скомбинированный тензор картинок.
    batch = [{'image': img_tensor}]

    # 4. Прогоняем через коллатор
    for _ in range(10 ** 5):
        output_dict = collator(batch)

    # Извлекаем маски для нулевого элемента батча
    # Индексы масок имеют форму: [B, nenc/npred, num_kept_patches]
    masks_enc = [mask[0] for mask in output_dict["masks_enc"]]  # [nenc, num_kept_patches]
    masks_pred = [mask[0] for mask in output_dict["masks_pred"]]  # [npred, num_kept_patches]

    # Функция для перевода одномерных индексов патчей в двумерную бинарную маску [8, 8]
    def indices_to_mask(indices):
        mask2d = torch.zeros(h_patches * w_patches, dtype=torch.float32)
        mask2d[indices.long()] = 1.0
        return mask2d.view(h_patches, w_patches).numpy()

    # 5. Настройка сетки отрисовки matplotlib (1 + nenc + npred)
    total_plots = 1 + nenc + npred
    fig, axes = plt.subplots(1, total_plots, figsize=(3 * total_plots, 3))

    # Переводим исходный тензор обратно в формат numpy [32, 32, 3] для отрисовки
    orig_img_np = img_tensor.permute(1, 2, 0).numpy()

    # Сначала рисуем оригинальное чистое изображение
    axes[0].imshow(orig_img_np)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    # Рисуем маски кодирования (Context Encoder Masks)
    for i in range(nenc):
        ax = axes[1 + i]
        mask2d = indices_to_mask(masks_enc[i])

        # Интерполируем маску 8x8 до размеров изображения 32x32 методом nearest
        mask_resized = np.repeat(np.repeat(mask2d, patch_size, axis=0), patch_size, axis=1)

        # Показываем полупрозрачную картинку, а маску накладываем сверху (выделенные патчи)
        ax.imshow(orig_img_np, alpha=0.4)
        ax.imshow(mask_resized, cmap="Reds", alpha=0.6 * mask_resized)
        ax.set_title(f"Enc Mask (Context) {i+1}")
        ax.axis("off")

    # Рисуем маски предсказания (Predictor Target Masks)
    for i in range(npred):
        ax = axes[1 + nenc + i]
        mask2d = indices_to_mask(masks_pred[i])

        mask_resized = np.repeat(np.repeat(mask2d, patch_size, axis=0), patch_size, axis=1)

        ax.imshow(orig_img_np, alpha=0.4)
        ax.imshow(mask_resized, cmap="Blues", alpha=0.6 * mask_resized)
        ax.set_title(f"Pred Mask (Target) {i+1}")
        ax.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    visualize_masks()
