import torchvision.transforms as T 


def get_transform(img_size: int = 512):
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5),
                    (0.5, 0.5, 0.5)),
    ])