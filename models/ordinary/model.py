from models.resnet import ResNet


class ordResNet(ResNet):
    def __init__(self, depth, num_classes, in_planes=3):
        super().__init__(depth, num_classes, in_planes)
