import numpy as np

# 数据加载器 创建可迭代对象
class Dataloader(): 
    def __init__(self, images, labels, batch_size):
        self.images = images
        self.labels = labels
        self.batch_size = batch_size
        self.index = 0
        self.indices = np.arange(len(images))

    # 返回迭代器对象本身
    def __iter__(self):
        return self

    # 返回序列的下一个元素
    def __next__(self):
        # 当遍历完所有元素后，__next__抛出StopIteration异常
        if self.index >= len(self.images):
            self.index = 0
            np.random.shuffle(self.indices)
            raise StopIteration

        image_batch = self.images[self.indices[self.index:self.index+self.batch_size]]
        label_batch = self.labels[self.indices[self.index:self.index+self.batch_size]]
        self.index += self.batch_size
        return image_batch, label_batch
