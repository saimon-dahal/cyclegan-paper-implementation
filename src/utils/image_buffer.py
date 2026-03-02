import random
import torch


class ImageBuffer:
    """
    Buffer of previously generated images.
    Returns generated images from buffer with 50% probability.
    This helps reduce model oscillation.
    """
    
    def __init__(self, buffer_size=50):
        self.buffer_size = buffer_size
        self.buffer = []
    
    def push_and_pop(self, images):
        to_return = []
        for image in images:
            image = torch.unsqueeze(image.data, 0)
            
            if len(self.buffer) < self.buffer_size:
                self.buffer.append(image)
                to_return.append(image)
            else:
                if random.uniform(0, 1) > 0.5:
                    i = random.randint(0, self.buffer_size - 1)
                    to_return.append(self.buffer[i].clone())
                    self.buffer[i] = image
                else:
                    to_return.append(image)
        
        return torch.cat(to_return, 0)
