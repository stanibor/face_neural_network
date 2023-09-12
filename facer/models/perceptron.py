from torch import nn


class MLPerceptron(nn.Module):

    @staticmethod
    def _layer(input_size, output_size):
        return nn.Sequential(nn.Linear(input_size, output_size), nn.LeakyReLU())

    def __init__(self, input_size, output_size, hidden_sizes=[1024]):
        super().__init__()
        sizes = [input_size] + hidden_sizes
        self.layers = nn.Sequential(*[self._layer(i, o) for i,o in zip(sizes, sizes[1:])])
        self.last_layer = nn.Linear(sizes[-1], output_size)

    def _forward_impl(self, x):
        x = self.layers(x)
        return self.last_layer(x)

    def forward(self, x):
        return self._forward_impl(x)