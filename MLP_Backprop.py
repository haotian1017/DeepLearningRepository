import torch
import numpy as np

class MLP:
    def __init__(
        self,
        linear_1_in_features,
        linear_1_out_features,
        f_function,
        linear_2_in_features,
        linear_2_out_features,
        g_function
    ):
        """
        Args:
            linear_1_in_features: the in features of first linear layer
            linear_1_out_features: the out features of first linear layer
            linear_2_in_features: the in features of second linear layer
            linear_2_out_features: the out features of second linear layer
            f_function: string for the f function: relu | sigmoid | identity
            g_function: string for the g function: relu | sigmoid | identity
        """
        self.f_function = f_function
        self.g_function = g_function

        self.parameters = dict(
            W1 = torch.randn(linear_1_out_features, linear_1_in_features),
            b1 = torch.randn(linear_1_out_features),
            W2 = torch.randn(linear_2_out_features, linear_2_in_features),
            b2 = torch.randn(linear_2_out_features),
        )
        self.grads = dict(
            dJdW1 = torch.zeros(linear_1_out_features, linear_1_in_features),
            dJdb1 = torch.zeros(linear_1_out_features),
            dJdW2 = torch.zeros(linear_2_out_features, linear_2_in_features),
            dJdb2 = torch.zeros(linear_2_out_features),
        )

        # put all the cache value you need in self.cache
        self.cache = dict()
        self.activation = dict(
            relu = torch.nn.ReLU(),
            sigmoid = torch.nn.Sigmoid(),
            identity = torch.nn.Identity()
        )
        self.partial = dict(
            relu = lambda x: torch.tensor((x >0)*1.0),
            identity = lambda x: torch.ones([x.shape[0],x.shape[1]]),
            sigmoid = lambda x: torch.nn.Sigmoid()(x)*(1-torch.nn.Sigmoid()(x))
        )
    def forward(self, x):
        """
        Args:
            x: tensor shape (batch_size, linear_1_in_features)
        """
        # TODO: Implement the forward function
        self.cache["x"] = x
        z1 = torch.mm(x,self.parameters["W1"].T)+self.parameters["b1"]
        self.cache["z1"] = z1

        z2 = self.activation[self.f_function](z1)
        self.cache["z2"] = z2

        z3 = torch.mm(z2, self.parameters["W2"].T)+self.parameters["b2"]
        self.cache["z3"] = z3
        y_hat = self.activation[self.g_function](z3)
        pass
        return y_hat
    def backward(self, dJdy_hat):
        """
        Args:
            dJdy_hat: The gradient tensor of shape (batch_size, linear_2_out_features)
        """
        # TODO: Implement the backward function
        pass
        dydz3 = self.partial[self.g_function](self.cache["z3"])
        dldz3 = dJdy_hat * dydz3
        dz2dz1 = self.partial[self.f_function](self.cache["z1"])
        dldz1 = dldz3 * self.parameters["W2"] * dz2dz1

        self.grads["dJdb2"] =  torch.sum(dldz3).unsqueeze(0)
        self.grads["dJdW2"] =  torch.sum(dldz3 * self.cache["z2"],dim=0).unsqueeze(0)
        self.grads["dJdb1"] =  torch.sum(dldz1, dim=0)
        self.grads["dJdW1"] =  torch.mm(dldz1.T, self.cache["x"])
    
    def clear_grad_and_cache(self):
        for grad in self.grads:
            self.grads[grad].zero_()
        self.cache = dict()

def mse_loss(y, y_hat):
    """
    Args:
        y: the label tensor (batch_size, linear_2_out_features)
        y_hat: the prediction tensor (batch_size, linear_2_out_features)

    Return:
        J: scalar of loss
        dJdy_hat: The gradient tensor of shape (batch_size, linear_2_out_features)
    """
    # TODO: Implement the mse loss
    loss=torch.mean((y_hat - y)**2/2)
    dJdy_hat = (y_hat - y)/y.shape[0]
    return loss, dJdy_hat
    pass


    # return loss, dJdy_hat

def bce_loss(y, y_hat):
    """
    Args:
        y_hat: the prediction tensor
        y: the label tensor
        
    Return:
        loss: scalar of loss
        dJdy_hat: The gradient tensor of shape (batch_size, linear_2_out_features)
    """
    # TODO: Implement the bce loss
    pass
    loss = torch.mean((-(y*torch.log(y_hat) + (1-y)*torch.log(1-y_hat))))
    dJdy_hat = (-y/(y_hat) + (1-y)/(1-y_hat))/y.shape[0]
    return loss, dJdy_hat

    # return loss, dJdy_hat











