

import torch
import torch.nn as nn

from pytorch_wavelets import DWT1DForward, DWT1DInverse

x = torch.randn(2, 3, 95)

# dwt = DWT1DForward(wave='db8', J=4, mode='symmetric')
dwt = DWT1DForward(wave='coif6', J=3, mode='symmetric')
# dwt = DWT1DForward(wave='haar', J=5, mode='symmetric')
pool = nn.MaxPool1d(kernel_size=2, stride=2)

# y_low, y_high = dwt(x)
# print("y_low:",y_low.shape)
x = x[:,:,-1][-1]
x = x.item()
#
#
print(x)   #  95 __> 47
#
# for i  in y_high:
#     print(i.shape)



# ##########Daubechies小波是一种正交小波基
# DWT1DForward(wave='db8', J=5, mode='symmetric')
# y_low: torch.Size([1, 1, 17])
# y_high: torch.Size([1, 1, 55]) torch.Size([1, 1, 35]) torch.Size([1, 1, 25])


# DWT1DForward(wave='db8', J=4, mode='symmetric')
# y_low: torch.Size([1, 1, 20])
# y_high: torch.Size([1, 1, 55]) torch.Size([1, 1, 35]) torch.Size([1, 1, 25])


# DWT1DForward(wave='db8', J=3, mode='symmetric')
# y_low: torch.Size([1, 1, 25])
# y_high: torch.Size([1, 1, 55]) torch.Size([1, 1, 35]) torch.Size([1, 1, 25])

# DWT1DForward(wave='db7', J=3, mode='symmetric')
# y_low: torch.Size([1, 1, 23])
# y_high: torch.Size([1, 1, 54]) torch.Size([1, 1, 33]) torch.Size([1, 1, 23])

# DWT1DForward(wave='db6', J=3, mode='symmetric')
# y_low: torch.Size([1, 1, 21])
# y_high: torch.Size([1, 1, 53]) torch.Size([1, 1, 32]) torch.Size([1, 1, 21])

# dwt = DWT1DForward(wave='db5', J=3, mode='symmetric')
# y_low: torch.Size([1, 1, 19])
# y_high: torch.Size([1, 1, 52]) torch.Size([1, 1, 30]) torch.Size([1, 1, 19])

# DWT1DForward(wave='db4', J=3, mode='symmetric')
# y_low: torch.Size([1, 1, 18])
# y_high: torch.Size([1, 1, 51]) torch.Size([1, 1, 29]) torch.Size([1, 1, 18])


############################################
# Coiflets小波基是对称的正交小波基
# DWT1DForward(wave='coif3', J=3, mode='symmetric')
# y_low: torch.Size([1, 1, 26])
# y_high: torch.Size([1, 1, 56]) torch.Size([1, 1, 36]) torch.Size([1, 1, 26])

# DWT1DForward(wave='coif4', J=3, mode='symmetric')
# y_low: torch.Size([1, 1, 32])
# y_high: torch.Size([1, 1, 59]) torch.Size([1, 1, 41]) torch.Size([1, 1, 32])

# DWT1DForward(wave='coif5', J=3, mode='symmetric')
# y_low: torch.Size([1, 1, 37])
# y_high: torch.Size([1, 1, 62]) torch.Size([1, 1, 45]) torch.Size([1, 1, 37])

# DWT1DForward(wave='coif6', J=3, mode='symmetric')
# y_low: torch.Size([1, 1, 42])
# y_high: torch.Size([1, 1, 65]) torch.Size([1, 1, 50]) torch.Size([1, 1, 42])

# DWT1DForward(wave='coif7', J=3, mode='symmetric')
# y_low: torch.Size([1, 1, 47])
# y_high: torch.Size([1, 1, 68]) torch.Size([1, 1, 54]) torch.Size([1, 1, 47])

# DWT1DForward(wave='coif6', J=4, mode='symmetric')
# y_low: torch.Size([1, 1, 38])
# y_high: torch.Size([1, 1, 65]) torch.Size([1, 1, 50]) torch.Size([1, 1, 42])